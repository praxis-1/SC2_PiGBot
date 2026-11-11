import math
import numpy as np

from sc2.position import Point2, Point3
from sc2.units import Units
from sc2.unit import Unit
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.data import Race
from ares.consts import LOSS_DECISIVE_OR_WORSE, TIE_OR_BETTER, UnitTreeQueryType, VICTORY_DECISIVE_OR_BETTER, VICTORY_MARGINAL_OR_BETTER, LOSS_OVERWHELMING_OR_WORSE, WORKER_TYPES

from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    KeepUnitSafe, PathUnitToTarget, StutterUnitBack, AMove, ShootTargetInRange
)
from ares.behaviors.combat.group import PathGroupToTarget
from ares.managers.squad_manager import UnitSquad

from ares.consts import UnitRole

from bot.managers.reactions import assess_threat
from bot.constants import (
    ATTACKING_SQUAD_RADIUS,
    DEFENDER_SQUAD_RADIUS,
    COMMON_UNIT_IGNORE_TYPES,
    DISRUPTOR_IGNORE_TYPES,
    MELEE_RANGE_THRESHOLD,
    STAY_AGGRESSIVE_DURATION,
    UNSAFE_GROUND_CHECK_RADIUS,
    DISRUPTOR_SQUAD_FOLLOW_DISTANCE,
    DISRUPTOR_SQUAD_TARGET_DISTANCE,
    DISRUPTOR_FORWARD_OFFSET,
    STRUCTURE_ATTACK_RANGE,
    PROXIMITY_STICKY_DISTANCE_SQ,
    MAP_CROSSING_DISTANCE_SQ,
    MAP_CROSSING_SUCCESS_DISTANCE,
    UNIT_ENEMY_DETECTION_RANGE,
    GATEKEEPER_DETECTION_RANGE,
    GATEKEEPER_ZEALOT_RADIUS,
    GATEKEEPER_CLEARANCE_MARGIN,
    GATEKEEPER_PARK_DISTANCES,
    GATEKEEPER_PARK_MIN_DISTANCE,
    GATEKEEPER_PARK_LATERAL_OFFSETS,
    GATEKEEPER_CLEARANCE_SEARCH_RADIUS,
    GATEKEEPER_POSITION_TOLERANCE,
    GATEKEEPER_ORDER_TOLERANCE,
    WARP_PRISM_FOLLOW_DISTANCE,
    WARP_PRISM_FOLLOW_OFFSET,
    WARP_PRISM_UNIT_CHECK_RANGE,
    WARP_PRISM_DANGER_DISTANCE,
    WARP_PRISM_MIN_ENEMY_DISTANCE,
    WARP_PRISM_MATRIX_RADIUS,
    WARP_PRISM_POSITION_SEARCH_RANGE,
    WARP_PRISM_POSITION_SEARCH_STEP,
    EARLY_GAME_TIME_LIMIT,
    EARLY_GAME_SAFE_GROUND_CHECK_BASES,
    SQUAD_NEARBY_FRIENDLY_RANGE_SQ,
    STALE_INTEL_THRESHOLD,
    FORMATION_AHEAD_BASE,
    FORMATION_AHEAD_SCALE,
    FORMATION_SPREAD_BASE,
    FORMATION_SPREAD_SCALE,
    FORMATION_UNIT_MULTIPLIER,
    FORMATION_RETREAT_ANGLE,
    ENGAGEMENT_ARMY_VALUE_THRESHOLD,
    ACTIVE_ENGAGE_RANGE_BUFFER,
    ACTIVE_ENGAGE_ANGLE,
    CHOKE_SAMPLE_POINTS,
    CHOKE_MELEE_DPS_THRESHOLD,
    CHOKE_MELEE_RANGE,
    CHOKE_MIN_ARMY_WIDTH,
    CHOKE_RETREAT_DIST,
    HT_SQUAD_FOLLOW_DISTANCE,
    HT_SQUAD_TARGET_DISTANCE,
    MASS_RECALL_RADIUS,
    MASS_RECALL_MIN_OWN_SUPPLY,
    MASS_RECALL_RETREAT_SEARCH_RADIUS,
)
from bot.combat.unit_micro import (
    micro_ranged_unit,
    micro_melee_unit,
    micro_disruptor,
    micro_high_templar,
    merge_high_templars,
    micro_sentry,
    micro_stalker,
    micro_air_unit,
)
from bot.combat.formation import execute_fan_out, clear_formation_state
from bot.combat.scoring_table import select_target, update_upgrades, update_repairer_tags
from bot.combat.force_field import compute_ff_split, compute_ff_main_ramp_block, compute_ff_choke_block
from bot.utilities.choke_grid import get_or_refine_choke, detect_dynamic_choke
from bot.combat.group_snipe import try_commit_snipe, execute_snipe_a, execute_snipe_b, execute_focus
from bot.combat.group_chase import try_commit_chase, execute_chase
from ares.dicts.unit_data import UNIT_DATA
from bot.utilities.debug import (
    render_combat_state_overlay,
    render_mass_recall_debug,
    render_disruptor_labels,
    render_nova_labels,
    render_base_defender_debug,
    render_target_scoring_debug,
    render_observer_debug,

    log_nova_error,
    render_formation_debug,
    render_concave_formation_debug,
    render_blind_ramp_debug,
    render_choke_policy_debug,
    render_choke_decision_debug,
    render_ff_split_debug,
    render_refined_choke_debug,
    render_gs_debug,
    render_snipe_debug,
    render_chase_debug,
    render_focus_debug,
    render_micro_state_debug,
    render_repair_debug,
)
from bot.intel import get_enemy_intel_quality, get_sim_static_defense
from bot.managers.structure_manager import use_mass_recall
from bot.utilities.telemetry import log_transition

from cython_extensions import (
    cy_pick_enemy_target, cy_closest_to, cy_distance_to, cy_distance_to_squared, cy_in_attack_range, cy_find_units_center_mass,
    cy_adjust_moving_formation, cy_is_facing
)
from cython_extensions.general_utils import cy_in_pathing_grid_ma
from cython_extensions.numpy_helper import cy_point_below_value


def _circular_mask(center: Point2, radius: float, shape: tuple) -> np.ndarray:
    """Boolean mask of cells within `radius` of `center`, shaped to `shape`.

    Mirrors the NovaManager.get_exclusion_mask pattern (nova_manager.py:335).
    Used to zero out shielded areas on the GS density grid.
    Grid is indexed [x, y] (ARES convention), so ogrid axis 0 = x, axis 1 = y.
    """
    x_idx, y_idx = np.ogrid[:shape[0], :shape[1]]
    return ((x_idx - center.x) ** 2 + (y_idx - center.y) ** 2) <= radius ** 2


def _compute_guardian_shield_assignments(
    sentries: list[Unit],
    squad_units: list[Unit],
    enemies: Units,
    bot,
) -> tuple[set[int], dict[int, Point2]]:
    """Determine which Sentries should cast Guardian Shield and where each should go.

    Builds a per-squad friendly-density influence grid, then uses a
    Sentry-centric assignment: each Sentry casts where it currently is, NOT
    sent across the army to a distant pocket. Sending every Sentry to a
    different pocket of a spread-out army makes them all leave the squad —
    the influence map's job is just to tell whether there are uncovered
    friendlies near each Sentry, not to scatter Sentries across the army.

    Deployment rules:
      1. Range is the trigger — no ranged enemy, no cast.
      2. No redundant deployment — a Sentry casts only if there are
         uncovered friendlies within GS radius of its current position.
      3. Overlap handling — a Sentry standing on an active shield (overlap)
         is assigned the nearest uncovered pocket so it paths away, but
         does NOT cast until it arrives. This mirrors the Disruptor Nova
         exclusion-mask pattern.

    Returns:
        (approved_tags, peaks) where peaks maps each sentry tag to
        the position it should path to (its own position for in-place casts,
        or the nearest uncovered pocket for overlapping Sentries repositioning).
    """
    from bot.constants import (
        GUARDIAN_SHIELD_ENERGY_COST,
        GUARDIAN_SHIELD_RADIUS,
        GUARDIAN_SHIELD_OVERLAP_DISTANCE,
        GS_INFLUENCE_RADIUS,
        GS_IGNORE_TYPES,
        MELEE_RANGE_THRESHOLD,
    )
    from sc2.ids.buff_id import BuffId

    # Rule 1: Range is the trigger. No ranged enemy → no GS.
    has_ranged_enemies = any(
        u.ground_range > MELEE_RANGE_THRESHOLD for u in enemies
    )
    if not has_ranged_enemies:
        return set(), {}

    # Candidates: sentries that can cast (enough energy, not already shielded)
    candidates = [
        s for s in sentries
        if s.energy >= GUARDIAN_SHIELD_ENERGY_COST
        and not s.has_buff(BuffId.GUARDIANSHIELD)
    ]

    # Already-shielded sentries provide free coverage — they count toward
    # full coverage but don't need to be in the candidate pool.
    already_shielded = [s for s in sentries if s.has_buff(BuffId.GUARDIANSHIELD)]

    # If no candidates can cast, nothing to assign (existing shields persist).
    if not candidates:
        return set(), {}

    # Build the friendly-density grid on a zero baseline so that:
    #   - unpainted cells = 0.0 (no friendly value there)
    #   - painted cells  = army_value (the unit's worth)
    #   - zeroed cells   = 0.0 (already covered by an active shield)
    # This makes density.max() > 0 a correct "uncovered pocket exists" check.
    # safe=False because we're not pathing on this grid (only argmax + masking),
    # so the <1.0 clamp meant for PyAStar would corrupt our zero baseline.
    map_data = bot.mediator.get_map_data_object
    base = map_data.get_pyastar_grid()
    density = np.zeros_like(base)

    for u in squad_units:
        if u.type_id in GS_IGNORE_TYPES or u.is_hallucination or u.is_flying:
            continue
        value = UNIT_DATA.get(u.type_id, {}).get("army_value", 1.0)
        density = map_data.add_cost(
            position=u.position,
            radius=GS_INFLUENCE_RADIUS,
            grid=density,
            weight=value,
            safe=False,
        )

    # Zero out areas already covered by active shields (free coverage sources).
    # This is the exclusion-mask step: covered pockets drop to zero so the next
    # Sentry knows those areas don't need another shield.
    for s in already_shielded:
        mask = _circular_mask(s.position, GUARDIAN_SHIELD_RADIUS, density.shape)
        density[mask] = 0.0

    approved: set[int] = set()
    peaks: dict[int, Point2] = {}
    approved_positions: list[Point2] = []

    # Sentry-centric assignment: each Sentry casts where it is, NOT sent across
    # the army to a distant pocket. Sending every Sentry to a different pocket
    # of a spread-out army makes them all leave the squad — the cure is worse
    # than the disease. Instead:
    #
    # 1. Check if the Sentry's current position has uncovered density nearby
    #    (is there friendly value within GS radius that isn't already shielded?).
    # 2. If yes → approve the cast at the Sentry's current position. It's
    #    already in the army, covering units around it.
    # 3. If no but the Sentry is too close to an active shield → find the
    #    nearest uncovered pocket and assign it as a move target (so it paths
    #    away from the overlap), but don't approve casting until it arrives.
    # 4. If no and the Sentry isn't near any shield → skip it (nothing to cover
    #    where it is, and we don't send it wandering).
    #
    # This keeps Sentries with the squad. Only overlapping Sentries reposition.
    #
    # Same-frame overlap guard: when Sentries are stacked (common on the first
    # engagement — they all follow ranged_center), their local masks overlap
    # but aren't identical. Edge cells of the second Sentry's mask can still
    # have density > 0 after the first Sentry's zeroing, causing both to cast
    # simultaneously. The geographic zeroing alone can't prevent this because
    # the masks are circles centered on different (but nearby) positions.
    # The overlap distance check below is the per-Sentry gate that fixes this:
    # once a Sentry is approved this frame, any other candidate within
    # GUARDIAN_SHIELD_OVERLAP_DISTANCE is skipped. Next frame the first
    # Sentry's buff makes it a free coverage source on the density grid, and
    # the influence map naturally decides if a second Sentry is needed.
    for s in candidates:
        # Skip if too close to a Sentry already approved this frame — their
        # shields would overlap heavily, so only one should cast. The rest
        # will re-evaluate next frame once the first shield is active.
        if any(
            cy_distance_to(s.position, ap) < GUARDIAN_SHIELD_OVERLAP_DISTANCE
            for ap in approved_positions
        ):
            continue

        # Density within GS radius of this Sentry's current position
        local_mask = _circular_mask(s.position, GUARDIAN_SHIELD_RADIUS, density.shape)
        local_density = density[local_mask].max() if density[local_mask].size > 0 else 0.0

        if local_density > 0:
            # There are uncovered friendlies around this Sentry → cast here.
            approved.add(s.tag)
            peaks[s.tag] = s.position
            approved_positions.append(s.position)

            # Zero out the area this Sentry will cover so the next Sentry
            # only casts if there's still uncovered density near *it*.
            density[local_mask] = 0.0
        else:
            # No uncovered density near this Sentry. Is it overlapping an
            # existing shield or a Sentry just approved this frame? If so,
            # assign it the nearest uncovered pocket so it paths away — but
            # don't approve casting this frame.
            too_close = any(
                cy_distance_to(s.position, sh.position) < GUARDIAN_SHIELD_OVERLAP_DISTANCE
                for sh in already_shielded
            ) or any(
                cy_distance_to(s.position, ap) < GUARDIAN_SHIELD_OVERLAP_DISTANCE
                for ap in approved_positions
            )
            if too_close and density.max() > 0:
                # Find the nearest uncovered pocket to this Sentry (not the
                # densest globally — the nearest, so it doesn't run across
                # the map). Use the density grid: find the max-value cell
                # within a reasonable radius of the Sentry, not globally.
                search_radius = GUARDIAN_SHIELD_OVERLAP_DISTANCE * 2
                search_mask = _circular_mask(s.position, search_radius, density.shape)
                search_density = density.copy()
                search_density[~search_mask] = 0.0
                if search_density.max() > 0:
                    flat_idx = int(np.argmax(search_density))
                    peak_x = flat_idx // density.shape[1]
                    peak_y = flat_idx % density.shape[1]
                    peaks[s.tag] = Point2((peak_x, peak_y))
                    # Don't zero — this Sentry isn't casting yet, so the
                    # pocket is still uncovered for next frame.

    # Stash debug state for render_gs_debug() to pick up. No per-frame cost
    # when bot.debug is False — this block is a plain attribute assignment.
    if getattr(bot, "debug", False):
        bot._gs_debug_density = density
        bot._gs_debug_peaks = peaks

    return approved, peaks


def get_attackable_enemies(unit: Unit, enemies: list, grid: np.ndarray) -> list:
    """
    Filter enemies to only those this unit can attack AND reach.
    
    Checks:
    1. Attackability: can unit attack air/ground targets?
    2. Reachability: for ground units, is enemy position pathable?
    
    Args:
        unit: The unit doing the attacking
        enemies: List of potential enemy targets
        grid: Ground pathing grid for reachability checks
        
    Returns:
        Filtered list of enemies this unit can actually engage
    """
    attackable = []
    for e in enemies:
        # Check attackability (ground vs air)
        if e.is_flying:
            if not unit.can_attack_air:
                continue
        else:
            if not unit.can_attack_ground:
                continue
        
        # Check reachability for ground units
        # Ground units can't reach positions that aren't in the pathing grid
        if not unit.is_flying:
            if not cy_in_pathing_grid_ma(grid, e.position):
                continue
        
        attackable.append(e)
    return attackable



def is_choke_between(
    choke_width_map: dict[Point2, float], squad_pos: Point2, enemy_pos: Point2
) -> tuple[float, Point2 | None]:
    """
    Detect if a choke exists on the line between squad and enemy.
    
    Samples points along the straight line and checks against the
    dict of choke tile → width. Returns the width AND the specific tile
    position of the matched choke so callers can anchor retreat logic to
    that exact choke rather than the squad's current position.
    
    O(CHOKE_SAMPLE_POINTS) with O(1) dict lookups — fast and accurate.
    
    Args:
        choke_width_map: Dict[Point2, float] from create_narrow_choke_points()
        squad_pos: Squad center position
        enemy_pos: Enemy group center position
        
    Returns:
        (width, choke_tile): Width in tiles and the matched tile position,
        or (0.0, None) if no choke detected.
    """
    dx = enemy_pos.x - squad_pos.x
    dy = enemy_pos.y - squad_pos.y
    for i in range(1, CHOKE_SAMPLE_POINTS):
        t = i / CHOKE_SAMPLE_POINTS
        sample = Point2((int(squad_pos.x + dx * t), int(squad_pos.y + dy * t)))
        if sample in choke_width_map:
            return choke_width_map[sample], sample
    return 0.0, None


def effective_army_width(units: list[Unit]) -> float:
    """
    Estimate how much horizontal space an army occupies.
    
    A choke narrower than this forces the army to compress/funnel.
    Uses 2× mean distance from center mass as a diameter approximation.
    max_dist was too sensitive to outliers — one kiting unit far from the
    group could flip the funnel assessment every other frame.
    
    Small armies (≤2 units) return a minimum of 2.0 to avoid
    trivially passing through any choke.
    
    Args:
        units: List of units in the army/squad
        
    Returns:
        Estimated army diameter in tiles.
    """
    if len(units) <= 2:
        return CHOKE_MIN_ARMY_WIDTH
    
    center, _ = cy_find_units_center_mass(units, 10.0)
    center = Point2(center)
    mean_dist = sum(cy_distance_to(u.position, center) for u in units) / len(units)
    return mean_dist * 2


def compute_choke_melee_ratio(enemies) -> float:
    """
    Compute enemy_melee_dps / total_enemy_dps.
    
    Measures how much damage a choke would deny. High ratio means the choke
    bottlenecks most of the enemy's DPS (melee can't get through).
    
    Perf note: O(n) single pass over enemies.
    
    Returns:
        Ratio 0.0-1.0. High = choke is valuable. 0.0 = all ranged, choke doesn't help.
    """
    total_dps = 0.0
    melee_dps = 0.0
    for enemy in enemies:
        dps = enemy.ground_dps
        if dps <= 0:
            continue
        total_dps += dps
        if enemy.ground_range < CHOKE_MELEE_RANGE:
            melee_dps += dps
    if total_dps <= 0:
        return 0.0
    return melee_dps / total_dps


def is_near_choke_or_ramp(bot, army_center: Point2) -> bool:
    """
    O(1) check if army is near a choke/ramp using precomputed grid.
    
    When true, formation logic should be skipped to prevent bottlenecks
    where ranged units block melee units from moving through narrow passages.
    
    Args:
        bot: Bot instance with choke_grid attribute (created in on_start)
        army_center: Center of mass of the squad
        
    Returns:
        True if near choke/ramp (skip formation), False otherwise
    """
    x, y = int(army_center.x), int(army_center.y)
    return bot.choke_grid[x, y] > 1.0


def is_blind_ramp_attack(bot, unit: Unit):
    """
    Check if unit is at the bottom of a ramp with enemy ranged units at the
    unseen top - attacking would be a blind disadvantage.
    
    From anglerbot: Used to trigger KeepUnitSafe instead of attacking.
    
    Returns:
        The matched Ramp if unit should NOT attack, None otherwise
    """
    for ramp in bot.game_info.map_ramps:
        # Check if unit is at bottom of this ramp
        if cy_distance_to(unit.position, ramp.bottom_center) > 5.0:
            continue
            
        # Skip if we can see the top - no vision disadvantage
        if bot.is_visible(ramp.top_center):
            continue
            
        # Unit is at bottom, can't see top - check for enemy ranged at top
        # Use all_enemy_units (includes memory) since we can't see them if top is dark
        for enemy in bot.all_enemy_units:
            if enemy.ground_range < 2:  # Skip melee
                continue
            if cy_distance_to(enemy.position, ramp.top_center) < 6.0:
                return ramp  # Enemy ranged at top - don't attack blind
                
    return None


def is_unit_on_ramp(bot, unit: Unit) -> bool:
    """
    Check if unit is on or near a ramp.
    
    When true and no enemies nearby, unit should move through the ramp
    instead of standing there (even to attack structures).
    
    Returns:
        True if unit is on/near any ramp
    """
    for ramp in bot.game_info.map_ramps:
        # Check if unit is near top or bottom of ramp
        if cy_distance_to(unit.position, ramp.top_center) < 5.0:
            return True
        if cy_distance_to(unit.position, ramp.bottom_center) < 5.0:
            return True
                
    return False


def get_formation_move_target(
    unit: Unit, 
    squad_units: list[Unit], 
    target: Point2,
    bot=None,
) -> Point2:
    """
    Get move target that maintains formation AND cohesion during map crossing.
    
    Formation: Uses cy_adjust_moving_formation to keep melee in front, ranged behind
    Cohesion: Slows down units that get too far ahead OR too spread out
    
    SKIP formation when near chokes/ramps to prevent bottlenecks where ranged
    units block melee from moving through narrow passages.
    
    Args:
        unit: The unit to get move target for
        squad_units: All units in the squad (for center mass calculation)
        target: The ultimate destination
        bot: Bot instance for choke/ramp detection (optional)
        
    Returns:
        Adjusted move target (may be army center if unit is too far ahead/spread)
    """
    if len(squad_units) < 2:
        return target
    
    # Dynamic cohesion thresholds: scale with sqrt(unit_count)
    # Small armies get tighter bounds, large armies get more breathing room
    n_sqrt = math.sqrt(len(squad_units))
    ahead_threshold = FORMATION_AHEAD_BASE + n_sqrt * FORMATION_AHEAD_SCALE
    spread_threshold = FORMATION_SPREAD_BASE + n_sqrt * FORMATION_SPREAD_SCALE
    
    # Get army center of mass
    army_center, _ = cy_find_units_center_mass(squad_units, 10.0)
    army_center = Point2(army_center)
    
    # Skip formation logic near chokes/ramps to prevent bottlenecks
    if bot is not None and is_near_choke_or_ramp(bot, army_center):
        return target
    
    # Identify melee units (fodder - should be in front)
    fodder_tags = [u.tag for u in squad_units if u.ground_range <= MELEE_RANGE_THRESHOLD]
    
    # Only use cy_adjust_moving_formation if we have melee units (otherwise it does nothing)
    if fodder_tags:
        # Get formation adjustments from cython function
        # Returns dict of {unit_tag: (x, y)} for units that need to move back
        adjusted_positions = cy_adjust_moving_formation(
            our_units=squad_units,
            target=target,
            fodder_tags=fodder_tags,
            unit_multiplier=FORMATION_UNIT_MULTIPLIER,
            retreat_angle=FORMATION_RETREAT_ANGLE
        )
        
        # If this unit needs formation adjustment (ranged unit too far forward), use that
        if unit.tag in adjusted_positions:
            return Point2(adjusted_positions[unit.tag])
    
    # Cohesion check 1: Is this unit too far ahead of army center (toward target)?
    unit_dist_to_target = cy_distance_to(unit.position, target)
    center_dist_to_target = cy_distance_to(army_center, target)
    ahead_distance = center_dist_to_target - unit_dist_to_target
    
    if ahead_distance > ahead_threshold:
        # Unit is streaming ahead - move toward army center to wait for others
        return army_center.towards(target, ahead_threshold * 0.5)
    
    # Cohesion check 2: Is this unit too far from army center (spread out)?
    dist_from_center = cy_distance_to(unit.position, army_center)
    if dist_from_center > spread_threshold:
        # Unit is too spread out - move toward army center
        return army_center.towards(target, spread_threshold * 0.5)
    
    # Unit is in good position - continue to target
    return target


def _expire_snipe_committed(bot) -> None:
    """Remove expired entries from snipe/focus skip-sets and stale state entries.

    Safety net: if a state outlives its skip-set, clean it up to prevent
    stale state machines from conflicting with per-unit micro.
    """
    from bot.combat.group_snipe import SNIPE_EXIT_FRAMES

    game_loop = bot.state.game_loop
    expired = [tag for tag, expiry in bot._snipe_committed.items() if game_loop >= expiry]
    for tag in expired:
        bot._snipe_committed.pop(tag, None)

    # Also clean up _snipe_state entries that have been alive too long
    max_lifetime = SNIPE_EXIT_FRAMES * 4
    stale_squads = [
        sid for sid, info in bot._snipe_state.items()
        if (game_loop - info["commit_frame"]) > max_lifetime
    ]
    for sid in stale_squads:
        # Remove stalkers from skip-set and delete state
        info = bot._snipe_state[sid]
        for tag in info["stalker_tags"]:
            bot._snipe_committed.pop(tag, None)
        del bot._snipe_state[sid]

    # Expire focus-fire skip-set entries
    expired_focus = [tag for tag, expiry in bot._focus_committed.items() if game_loop >= expiry]
    for tag in expired_focus:
        bot._focus_committed.pop(tag, None)


def control_main_army(bot, main_army: Units, target: Point2, squads: list[UnitSquad]) -> Point2:
    """
    Controls the main army's movement and engagement logic using individual unit behaviors.
    """
    # Cache upgrades once per frame for target scoring (Charge check, etc.)
    update_upgrades(bot.state.upgrades)
    # Refresh confirmed repairer tags for target scoring (Bunker/PF repair)
    update_repairer_tags(getattr(bot, "_repairer_tags", set()))
    
    # HT archon merge: once per frame, merge low-energy HT pairs
    merge_high_templars(bot)
    
    # ARES requires get_squads() to be called before get_position_of_main_squad()
    if not squads:
        squads = bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=ATTACKING_SQUAD_RADIUS)
    
    pos_of_main_squad: Point2 = bot.mediator.get_position_of_main_squad(role=UnitRole.ATTACKING)
    grid: np.ndarray = bot.mediator.get_ground_grid
    avoid_grid: np.ndarray = bot.mediator.get_ground_avoidance_grid
    air_avoid_grid: np.ndarray = bot.mediator.get_air_avoidance_grid
    choke_width_map: dict[Point2, float] = bot.narrow_choke_points
    
    if main_army:
        bot.total_health_shield_percentage = (
            sum(unit.shield_health_percentage for unit in main_army) / len(main_army)
        )

    # Expire stale snipe skip-set entries (safety net for edge cases)
    _expire_snipe_committed(bot)

    for squad in squads:
        squad_position: Point2 = squad.squad_position
        units: list[Unit] = squad.squad_units
        
        # Skip empty squads
        if not units:
            continue

        # Main squad coordination: main squad goes to target, others converge on main squad
        move_to: Point2 = target if squad.main_squad else pos_of_main_squad

        # Find nearby enemy units using per-unit detection (larger range, checks each unit)
        unit_positions = [u.position for u in units]
        all_close_results = bot.mediator.get_units_in_range(
            start_points=unit_positions,
            distances=UNIT_ENEMY_DETECTION_RANGE,
            query_tree=UnitTreeQueryType.AllEnemy,
            return_as_dict=False,
        )
        # Combine all detected enemies and deduplicate by tag
        seen_tags = set()
        all_close_list = []
        for result in all_close_results:
            for u in result:
                if u.tag not in seen_tags and not u.is_memory and not u.is_structure and u.type_id not in COMMON_UNIT_IGNORE_TYPES:
                    seen_tags.add(u.tag)
                    all_close_list.append(u)
        all_close = Units(all_close_list, bot)
        
        # Army value gate: sum enemy army_value to decide formation vs full micro.
        # Low threat (scout, lone ling) → stay in formation, ShootTargetInRange only.
        # Real threat → full individual micro.
        enemy_army_value = sum(
            UNIT_DATA.get(u.type_id, {}).get("army_value", 0.0)
            for u in all_close
        )
        
        # Fallback: if any enemy is actively attacking our units (in weapon range + facing),
        # override the army_value gate so our units can micro (e.g. Stalker kites lone Ling).
        enemy_actively_engaging = False
        decisive_victory = False  # set True below when sim result is decisive — overrides choke hold
        if all_close and enemy_army_value < ENGAGEMENT_ARMY_VALUE_THRESHOLD:
            for enemy in all_close:
                attack_range = enemy.ground_range + enemy.radius + ACTIVE_ENGAGE_RANGE_BUFFER
                for own_unit in units:
                    dist = cy_distance_to(enemy.position, own_unit.position)
                    if dist <= attack_range + own_unit.radius and cy_is_facing(enemy, own_unit, angle_error=ACTIVE_ENGAGE_ANGLE):
                        enemy_actively_engaging = True
                        break
                if enemy_actively_engaging:
                    break
        
        if all_close and (enemy_army_value >= ENGAGEMENT_ARMY_VALUE_THRESHOLD or enemy_actively_engaging):
            # Defensive mode override: when not attacking, always engage enemies near our bases
            # This prevents retreating from winnable local fights due to unfavorable global situation
            if not bot._commenced_attack:
                can_engage = True
            else:
                # Attacking mode: use localized combat simulation for tactical decisions
                # Squad-level combat sim evaluates if THIS squad can win against THESE enemies
                own_nearby = main_army.filter(
                    lambda u: cy_distance_to_squared(u.position, squad_position) < SQUAD_NEARBY_FRIENDLY_RANGE_SQ
                )
                
                # Filter enemy units to focus on actual combat units for more conservative simulation
                combat_enemies = all_close.filter(
                    lambda u: u.type_id not in WORKER_TYPES and not u.is_structure
                )
                # Static D within the squad's detection zone joins the sim —
                # all_close drops structures, but garrison/cannon fire is real
                combat_enemies.extend(
                    s for s in get_sim_static_defense(bot)
                    if cy_distance_to_squared(s.position, squad_position) < UNIT_ENEMY_DETECTION_RANGE ** 2
                )
                
                squad_fight_result = bot.mediator.can_win_fight(
                    own_units=own_nearby,
                    enemy_units=combat_enemies,
                    workers_do_no_damage=True,
                )
                decisive_victory = squad_fight_result in VICTORY_DECISIVE_OR_BETTER
                
                # Initialize squad tracking if first encounter
                squad_id = squad.squad_id
                if squad_id not in bot._squad_engagement_tracker:
                    bot._squad_engagement_tracker[squad_id] = {"can_engage": True}
                
                # Apply engagement hysteresis: different thresholds for starting vs stopping engagement
                # This prevents squads from flip-flopping between aggressive and defensive behavior
                if bot._squad_engagement_tracker[squad_id]["can_engage"]:
                    # Already engaging: only retreat if situation becomes dire
                    if squad_fight_result in LOSS_OVERWHELMING_OR_WORSE:
                        bot._squad_engagement_tracker[squad_id]["can_engage"] = False
                else:
                    # Not engaging: require advantage before committing to fight
                    if squad_fight_result in VICTORY_MARGINAL_OR_BETTER:
                        bot._squad_engagement_tracker[squad_id]["can_engage"] = True
                
                can_engage = bot._squad_engagement_tracker[squad_id]["can_engage"]
            
            # --- Choke policy: don't push through chokes that constrain us ---
            # Compare choke width against effective army widths to determine
            # who actually gets funneled. A choke only matters if it compresses
            # at least one side's army below its natural width.
            choke_active = False
            enemy_center_chk: Point2 | None = None
            choke_tile: Point2 | None = None
            if can_engage and bot._commenced_attack:
                # Only ground combat units matter for choke calculations.
                # Flying units bypass chokes entirely; workers don't fight.
                # COMMON_UNIT_IGNORE_TYPES already removed overlords/observers from all_close.
                choke_enemies = [u for u in all_close if not u.is_flying and u.type_id not in WORKER_TYPES]
                if choke_enemies:
                    enemy_center_mass_chk, _ = cy_find_units_center_mass(choke_enemies, 20.0)
                    enemy_center_chk = Point2(enemy_center_mass_chk)
                    choke_width, choke_tile = is_choke_between(choke_width_map, squad_position, enemy_center_chk)
                    if choke_width > 0.0:
                        our_width = effective_army_width(units)
                        enemy_width = effective_army_width(choke_enemies)
                        we_funnel = our_width > choke_width
                        they_funnel = enemy_width > choke_width
                        
                        if decisive_victory:
                            # Winning so decisively that funneling doesn't matter — push through
                            render_choke_decision_debug(
                                bot, squad_position, enemy_center_chk,
                                choke_width, our_width, enemy_width, "PASS:decisive_win"
                            )
                        elif we_funnel and not they_funnel:
                            # We compress, they don't — bad for us
                            can_engage = False
                            choke_active = True
                            render_choke_decision_debug(
                                bot, squad_position, enemy_center_chk,
                                choke_width, our_width, enemy_width, "HOLD:we_funnel"
                            )
                        elif they_funnel and not we_funnel:
                            # They compress, we don't — lure them through
                            can_engage = False
                            choke_active = True
                            render_choke_decision_debug(
                                bot, squad_position, enemy_center_chk,
                                choke_width, our_width, enemy_width, "HOLD:lure"
                            )
                        elif we_funnel and they_funnel:
                            # Both funnel — melee ratio decides
                            melee_ratio = compute_choke_melee_ratio(choke_enemies)
                            if melee_ratio >= CHOKE_MELEE_DPS_THRESHOLD:
                                can_engage = False
                                choke_active = True
                                render_choke_decision_debug(
                                    bot, squad_position, enemy_center_chk,
                                    choke_width, our_width, enemy_width, "HOLD:melee_ratio"
                                )
                            else:
                                render_choke_decision_debug(
                                    bot, squad_position, enemy_center_chk,
                                    choke_width, our_width, enemy_width, "PASS:low_melee"
                                )
                        else:
                            # Neither funnels, choke is irrelevant
                            render_choke_decision_debug(
                                bot, squad_position, enemy_center_chk,
                                choke_width, our_width, enemy_width, "PASS:both_fit"
                            )
                        
                        # Detailed HOLD label with melee ratio (only when suppressed)
                        if choke_active:
                            melee_ratio = compute_choke_melee_ratio(choke_enemies)
                            render_choke_policy_debug(
                                bot, squad_position, enemy_center_chk,
                                melee_ratio, choke_width, our_width, enemy_width
                            )

            # --- Choke retreat: pull squad back as a group ---
            # Direction mirrors ramp retreat: away from enemy (the "dangerous side").
            # PathGroupToTarget issues one MOVE command to all squad tags via give_same_action.
            # Deduplication prevents spam — same as formation group commands.
            # Ranged and melee loops below are gated so individual micro can't override this.
            if choke_active and enemy_center_chk is not None and choke_tile is not None:
                retreat_dir = (squad_position - enemy_center_chk).normalized
                # Anchor to the specific choke tile, not the squad's current position.
                # This gives a stable fixed target — prevents retreating through multiple
                # chokes as each new choke becomes the new anchor.
                retreat_target = Point2(choke_tile + retreat_dir * CHOKE_RETREAT_DIST)
                choke_retreat = CombatManeuver()
                choke_retreat.add(PathGroupToTarget(
                    start=squad_position,
                    group=units,
                    group_tags={u.tag for u in units},
                    grid=grid,
                    target=retreat_target,
                    success_at_distance=3.0,
                ))
                bot.register_behavior(choke_retreat)

            # --- Blink snipe evaluation (before per-unit micro) ---
            # Try to commit a snipe if conditions are met, then execute active snipes.
            # Committed stalkers are added to bot._snipe_committed and skipped in per-unit loop.
            squad_stalkers = [
                u for u in units
                if u.type_id == UnitTypeId.STALKER
                and u.tag not in bot._snipe_committed
                and u.tag not in bot._chase_committed
                and u.tag not in bot._focus_committed
            ]
            if can_engage and squad_stalkers and all_close:
                try_commit_snipe(
                    bot=bot,
                    squad_id=squad.squad_id,
                    enemies=all_close,
                    squad_stalkers=squad_stalkers,
                    squad_position=squad_position,
                )
            # Execute active snipe (runs every frame for committed squads)
            snipe_info = bot._snipe_state.get(squad.squad_id)
            if snipe_info and snipe_info.get("mode") == "b":
                execute_snipe_b(bot, squad.squad_id, grid)
            else:
                execute_snipe_a(bot, squad.squad_id, grid)
            # Debug: render snipe state
            render_snipe_debug(bot, squad.squad_id)

            # --- Blink focus-fire evaluation (after snipe, before chase) ---
            # Focus-fire: multi-volley pursuit of harassers (Oracle, Banshee, Tempest, etc.)
            # Committed inside try_commit_snipe when mode="focus". Uses stalkers not already committed.
            # Execute active focus-fire (runs every frame for committed squads)
            execute_focus(bot, squad.squad_id, grid, all_close)
            render_focus_debug(bot, squad.squad_id)

            # --- Blink chase evaluation (after snipe/focus, before per-unit micro) ---
            # Chase retreating enemies if we're winning. Uses stalkers not committed to snipe/focus/chase.
            chase_stalkers = [
                u for u in units
                if u.type_id == UnitTypeId.STALKER
                and u.tag not in bot._snipe_committed
                and u.tag not in bot._chase_committed
                and u.tag not in bot._focus_committed
            ]
            if can_engage and chase_stalkers and all_close:
                try_commit_chase(
                    bot=bot,
                    squad_id=squad.squad_id,
                    enemies=all_close,
                    squad_stalkers=chase_stalkers,
                    squad_position=squad_position,
                    can_engage=can_engage,
                )
            # Execute active chase (runs every frame for committed squads)
            execute_chase(bot, squad.squad_id, grid, all_close)
            render_chase_debug(bot, squad.squad_id)

            # Separate melee, ranged, air, spell casters, and high templars
            # Ground combat units only (air units get their own micro)
            # HTs get their own control (like disruptors) - Feedback + safe follow
            # Other spellcasters: Sentries, etc.
            # Air excludes Observers/Warp Prisms (can't attack), includes Phoenixes/Oracles/Tempests
            high_templars = [u for u in units if u.type_id == UnitTypeId.HIGHTEMPLAR]
            melee = [u for u in units if u.ground_range <= MELEE_RANGE_THRESHOLD and u.energy == 0 and u.can_attack and not u.is_flying]
            ranged = [u for u in units if u.ground_range > MELEE_RANGE_THRESHOLD and u.energy == 0 and u.can_attack and not u.is_flying]
            air = [u for u in units if u.is_flying and u.can_attack]
            spellcasters = [u for u in units if (u.energy > 0 and not u.is_flying and u.type_id != UnitTypeId.HIGHTEMPLAR) or u.type_id == UnitTypeId.DISRUPTOR]
            
            # --- Concave formation: fan out ranged units before engagement ---
            # Compute enemy center for formation targeting
            enemy_center_mass, _ = cy_find_units_center_mass(all_close, 20.0)
            enemy_center = Point2(enemy_center_mass)
            
            # Only attempt formation when engaging (not retreating)
            formation_active = False
            if can_engage and ranged:
                formation_active = execute_fan_out(
                    bot, squad.squad_id, ranged, squad_position, enemy_center
                )
            
            # Debug: visualize concave formation state
            render_concave_formation_debug(
                bot, squad.squad_id, ranged, squad_position, enemy_center, formation_active
            )
            
            # Weighted scoring: pick best target for fallback movement direction
            # Uses first squad unit as reference (any unit works — just for direction)
            enemy_target = select_target(units[0], all_close)
            
            # ARES safety awareness - avoid unsafe positions when defending (ranged only)
            ranged_on_unsafe_ground = []
            if not bot._commenced_attack and len(bot.townhalls) <= EARLY_GAME_SAFE_GROUND_CHECK_BASES and bot.time < EARLY_GAME_TIME_LIMIT:
                for r_unit in ranged:
                    if not bot.mediator.is_position_safe(grid=grid, position=r_unit.position):
                        ranged_on_unsafe_ground.append(r_unit)
                        # Find safer position instead of hardcoded start_location
                        safe_spot = bot.mediator.find_closest_safe_spot(
                            from_pos=r_unit.position, 
                            grid=grid,
                            radius=UNSAFE_GROUND_CHECK_RADIUS
                        )
                        r_unit.move(safe_spot)
                        
            # Ranged micro - skip if formation fan-out is handling them
            for r_unit in ranged:
                if formation_active or choke_active:
                    continue  # Group command active — skip individual micro
                if r_unit in ranged_on_unsafe_ground:
                    continue  # Skip units retreating to safer ground
                # Skip stalkers committed to a group snipe, chase, or focus-fire (they get group commands)
                if r_unit.tag in bot._snipe_committed and bot.state.game_loop < bot._snipe_committed[r_unit.tag]:
                    continue
                if r_unit.tag in bot._chase_committed and bot.state.game_loop < bot._chase_committed[r_unit.tag]:
                    continue
                if r_unit.tag in bot._focus_committed and bot.state.game_loop < bot._focus_committed[r_unit.tag]:
                    continue
                
                # Ramp safety: don't attack up ramp without vision (blind disadvantage)
                blind_ramp = is_blind_ramp_attack(bot, r_unit)
                if blind_ramp:
                    render_blind_ramp_debug(bot, r_unit, blind_ramp)
                    if bot._blind_ramp_target is None:
                        bot._blind_ramp_target = Point2(blind_ramp.top_center)
                    away_from_top = (r_unit.position - blind_ramp.top_center).normalized
                    retreat_pos = Point2(r_unit.position + away_from_top * 4.0)
                    retreat_maneuver = CombatManeuver()
                    retreat_maneuver.add(PathUnitToTarget(unit=r_unit, grid=grid, target=retreat_pos, success_at_distance=1.0))
                    bot.register_behavior(retreat_maneuver)
                    continue
                
                # Filter enemies to only those this unit can attack and reach
                unit_enemies = get_attackable_enemies(r_unit, all_close, grid)
                if not unit_enemies:
                    continue
                    
                # Stalkers get blink-aware micro when low health
                if r_unit.type_id == UnitTypeId.STALKER:
                    ranged_maneuver = micro_stalker(
                        stalker=r_unit,
                        enemies=unit_enemies,
                        grid=grid,
                        avoid_grid=avoid_grid,
                        aggressive=can_engage,
                        squad_center=squad_position,
                        enemy_center=enemy_center,
                        ranged_units=ranged,
                        all_close=all_close,
                        bot=bot,
                    )
                else:
                    ranged_maneuver = micro_ranged_unit(
                        unit=r_unit,
                        enemies=unit_enemies,
                        grid=grid,
                        avoid_grid=avoid_grid,
                        aggressive=can_engage
                    )
                bot.register_behavior(ranged_maneuver)

                # Debug: show micro state above ranged units
                from bot.combat.unit_micro import _has_melee_threat, _is_targeted_by_ranged
                is_threatened = _has_melee_threat(r_unit, unit_enemies) or _is_targeted_by_ranged(r_unit, unit_enemies)
                render_micro_state_debug(
                    bot, r_unit,
                    is_threatened=is_threatened,
                    aggressive=can_engage,
                    weapon_ready=r_unit.weapon_ready,
                    in_range=bool(cy_in_attack_range(r_unit, unit_enemies)),
                )

            # Air micro - separate handling for flying combat units
            for a_unit in air:
                if formation_active or choke_active:
                    continue  # Group command active — skip individual micro
                unit_enemies = get_attackable_enemies(a_unit, all_close, grid)
                if not unit_enemies:
                    continue
                air_maneuver = micro_air_unit(
                    unit=a_unit,
                    enemies=unit_enemies,
                    avoid_grid=air_avoid_grid,
                    aggressive=can_engage,
                )
                bot.register_behavior(air_maneuver)

            # Melee micro - weighted scoring handles priority targeting
            for m_unit in melee:
                if choke_active:
                    continue  # Group retreat active — skip individual micro
                # Filter enemies to only those this unit can attack and reach
                unit_enemies = get_attackable_enemies(m_unit, all_close, grid)
                if not unit_enemies:
                    continue
                
                melee_maneuver = micro_melee_unit(
                    unit=m_unit,
                    enemies=unit_enemies,
                    avoid_grid=avoid_grid,
                    grid=grid,
                    fallback_position=move_to if not can_engage else (enemy_target.position if enemy_target else move_to),
                    aggressive=can_engage
                )
                bot.register_behavior(melee_maneuver)
                
            # Compute ranged center for all ground spellcasters to follow
            # during combat — mirrors the ranged line's movement (concave +
            # stutter-back) instead of the full squad center (which includes
            # melee at the front).
            ranged_center: Point2 | None = None
            if ranged:
                rc, _ = cy_find_units_center_mass(ranged, 10.0)
                ranged_center = Point2(rc)

            # Spellcasters 
            if spellcasters:
                disruptors = [spellcaster for spellcaster in spellcasters if spellcaster.type_id == UnitTypeId.DISRUPTOR]
                other_casters = [spellcaster for spellcaster in spellcasters if spellcaster.type_id != UnitTypeId.DISRUPTOR]
                
                # Handle Disruptors with nova logic

                if disruptors:
                    # Filter enemies for disruptors once: exclude workers and broodlings
                    # Exception: confirmed repairer SCVs (repairing a Bunker/PF) are
                    # valid nova targets — killing the heal beats out-damaging it
                    repairer_tags = getattr(bot, "_repairer_tags", set())
                    disruptor_targets = all_close.filter(
                        lambda u: u.type_id not in DISRUPTOR_IGNORE_TYPES
                        or u.tag in repairer_tags
                    )
                    
                    nova_manager = bot.nova_manager if hasattr(bot, 'nova_manager') else None
                    
                    for disruptor in disruptors:
                        micro_disruptor(
                            disruptor=disruptor,
                            enemies=disruptor_targets,
                            friendly_units=units,
                            avoid_grid=avoid_grid,
                            grid=grid,
                            bot=bot,
                            nova_manager=nova_manager,
                            squad_position=squad_position,
                            ranged_center=ranged_center,
                        )
                
                # Handle Sentries - Force Field split > Guardian Shield > safe follow
                sentries = [c for c in other_casters if c.type_id == UnitTypeId.SENTRY]

                # Compute Guardian Shield assignments once per squad:
                # an influence map of friendly density picks the densest
                # uncovered pocket for each Sentry. Already-shielded areas
                # are zeroed (exclusion-mask pattern, like Disruptor Nova),
                # so Sentries spread to different pockets instead of stacking.
                gs_approved_tags: set[int] = set()
                gs_peaks: dict[int, Point2] = {}
                if sentries:
                    gs_approved_tags, gs_peaks = _compute_guardian_shield_assignments(
                        sentries=sentries,
                        squad_units=units,
                        enemies=all_close,
                        bot=bot,
                    )

                # Compute FF assignments once per squad (pools energy across all sentries)
                # Priority: ramp block > choke block > army split
                # A single FF at a real bottleneck (ramp or narrow choke) is
                # more impactful than a generic split through the enemy center.
                ff_assignments: dict[int, list[Point2]] | None = None
                ff_debug_center: Point2 | None = None
                ff_debug_mode: str = ""
                ff_debug_refined = None
                if sentries and all_close:
                    # Main ramp block: specialized single-FF for the two main-base ramps only.
                    # All other ramps are handled by the general choke-block path below
                    # (they're in map_chokes as MDRamp instances → create_narrow_choke_points).
                    own_ramp = bot.main_base_ramp
                    enemy_ramp = bot.mediator.get_enemy_ramp

                    # Try main ramp block first: single FF at ramp center if enemy is crossing
                    ramp_result = compute_ff_main_ramp_block(
                        enemies=list(all_close),
                        sentries=sentries,
                        own_ramp=own_ramp,
                        enemy_ramp=enemy_ramp,
                        active_ffs=bot.mediator.get_forcefield_positions,
                    )
                    if ramp_result is not None and ramp_result.assignments:
                        ff_debug_center = ramp_result.enemy_center
                        ff_debug_mode = "RAMP"
                        ff_assignments = {}
                        for sentry_unit, pos in ramp_result.assignments:
                            ff_assignments.setdefault(sentry_unit.tag, []).append(pos)
                    elif choke_tile is not None and enemy_center_chk is not None:
                        # No ramp block — try static choke block using raycast-refined position
                        passage_dir = enemy_center_chk - squad_position
                        refined = get_or_refine_choke(bot, choke_tile, passage_dir)
                        if refined is not None:
                            # Show the refined choke overlay even if FF doesn't fire,
                            # so the raycast result is visible during testing
                            ff_debug_refined = refined
                            ff_debug_center = enemy_center_chk
                            choke_result = compute_ff_choke_block(
                                enemies=list(all_close),
                                sentries=sentries,
                                refined=refined,
                                active_ffs=bot.mediator.get_forcefield_positions,
                            )
                            if choke_result is not None and choke_result.assignments:
                                ff_debug_mode = "CHOKE"
                                ff_assignments = {}
                                for sentry_unit, pos in choke_result.assignments:
                                    ff_assignments.setdefault(sentry_unit.tag, []).append(pos)
                    if ff_assignments is None:
                        # No ramp or static choke block — try dynamic choke detection.
                        # Scans for building/resource-created chokes around the squad
                        # (own + enemy buildings, minerals, geysers). Runs every
                        # DYNAMIC_CHOKE_SCAN_INTERVAL frames per squad, cached per squad.
                        from bot.constants import DYNAMIC_CHOKE_SCAN_INTERVAL
                        squad_id = squad.squad_id
                        current_frame = bot.state.game_loop
                        cached = bot.dynamic_choke_cache.get(squad_id)
                        if cached is None or current_frame - cached[0] >= DYNAMIC_CHOKE_SCAN_INTERVAL:
                            dynamic_refined = detect_dynamic_choke(
                                bot, squad_position, enemy_center_chk
                            )
                            bot.dynamic_choke_cache[squad_id] = (current_frame, dynamic_refined)
                        else:
                            dynamic_refined = cached[1]
                        if dynamic_refined is not None:
                            ff_debug_refined = dynamic_refined
                            ff_debug_center = enemy_center_chk if enemy_center_chk is not None else squad_position
                            choke_result = compute_ff_choke_block(
                                enemies=list(all_close),
                                sentries=sentries,
                                refined=dynamic_refined,
                                active_ffs=bot.mediator.get_forcefield_positions,
                            )
                            if choke_result is not None and choke_result.assignments:
                                ff_debug_mode = "DCHOKE"
                                ff_assignments = {}
                                for sentry_unit, pos in choke_result.assignments:
                                    ff_assignments.setdefault(sentry_unit.tag, []).append(pos)
                    if ff_assignments is None:
                        # No ramp, choke, or dynamic choke block — try army split
                        ff_result = compute_ff_split(
                            enemies=list(all_close),
                            sentries=sentries,
                            own_center=squad_position,
                            ground_grid=grid,
                        )
                        if ff_result is not None:
                            ff_debug_center = ff_result.enemy_center
                            ff_debug_mode = "SPLIT"
                            if ff_result.assignments:
                                # Convert [(sentry, pos), ...] → {sentry_tag: [pos1, pos2, ...]}
                                ff_assignments = {}
                                for sentry_unit, pos in ff_result.assignments:
                                    ff_assignments.setdefault(sentry_unit.tag, []).append(pos)

                # Debug visualization for FF placement
                render_ff_split_debug(
                    bot,
                    ff_assignments=ff_assignments,
                    sentries=sentries,
                    enemy_center=ff_debug_center,
                    ff_mode=ff_debug_mode,
                )
                # Refined choke overlay (green) when a choke block is active
                if ff_debug_refined is not None:
                    render_refined_choke_debug(
                        bot,
                        refined=ff_debug_refined,
                        enemy_center=ff_debug_center,
                    )

                # Guardian Shield influence-map overlay
                render_gs_debug(bot, sentries, squad_position)

                for sentry in sentries:
                    micro_sentry(
                        sentry=sentry,
                        friendly_units=units,
                        enemies=all_close,
                        avoid_grid=avoid_grid,
                        grid=grid,
                        bot=bot,
                        squad_position=squad_position,
                        ranged_center=ranged_center,
                        ff_assignments=ff_assignments,
                        gs_approved=sentry.tag in gs_approved_tags,
                        gs_target=gs_peaks.get(sentry.tag),
                    )
                
                # Handle other spellcasters (Observers, etc.) - stay with army, stay safe
                remaining_casters = [c for c in other_casters if c.type_id != UnitTypeId.SENTRY]
                for caster in remaining_casters:
                    caster_maneuver = CombatManeuver()
                    caster_maneuver.add(KeepUnitSafe(unit=caster, grid=avoid_grid))
                    caster_maneuver.add(AMove(unit=caster, target=move_to))
                    bot.register_behavior(caster_maneuver)
            
            # High Templars: Feedback + safe follow (separate from other casters)
            for ht in high_templars:
                micro_high_templar(
                    ht=ht,
                    enemies=all_close,
                    avoid_grid=avoid_grid,
                    grid=grid,
                    bot=bot,
                    squad_position=squad_position,
                    ranged_center=ranged_center,
                )
                        
        else:
            # No enemies — clear formation state so next engagement starts fresh
            clear_formation_state(bot, squad.squad_id)
            
            # Check for nearby enemy structures
            nearby_structures = bot.enemy_structures.closer_than(STRUCTURE_ATTACK_RANGE, squad_position)
            
            # Low-threat enemies present (below army_value threshold)?
            # Units maintain formation but shoot anything in weapon range.
            # NOTE: all_close is already filtered by COMMON_UNIT_IGNORE_TYPES, so
            # Overlords/Observers/Changelings won't disable formation here.
            low_threat_targets = all_close if all_close else None
            
            # Debug: render formation visualization (during map crossing, even with low-threat enemies)
            if not nearby_structures:
                render_formation_debug(bot, units, move_to)
            
            # Movement logic: formation + ShootTargetInRange for low-threat situations
            for unit in units:
                unit_grid = bot.mediator.get_air_grid if unit.is_flying else grid
                if unit.type_id == UnitTypeId.COLOSSUS:
                    unit_grid = bot.mediator.get_climber_grid
                    
                no_enemy_maneuver = CombatManeuver()
                
                # Shoot low-threat enemies in range without breaking formation
                if low_threat_targets and unit.can_attack:
                    no_enemy_maneuver.add(ShootTargetInRange(unit=unit, targets=low_threat_targets))
                
                # Disruptors follow army target when no enemies nearby (like observers)
                if unit.type_id == UnitTypeId.DISRUPTOR:
                    # Position between army and target: lean forward toward action
                    # Uses pos_of_main_squad (where army IS) + offset toward target (where it's going)
                    disruptor_target = Point2(pos_of_main_squad.towards(target, DISRUPTOR_FORWARD_OFFSET))
                    
                    distance_to_target = cy_distance_to(unit.position, disruptor_target)
                    
                    # If far from target, move towards it
                    if distance_to_target > DISRUPTOR_SQUAD_FOLLOW_DISTANCE:
                        no_enemy_maneuver.add(PathUnitToTarget(
                            unit=unit, grid=grid, target=disruptor_target,
                            success_at_distance=DISRUPTOR_SQUAD_TARGET_DISTANCE
                        ))
                    # Otherwise stay put (close enough to target)
                    bot.register_behavior(no_enemy_maneuver)
                    continue  # Skip rest of movement logic for disruptors
                
                # High Templars follow army safely (like disruptors) — skip merging HTs
                if unit.type_id == UnitTypeId.HIGHTEMPLAR:
                    if unit.orders and unit.orders[0].ability.id == AbilityId.MORPH_ARCHON:
                        continue  # Let merge complete
                    ht_target = pos_of_main_squad
                    distance_to_ht_target = cy_distance_to(unit.position, ht_target)
                    if distance_to_ht_target > HT_SQUAD_FOLLOW_DISTANCE:
                        no_enemy_maneuver.add(PathUnitToTarget(
                            unit=unit, grid=grid, target=ht_target,
                            success_at_distance=HT_SQUAD_TARGET_DISTANCE
                        ))
                    bot.register_behavior(no_enemy_maneuver)
                    continue  # Skip rest of movement logic for HTs
                
                should_wait = False  # Flag to skip fallback movement when unit should wait
                
                # Ramp check - push through ramps when no real threats nearby
                if not low_threat_targets and is_unit_on_ramp(bot, unit):
                    no_enemy_maneuver.add(PathUnitToTarget(
                        unit=unit, grid=unit_grid, target=move_to, success_at_distance=3.0
                    ))
                elif nearby_structures:
                    # Structures nearby - attack them with AMove for base clearing
                    closest_structure = cy_closest_to(unit.position, nearby_structures)
                    no_enemy_maneuver.add(AMove(unit=unit, target=closest_structure.position))
                else:
                    # Formation movement — works with or without low-threat enemies.
                    # cy_adjust_moving_formation keeps melee front, ranged back.
                    # Cohesion checks prevent streaming ahead or spreading out.
                    if cy_distance_to_squared(unit.position, move_to) > MAP_CROSSING_DISTANCE_SQ:
                        formation_target = get_formation_move_target(unit, units, move_to, bot=bot)
                        
                        # If formation_target != move_to, unit is ahead/spread — drift back smoothly
                        if formation_target != move_to:
                            should_wait = True
                            no_enemy_maneuver.add(PathUnitToTarget(
                                unit=unit, grid=unit_grid, target=formation_target, success_at_distance=1.0
                            ))
                        else:
                            # Unit is in good position - continue to target
                            no_enemy_maneuver.add(PathUnitToTarget(
                                unit=unit, grid=unit_grid, target=move_to, success_at_distance=MAP_CROSSING_SUCCESS_DISTANCE
                            ))
                    
                # Fallback for units without orders (but NOT if they should be waiting)
                if not unit.orders and not should_wait:
                    no_enemy_maneuver.add(AMove(unit=unit, target=move_to))
                bot.register_behavior(no_enemy_maneuver)

    return pos_of_main_squad

def _gatekeeper_structures(bot, position: Point2) -> list[Unit]:
    """Structures that could crowd a parking spot: our own (including the ones still
    under construction) and the enemy's."""
    return list(bot.structures.closer_than(GATEKEEPER_CLEARANCE_SEARCH_RADIUS, position)) + list(
        bot.enemy_structures.closer_than(GATEKEEPER_CLEARANCE_SEARCH_RADIUS, position)
    )


def _clearance_to_structures(position: Point2, structures: list[Unit]) -> float:
    """Gap between a tile and the closest structure footprint.

    Negative means the tile overlaps a footprint. A zealot standing there is shoved
    by the engine, which is what turned the old fallback into a tug-of-war with the
    probe that was building.
    """
    gap: float = math.inf
    for structure in structures:
        footprint = structure.footprint_radius or structure.radius or 0.0
        gap = min(
            gap,
            cy_distance_to(structure.position, position) - footprint - GATEKEEPER_ZEALOT_RADIUS,
        )
    return gap


def _is_walkable(bot, position: Point2) -> bool:
    """False on tiles a zealot cannot stand on (rocks, cliffs, closed edges)."""
    return cy_in_pathing_grid_ma(bot.mediator.get_ground_grid, position)


def _parking_spot_is_free(bot, position: Point2, gate_pos: Point2, structures: list[Unit]) -> bool:
    """A spot is free when the zealot fits, the tile is walkable and the gap stays clear.

    The gap itself never qualifies: a probe may be about to build there.
    """
    if cy_distance_to(position, gate_pos) < GATEKEEPER_PARK_MIN_DISTANCE:
        return False
    if not _is_walkable(bot, position):
        return False
    return _clearance_to_structures(position, structures) >= 0.0


def _find_parking_spot(bot, gate_pos: Point2, structures: list[Unit]) -> Point2 | None:
    """Closest spot behind the gate where the zealot actually fits.

    The old fallback aimed at one fixed offset towards our base, which on cramped
    maps lands inside a building footprint - the engine shoves the zealot out and
    the next frame orders it straight back into the same tile.
    """
    base_target = bot.start_location if bot.enemy_race == Race.Protoss else bot.natural_expansion
    inward = base_target - gate_pos
    if inward.length == 0:
        return None
    inward = inward.normalized
    lateral = Point2((-inward.y, inward.x))
    best_spot: Point2 | None = None
    best_gap = -math.inf

    for distance in GATEKEEPER_PARK_DISTANCES:
        for offset in GATEKEEPER_PARK_LATERAL_OFFSETS:
            candidate = gate_pos + inward * distance + lateral * offset
            if not _is_walkable(bot, candidate):
                continue
            gap = _clearance_to_structures(candidate, structures)
            if gap >= GATEKEEPER_CLEARANCE_MARGIN:
                return candidate
            if gap > best_gap:
                best_spot, best_gap = candidate, gap

    # Nothing has real room: only park where the zealot still does not overlap a footprint.
    if best_spot is not None and best_gap >= 0.0:
        return best_spot
    return None


def _current_spot(gate: Unit) -> Point2:
    """Where the gatekeeper is heading, or where it stands when it has no order."""
    order_target = gate.order_target
    if isinstance(order_target, Point2):
        return order_target
    return gate.position


def _order_move(gate: Unit, target: Point2) -> None:
    """Issue a move only when the zealot is not already on its way there.

    Re-issuing the same order every frame fights the engine's push-out: the engine
    moves the zealot off a freshly placed building and the next order pulls it back.
    """
    order_target = gate.order_target
    if isinstance(order_target, Point2) and cy_distance_to(order_target, target) <= GATEKEEPER_ORDER_TOLERANCE:
        return
    gate.move(target)


def _is_holding_position(gate: Unit) -> bool:
    orders = gate.orders
    return bool(orders) and orders[0].ability.id == AbilityId.HOLDPOSITION


def _park_off_the_gate(bot, gate: Unit, gate_pos: Point2, structures: list[Unit]) -> None:
    """Keep the gap clear so a probe can build where the gatekeeper rests.

    The parking spot is only re-picked when the current one stops being usable (a
    building landed on it), otherwise the zealot shuffles back and forth.
    """
    if _parking_spot_is_free(bot, _current_spot(gate), gate_pos, structures):
        return

    spot = _find_parking_spot(bot, gate_pos, structures)
    if spot is None:
        status = "none"
    elif _clearance_to_structures(spot, structures) < GATEKEEPER_CLEARANCE_MARGIN:
        status = "tight"
    else:
        status = "ok"

    log_transition(
        subsystem="combat",
        action="gatekeeper_parking",
        reason="no footprint-free spot behind the gate" if status == "none" else "parking spot has no room to spare",
        key="gatekeeper_parking_status",
        value=status,
    )

    if spot is None:
        # Nothing fits next to the gate: stay where the engine put us rather than
        # ordering into a tile that overlaps a building.
        return
    _order_move(gate, spot)


def _hold_the_gate(bot, gate: Unit, gate_pos: Point2, structures: list[Unit]) -> None:
    """Stand on the gap and hold it while enemies are close."""
    if _clearance_to_structures(gate_pos, structures) < 0.0:
        # The gap is plugged by a building - there is no tile to hold.
        _park_off_the_gate(bot, gate, gate_pos, structures)
        return
    if cy_distance_to(gate.position, gate_pos) > GATEKEEPER_POSITION_TOLERANCE:
        _order_move(gate, gate_pos)
        return
    if not _is_holding_position(gate):
        gate(AbilityId.HOLDPOSITION)


def gatekeeper_control(bot, gatekeeper: Units) -> None:
    gate_keep_pos = bot.gatekeeping_pos
    if not gate_keep_pos:
        return

    any_close = bot.mediator.get_units_in_range(
        start_points=[gate_keep_pos],
        distances=GATEKEEPER_DETECTION_RANGE,
        query_tree=UnitTreeQueryType.EnemyGround,
        return_as_dict=False,
    )[0]
    structures = _gatekeeper_structures(bot, gate_keep_pos)

    for gate in gatekeeper:
        if any_close:
            _hold_the_gate(bot, gate, gate_keep_pos, structures)
        else:
            _park_off_the_gate(bot, gate, gate_keep_pos, structures)


def _is_valid_warp_in_position(bot, position: Point2, ground_grid: np.ndarray) -> bool:
    """
    Check if a position supports warp-ins within the full matrix radius.
    
    Checks:
    1. Center and edge points within matrix radius are pathable (cy_in_pathing_grid_ma)
    2. Position is safe enough from enemy army
    """
    # Check center position is pathable
    if not cy_in_pathing_grid_ma(ground_grid, position):
        return False
    
    # Check a few points around the matrix radius to ensure full coverage
    for angle in [0, 90, 180, 270]:
        rad = math.radians(angle)
        check_pos = Point2((
            position.x + WARP_PRISM_MATRIX_RADIUS * math.cos(rad),
            position.y + WARP_PRISM_MATRIX_RADIUS * math.sin(rad)
        ))
        if not cy_in_pathing_grid_ma(ground_grid, check_pos):
            return False
    
    # Check distance from enemy army
    enemy_army = bot.mediator.get_cached_enemy_army
    if enemy_army:
        closest_enemy = enemy_army.closest_to(position)
        if cy_distance_to(closest_enemy.position, position) < WARP_PRISM_MIN_ENEMY_DISTANCE:
            return False
    
    return True


def _find_valid_warp_in_position(bot, current_pos: Point2, army_center: Point2) -> Point2:
    """
    Find the closest valid position for warp-ins near the current position.
    Searches in a grid pattern biased toward the army center.
    
    Returns the valid position, or army_center as fallback.
    """
    ground_grid: np.ndarray = bot.mediator.get_ground_grid
    
    # Check current position first
    if _is_valid_warp_in_position(bot, current_pos, ground_grid):
        return current_pos
    
    # Search in expanding rings, prioritizing positions toward army
    best_pos = None
    best_dist_to_army = float('inf')
    
    search_range = WARP_PRISM_POSITION_SEARCH_RANGE
    step = WARP_PRISM_POSITION_SEARCH_STEP
    
    # Generate candidate positions in a grid around current position
    for dx in np.arange(-search_range, search_range + step, step):
        for dy in np.arange(-search_range, search_range + step, step):
            if dx == 0 and dy == 0:
                continue
            candidate = Point2((current_pos.x + dx, current_pos.y + dy))
            
            if _is_valid_warp_in_position(bot, candidate, ground_grid):
                dist_to_army = cy_distance_to(candidate, army_center)  # both are Point2, OK
                if dist_to_army < best_dist_to_army:
                    best_dist_to_army = dist_to_army
                    best_pos = candidate
    
    return best_pos if best_pos else army_center


def warp_prism_follower(bot, warp_prisms: Units, main_army: Units) -> None:
    """
    Controls Warp Prisms: follows army, morphs between Transport/Phasing.
    Finds valid pathable/safe positions before entering phase mode.
    """
    air_grid: np.ndarray = bot.mediator.get_air_grid
    if not warp_prisms:
        return

    maneuver: CombatManeuver = CombatManeuver()
    for prism in warp_prisms:
        if main_army:
            distance_to_center = cy_distance_to(prism.position, main_army.center)

            # If close to army, find valid position and phase or move there
            if distance_to_center < WARP_PRISM_FOLLOW_DISTANCE:
                ground_grid = bot.mediator.get_ground_grid
                if _is_valid_warp_in_position(bot, prism.position, ground_grid):
                    prism(AbilityId.MORPH_WARPPRISMPHASINGMODE)
                else:
                    # Find closest valid position and move there
                    valid_pos = _find_valid_warp_in_position(bot, prism.position, main_army.center)
                    maneuver.add(PathUnitToTarget(unit=prism, target=valid_pos, grid=air_grid))
            else:
                # If no warpin in progress, revert to Transport
                # Or simply path near the army
                not_ready_units = [unit for unit in bot.units if not unit.is_ready and cy_distance_to(unit.position, prism.position) < WARP_PRISM_UNIT_CHECK_RANGE]
                if prism.type_id == UnitTypeId.WARPPRISMPHASING and not not_ready_units:
                        prism(AbilityId.MORPH_WARPPRISMTRANSPORTMODE)

                elif prism.type_id == UnitTypeId.WARPPRISM:
                    # Keep prism at offset distance behind the army center
                    direction_vector = (prism.position - main_army.center).normalized
                    new_target = main_army.center + direction_vector * WARP_PRISM_FOLLOW_OFFSET
                    maneuver.add(PathUnitToTarget(unit=prism, target=new_target, grid=air_grid, danger_distance=WARP_PRISM_DANGER_DISTANCE))
        else:
            # If no main army, just retreat to natural (or wherever)
            maneuver.add(
                PathUnitToTarget(
                    unit=prism,
                    target=bot.natural_expansion,
                    grid=air_grid,
                    danger_distance=WARP_PRISM_DANGER_DISTANCE
                )
            )

    bot.register_behavior(maneuver)


def _is_retreat_blocked(bot, army_center: Point2, base_position: Point2) -> bool:
    """
    Check if the army's retreat to base is blocked by enemy influence.

    Uses ground_grid (which includes enemy UNIT positions) — not avoidance_grid
    (which only has spell effects like storms/biles). Regular combat units
    don't add cost to the avoidance grid.

    Focuses on the ESCAPE ZONE (~20 tiles from army toward base) instead of
    sampling the entire path. The full path can be 80+ tiles; only 1-2 of 8
    samples would land in the danger zone, making retreat always look clear.
    """
    ground_grid: np.ndarray = bot.mediator.get_ground_grid

    # Quick check: if army center isn't under enemy influence, retreat is trivial
    if cy_point_below_value(ground_grid, army_center):
        return False

    # Focus on escape zone: first ~20 tiles toward base (or half distance if closer)
    army_dist_to_base = cy_distance_to(army_center, base_position)
    escape_distance = min(20.0, army_dist_to_base * 0.5)

    # Direction vector toward base
    dx = base_position.x - army_center.x
    dy = base_position.y - army_center.y
    path_len = max(math.sqrt(dx * dx + dy * dy), 0.01)
    dx_norm = dx / path_len
    dy_norm = dy / path_len

    num_samples = 6
    blocked_count = 0
    for i in range(1, num_samples + 1):
        dist = escape_distance * i / num_samples
        sample = Point2((
            army_center.x + dx_norm * dist,
            army_center.y + dy_norm * dist,
        ))
        is_blocked = not cy_in_pathing_grid_ma(ground_grid, sample) or not cy_point_below_value(ground_grid, sample)
        if is_blocked:
            blocked_count += 1
        if bot.debug:
            bot._recall_escape_samples.append((sample, is_blocked))

    if blocked_count >= num_samples // 2:
        if bot.debug:
            bot._recall_blocked_reason = f"escape_zone:{blocked_count}/{num_samples}"
        return True

    # Fallback: check if closest safe spot is actually toward base
    safe_spot = bot.mediator.find_closest_safe_spot(
        from_pos=army_center,
        grid=ground_grid,
        radius=MASS_RECALL_RETREAT_SEARCH_RADIUS,
    )
    if safe_spot is None:
        if bot.debug:
            bot._recall_blocked_reason = "no_safe_spot"
        return True
    safe_dist_to_base = cy_distance_to(safe_spot, base_position)
    blocked = safe_dist_to_base >= army_dist_to_base
    if bot.debug:
        bot._recall_blocked_reason = f"safe_away:{safe_dist_to_base:.0f}>={army_dist_to_base:.0f}" if blocked else None
    return blocked


def _find_recall_target(bot, main_army: Units) -> Point2:
    """
    Find the position that maximizes total army_value of trapped units within MASS_RECALL_RADIUS.

    A unit is "trapped" if:
    1. It's under enemy influence on the ground_grid (not avoidance_grid, which
       only has spell effects — regular combat units don't add cost there)
    2. Its escape toward base is also blocked (first 5-10 tiles toward base are dangerous)

    Falls back to all units under enemy influence if the directional check is too strict,
    then to full army center as last resort.

    O(n²) over trapped units, but n is typically small and this runs once per recall.
    """
    ground_grid: np.ndarray = bot.mediator.get_ground_grid
    nearest_base = cy_closest_to(main_army.center, bot.townhalls) if bot.townhalls else None
    base_pos = nearest_base.position if nearest_base else bot.start_location

    # Pass 1: units under enemy influence whose escape toward base is blocked
    trapped_units = []
    influenced_units = []
    for u in main_army:
        if bot.mediator.is_position_safe(grid=ground_grid, position=u.position):
            continue  # Not under enemy influence at all
        influenced_units.append(u)

        # Check if escape toward base is blocked (sample 2 points in retreat direction)
        dx = base_pos.x - u.position.x
        dy = base_pos.y - u.position.y
        path_len = max(math.sqrt(dx * dx + dy * dy), 0.01)
        dx_n = dx / path_len
        dy_n = dy / path_len

        escape_clear = False
        for dist in (5.0, 10.0):
            sample = Point2((u.position.x + dx_n * dist, u.position.y + dy_n * dist))
            if cy_point_below_value(ground_grid, sample):
                escape_clear = True
                break

        if not escape_clear:
            trapped_units.append(u)

    # Fallback: if directional check was too strict, use all influenced units
    if not trapped_units:
        trapped_units = influenced_units

    # If no units under enemy influence at all, fall back to full army
    if not trapped_units:
        return main_army.center

    if bot.debug:
        bot._recall_trapped_tags = {u.tag for u in trapped_units}
        bot._recall_influenced_tags = {u.tag for u in influenced_units}

    # Find the position that captures the most value of trapped units
    best_pos = main_army.center
    best_value = 0.0
    radius_sq = MASS_RECALL_RADIUS * MASS_RECALL_RADIUS

    for candidate in trapped_units:
        value = 0.0
        for unit in trapped_units:
            if cy_distance_to_squared(candidate.position, unit.position) <= radius_sq:
                value += UNIT_DATA.get(unit.type_id, {}).get("army_value", 1.0)
        if value > best_value:
            best_value = value
            best_pos = candidate.position

    return best_pos


def try_mass_recall(
    bot, main_army: Units, fight_result, squads: list[UnitSquad] | None = None,
) -> bool:
    """
    Decide whether to trigger Nexus Mass Recall based on combat conditions.
    Delegates actual casting to use_mass_recall() in structure_manager.

    Conditions (ALL must be true):
    1. Combat sim says LOSS_DECISIVE_OR_WORSE (devastating loss)  [army-wide]
    2. Own army has enough supply to be worth saving               [army-wide]
    3. At least one squad's retreat path is blocked                 [per-squad]

    Per-squad Gate 3: iterates squads and checks _is_retreat_blocked for each
    squad center individually. This catches trapped subgroups that the army-wide
    center average would miss (e.g. half the army is fine, half is surrounded).
    Squads with < 3 units are skipped to avoid triggering recall for lone scouts.

    Returns True if recall was cast, False otherwise.
    """
    # Clear per-frame debug state
    if bot.debug:
        bot._recall_escape_samples = []
        bot._recall_blocked_reason = None
        bot._recall_trapped_tags = set()
        bot._recall_influenced_tags = set()
        bot._recall_target_pos = None

    own_supply = sum(bot.calculate_supply_cost(u.type_id) for u in main_army)

    if not bot.townhalls:
        bot._mass_recall_pending = False
        return False

    # Gate 3 (per-squad): find first squad whose retreat is blocked
    active_squads: list[UnitSquad] = list(squads) if squads else []
    if not active_squads:
        fetched = bot.mediator.get_squads(
            role=UnitRole.ATTACKING, squad_radius=ATTACKING_SQUAD_RADIUS,
        )
        if fetched:
            active_squads = list(fetched)
    trapped_squad: UnitSquad | None = None
    trapped_base = None
    for squad in active_squads:
        if len(squad.squad_units) < 3:
            continue
        squad_center = squad.squad_position
        nearest_base = cy_closest_to(squad_center, bot.townhalls)
        # Clear per-squad debug samples so only the trapped squad's data persists
        if bot.debug:
            bot._recall_escape_samples = []
        if _is_retreat_blocked(bot, squad_center, nearest_base.position):
            trapped_squad = squad
            trapped_base = nearest_base
            break

    retreat_blocked = trapped_squad is not None

    bot._mass_recall_gates = {
        "g1_loss": fight_result in LOSS_DECISIVE_OR_WORSE,
        "g2_supply": own_supply >= MASS_RECALL_MIN_OWN_SUPPLY,
        "g3_retreat_blocked": retreat_blocked,
        "fight_result": fight_result.name if hasattr(fight_result, 'name') else str(fight_result),
        "own_supply": own_supply,
        "trapped_squad_size": len(trapped_squad.squad_units) if trapped_squad else 0,
    }

    # Gate 1: Must be a devastating loss (decisive or worse)
    if fight_result not in LOSS_DECISIVE_OR_WORSE:
        bot._mass_recall_pending = False
        return False

    # Gate 2: Enough army to be worth saving
    if own_supply < MASS_RECALL_MIN_OWN_SUPPLY:
        bot._mass_recall_pending = False
        return False

    # Gate 3: At least one squad's retreat path is blocked
    if not retreat_blocked:
        bot._mass_recall_pending = False
        return False

    # ALL COMBAT GATES PASSED — find best recall target and delegate to structure_manager
    recall_target = _find_recall_target(bot, main_army)
    if bot.debug:
        bot._recall_target_pos = recall_target
    cast_result = use_mass_recall(bot, recall_target, avoid_nexus_tag=trapped_base.tag)
    bot._mass_recall_pending = False

    return cast_result


def handle_attack_toggles(
    bot, main_army: Units, attack_target: Point2,
    squads: list[UnitSquad] | None = None,
) -> Point2:
    """
    Enhanced attack toggles logic with smart threat response.
    
    Decides when to attack and when to retreat based on relative army strength
    and intelligent threat assessment that prevents bouncing.
    
    Returns the target position where army should move (attack target or retreat position).
    DOES NOT control units - only sets flags and returns target.
    """
    
    # When under attack, redirect entire army to defend
    # This takes priority - army moves cohesively to threat position
    if bot._under_attack and hasattr(bot, '_defender_threat_position') and bot._defender_threat_position:
        bot._commenced_attack = False  # Cancel any ongoing attack
        return bot._defender_threat_position
    
    # Assess the threat level of enemy units for main combat decisions
    enemy_threat_level = assess_threat(bot, bot.enemy_units, main_army)  # Returns simple int
    # Type assertion since return_details=False (default) returns int
    assert isinstance(enemy_threat_level, int), "assess_threat without return_details should return int"

    # Composition belief: use probability-weighted unit filtering when enabled
    use_belief = bot.config.get("Belief", {}).get("enable_composition", False)

    enemy_army = bot.enemy_army
    # Early game safety - don't attack during cheese reactions
    # ReactionManager handles transition; is_early_defensive is False once transitioned
    is_early_defensive_mode = bot.reaction_manager.is_early_defensive
    
    # Debug visualization (controlled by bot.debug flag)
    render_combat_state_overlay(bot, main_army, enemy_threat_level, is_early_defensive_mode)
    render_mass_recall_debug(bot, main_army)
    render_target_scoring_debug(bot, main_army)
    render_disruptor_labels(bot)
    render_nova_labels(bot, getattr(bot, 'nova_manager', None))
    render_base_defender_debug(bot)
    render_observer_debug(bot)
    render_repair_debug(bot)

    # Siege tanks: combat sim underestimates splash/positioning, so we elevate the
    # required sim threshold to VICTORY_MARGINAL_OR_BETTER instead of bypassing the
    # sim entirely with a raw supply ratio. This lets composition (e.g. Immortals)
    # win the sim while still demanding a safety margin into tanks.
    has_siege_tanks = bool(bot.enemy_units({UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED}))

    # Evaluate current attack state with engagement hysteresis
    if bot._commenced_attack:
        # Maintain commitment for minimum duration to prevent rapid oscillation
        if bot.time < bot._attack_commenced_time + STAY_AGGRESSIVE_DURATION:
            return attack_target
        # After minimum duration, check if we should retreat
        # For early defensive mode (cheese), check for active engagement before retreating
        elif is_early_defensive_mode:
            # Check for active engagement before retreating (prevents bouncing during cheese)
            enemies_near_army = bot.mediator.get_units_in_range(
                start_points=[main_army.center],
                distances=15,
                query_tree=UnitTreeQueryType.AllEnemy,
                return_as_dict=False,
            )[0].filter(lambda u: not u.is_memory and not u.is_structure)
            
            if enemies_near_army:
                # In active combat - check if we're losing badly enough for mass recall
                if use_belief:
                    weighted = bot.belief_state.composition.get_weighted_army(
                        exclude_workers=True, exclude_structures=True, exclude_ignored=True,
                        min_confidence=0.05,
                    )
                    combat_enemy_units = [wu.unit for wu in weighted]
                else:
                    combat_enemy_units = [
                        u for u in bot.mediator.get_cached_enemy_army
                        if u.type_id not in WORKER_TYPES and not u.is_structure
                    ]
                # Static D fights too — a finished bunker-rush bunker is a real
                # threat, and garrison fire decides this local fight
                combat_enemy_units += get_sim_static_defense(bot)
                fight_result = bot.mediator.can_win_fight(
                    own_units=main_army, enemy_units=combat_enemy_units,
                    workers_do_no_damage=True,
                )
                if fight_result in LOSS_DECISIVE_OR_WORSE:
                    if try_mass_recall(bot, main_army, fight_result, squads=squads):
                        bot._commenced_attack = False
                        if bot.townhalls:
                            return cy_closest_to(main_army.center, bot.townhalls).position
                        return bot.start_location
                # Maintain current engagement for stability
                return attack_target
            else:
                # No nearby enemies - safe to retreat
                bot._commenced_attack = False
                if bot.townhalls:
                    return cy_closest_to(main_army.center, bot.townhalls).position
                else:
                    return bot.start_location
        # For normal mode, re-evaluate fight result after minimum duration
        else:
            # Filter enemy units: exclude workers and structures
            # Use belief-weighted filtering when enabled
            if use_belief:
                weighted = bot.belief_state.composition.get_weighted_army(
                    exclude_workers=True, exclude_structures=True, exclude_ignored=True,
                    min_confidence=0.05,
                )
                combat_enemy_units = [wu.unit for wu in weighted]
            else:
                combat_enemy_units = [
                    u for u in bot.mediator.get_cached_enemy_army
                    if u.type_id not in WORKER_TYPES and not u.is_structure
                ]
            # Static D joins the retreat re-evaluation — fortified positions
            # should push us out even if the mobile army died
            combat_enemy_units += get_sim_static_defense(bot)
            fight_result = bot.mediator.can_win_fight(
                own_units=main_army,
                enemy_units=combat_enemy_units,
                workers_do_no_damage=True,
            )
            # Disengage if situation looks bad (decisive loss or worse)
            if fight_result in LOSS_DECISIVE_OR_WORSE:
                # Try mass recall first — if army is trapped and losing badly, recall out
                if try_mass_recall(bot, main_army, fight_result, squads=squads):
                    bot._commenced_attack = False
                    if bot.townhalls:
                        return cy_closest_to(main_army.center, bot.townhalls).position
                    return bot.start_location
                bot._commenced_attack = False
                if bot.townhalls:
                    return cy_closest_to(main_army.center, bot.townhalls).position
                else:
                    return bot.start_location
            else:
                return attack_target
    else:
        # Not currently attacking: evaluate if we should initiate attack
        
        # Max supply fallback: force attack when capped
        if bot.supply_used >= 199:
            bot._commenced_attack = True
            bot._attack_commenced_time = bot.time
            return attack_target
        
        # === INTEL QUALITY GATE ===
        # Don't trust combat sim if we haven't seen enemy army or info is too stale
        intel = get_enemy_intel_quality(bot)
        
        # Gate 1: Must have seen enemy army at least once
        if not intel["has_intel"]:
            # Combat sim has ZERO enemy data - don't trust it, stay defensive
            return select_defensive_anchor(bot, main_army)
        
        # Gate 2: Very stale intel (<0.05) - genuinely blind, don't initiate attacks
        # At 0.2 this blocked too aggressively; a brief scout glimpse would push freshness
        # above 0.2, then decay below within ~10s, locking out attacks for most of the game.
        # When composition belief is enabled, use its freshness (probability-weighted)
        # instead of the binary age-threshold freshness.
        freshness = intel["freshness"]
        if use_belief:
            freshness = bot.belief_state.composition.freshness
        if freshness < STALE_INTEL_THRESHOLD:
            return select_defensive_anchor(bot, main_army)
        
        # Filter enemy units: exclude workers and structures
        # When composition belief is enabled, use confidence-weighted filtering
        # instead of the raw cache (which includes stale ghost units at full weight)
        if use_belief:
            weighted = bot.belief_state.composition.get_weighted_army(
                exclude_workers=True, exclude_structures=True, exclude_ignored=True,
                min_confidence=0.05,
            )
            combat_enemy_units = [wu.unit for wu in weighted]
        else:
            # Original: use all cached units regardless of staleness
            combat_enemy_units = [
                u for u in bot.mediator.get_cached_enemy_army
                if u.type_id not in WORKER_TYPES and not u.is_structure
            ]
        # Static D joins the attack-initiate gate — stops walking armies into
        # fortified naturals (bunkers/cannons/spines count even if the enemy
        # mobile army is elsewhere)
        combat_enemy_units += get_sim_static_defense(bot)
        fight_result = bot.mediator.can_win_fight(
            own_units=main_army,
            enemy_units=combat_enemy_units,
            workers_do_no_damage=True,
        )
        # Determine required sim result based on conditions:
        # - Cheese defense or siege tanks present: VICTORY_MARGINAL_OR_BETTER (safety margin)
        # - Normal mode: TIE_OR_BETTER (attack when sim says even or better)
        # Siege tanks: combat sim underestimates splash/positioning, so demand a margin.
        # Intel freshness above STALE_INTEL_THRESHOLD (0.05) no longer inflates the
        # threshold — with some intel, TIE_OR_BETTER is reasonable (the sim already
        # overestimates enemy strength from cached units).
        required_result = VICTORY_MARGINAL_OR_BETTER if (is_early_defensive_mode or has_siege_tanks) else TIE_OR_BETTER
        
        if fight_result in required_result:
            bot._commenced_attack = True
            bot._attack_commenced_time = bot.time
            return attack_target
        
        # Default: use strategic anchor positioning (smart defensive placement)
        return select_defensive_anchor(bot, main_army)


def attack_target(bot, main_army_position: Point2) -> Point2:
    """
    Determines where our main attacking units should move toward.
    Uses targeting hierarchy with stable army center mass for proximity checks.
    """
    
    # Get stable army center mass
    attacking_units = bot.mediator.get_units_from_role(role=UnitRole.ATTACKING)
    if attacking_units:
        own_center_mass, num_own = cy_find_units_center_mass(attacking_units, 10)
        own_center_mass = Point2(own_center_mass)
    else:
        own_center_mass = main_army_position
    
    # 1. Calculate structure position with enhanced stability
    enemy_structure_pos: Point2 | None = None
    if bot.enemy_structures:
        valid_structures = bot.enemy_structures.filter(lambda s: s.type_id not in COMMON_UNIT_IGNORE_TYPES)
        if not valid_structures:
            valid_structures = bot.enemy_structures
        if valid_structures:
            # Prefer current target if it's still valid and among the closest structures
            if (bot.current_attack_target and 
                valid_structures.filter(lambda s: cy_distance_to_squared(s.position, bot.current_attack_target) < 25.0)):
                # Current target is still among valid structures, keep it for stability
                enemy_structure_pos = bot.current_attack_target
            else:
                # Use standard closest to enemy natural logic
                enemy_structure_pos = cy_closest_to(bot.mediator.get_enemy_nat, valid_structures).position
    
    # 2. Proximity stickiness - "idea here is if we are near enemy structures/production, don't get distracted"
    if (enemy_structure_pos and 
        cy_distance_to_squared(own_center_mass, enemy_structure_pos) < PROXIMITY_STICKY_DISTANCE_SQ):
        bot.current_attack_target = enemy_structure_pos
        return enemy_structure_pos
    
    # 3. Army targeting with supply thresholds
    enemy_army = bot.enemy_units.filter(lambda u: 
        u.type_id not in COMMON_UNIT_IGNORE_TYPES 
        and not u.is_structure 
        and not u.is_burrowed 
        and not u.is_cloaked
    )
    enemy_supply = sum(bot.calculate_supply_cost(u.type_id) for u in enemy_army)
    
    # Only prioritize army if substantial
    if enemy_supply >= 6 and enemy_army:
        enemy_center, _ = cy_find_units_center_mass(enemy_army, 10.0)
        
        all_close_enemy = bot.mediator.get_units_in_range(
            start_points=[Point2(enemy_center)],
            distances=11.5,
            query_tree=UnitTreeQueryType.EnemyGround,
            return_as_dict=False,
        )[0]
        clustered_supply = sum(bot.calculate_supply_cost(u.type_id) for u in all_close_enemy)
        
        # Only target army if clustered supply is significant
        if clustered_supply >= 7:
            bot.current_attack_target = Point2(enemy_center)
            return bot.current_attack_target
    
    # 4. Structure fallback
    if enemy_structure_pos:
        bot.current_attack_target = enemy_structure_pos
        return enemy_structure_pos
    
    # 5. Expansion cycling when no targets
    if bot.is_visible(bot.current_attack_target) if bot.current_attack_target else True:
        fallback = fallback_target(bot)
        bot.current_attack_target = fallback
        return fallback
    
    # 6. Keep current target if area not scouted yet
    if bot.current_attack_target:
        return bot.current_attack_target
    
    # 7. Final fallback
    fallback = bot.enemy_start_locations[0] if bot.time < 240.0 else fallback_target(bot)
    bot.current_attack_target = fallback
    return fallback


def fallback_target(bot) -> Point2:
    """
    Provides a fallback target for the main army when no clear target is available.
    Cycles through expansion locations with mineral field validation.
    """
    # Cycle through expansion locations with better validation
    if bot.is_visible(bot.current_base_target):
        # Check if there are mineral fields near this base location
        if mineral_fields := [
            mf for mf in bot.mineral_field 
            if cy_distance_to_squared(mf.position, bot.current_base_target) < 144.0
        ]:
            closest_mineral = cy_closest_to(bot.current_base_target, mineral_fields).position
            if bot.is_visible(closest_mineral):
                bot.current_base_target = next(bot.expansions_generator)
        else:
            bot.current_base_target = next(bot.expansions_generator)
    
    return bot.current_base_target


def control_defenders(bot) -> None:
    """
    Orchestration function called from bot.py on_step.
    Gets all BASE_DEFENDER units and controls them every frame.
    """
    defenders = bot.mediator.get_units_from_role(role=UnitRole.BASE_DEFENDER)
    if not defenders:
        return
    
    # Get threat position set by threat_detection in reactions.py
    threat_position = getattr(bot, '_defender_threat_position', None)
    if threat_position is None:
        # No active threat - move defenders to rally point
        threat_position = bot.main_base_ramp.top_center if bot.main_base_ramp else bot.start_location
    
    control_base_defenders(bot, defenders, threat_position)


def control_base_defenders(bot, defender_units: Units, threat_position: Point2) -> None:
    """
    Controls BASE_DEFENDER units using individual behaviors with squad formation.
    Uses the same micro functions as main army for consistent combat behavior.
    """
    # Wrap in try-except to handle ARES squad manager KeyError when units die mid-frame
    try:
        defensive_squads = bot.mediator.get_squads(role=UnitRole.BASE_DEFENDER, squad_radius=DEFENDER_SQUAD_RADIUS)
    except KeyError:
        # ARES internal tracking got out of sync - unit died but tag wasn't cleaned up
        # Fall back to using defender_units directly without squad grouping
        defensive_squads = None
    
    if not defensive_squads:
        return
        
    grid: np.ndarray = bot.mediator.get_ground_grid
    avoid_grid: np.ndarray = bot.mediator.get_ground_avoidance_grid
    
    for squad in defensive_squads:
        squad_position: Point2 = squad.squad_position
        units: list[Unit] = squad.squad_units
        
        # Skip empty squads
        if not units:
            continue
        
        # Find nearby enemy units using per-unit detection (larger range, checks each unit)
        unit_positions = [u.position for u in units]
        all_close_results = bot.mediator.get_units_in_range(
            start_points=unit_positions,
            distances=UNIT_ENEMY_DETECTION_RANGE,
            query_tree=UnitTreeQueryType.AllEnemy,
            return_as_dict=False,
        )
        # Combine all detected enemies and deduplicate by tag
        seen_tags = set()
        all_close_list = []
        for result in all_close_results:
            for u in result:
                if u.tag not in seen_tags and not u.is_memory and not u.is_structure and u.type_id not in COMMON_UNIT_IGNORE_TYPES:
                    seen_tags.add(u.tag)
                    all_close_list.append(u)
        all_close = Units(all_close_list, bot)
        
        if not all_close:
            # No enemies - move towards threat position
            for unit in units:
                move_maneuver = CombatManeuver()
                move_maneuver.add(AMove(unit=unit, target=threat_position))
                bot.register_behavior(move_maneuver)
            continue
        
        # Separate unit types for appropriate micro
        # Ground combat units only (air units get their own micro)
        melee = [u for u in units if u.ground_range <= MELEE_RANGE_THRESHOLD and u.energy == 0 and u.can_attack and not u.is_flying]
        ranged = [u for u in units if u.ground_range > MELEE_RANGE_THRESHOLD and u.energy == 0 and u.can_attack and not u.is_flying]
        air = [u for u in units if u.is_flying and u.can_attack]
        spellcasters = [u for u in units if u.energy > 0 or u.type_id == UnitTypeId.DISRUPTOR]
        
        # Compute enemy center for stalker blink targeting
        enemy_center_mass, _ = cy_find_units_center_mass(all_close, 20.0)
        enemy_center = Point2(enemy_center_mass)
        
        # --- Blink snipe/focus/chase for defenders ---
        # Same group micro as main army — defenders face the same harassers.
        squad_stalkers = [
            u for u in units
            if u.type_id == UnitTypeId.STALKER
            and u.tag not in bot._snipe_committed
            and u.tag not in bot._chase_committed
            and u.tag not in bot._focus_committed
        ]
        if squad_stalkers and all_close:
            try_commit_snipe(
                bot=bot,
                squad_id=squad.squad_id,
                enemies=all_close,
                squad_stalkers=squad_stalkers,
                squad_position=squad_position,
            )
        # Execute active snipe (runs every frame for committed squads)
        snipe_info = bot._snipe_state.get(squad.squad_id)
        if snipe_info and snipe_info.get("mode") == "b":
            execute_snipe_b(bot, squad.squad_id, grid)
        else:
            execute_snipe_a(bot, squad.squad_id, grid)
        render_snipe_debug(bot, squad.squad_id)

        # Execute active focus-fire (runs every frame for committed squads)
        execute_focus(bot, squad.squad_id, grid, all_close)
        render_focus_debug(bot, squad.squad_id)

        # Chase retreating enemies if we're winning
        chase_stalkers = [
            u for u in units
            if u.type_id == UnitTypeId.STALKER
            and u.tag not in bot._snipe_committed
            and u.tag not in bot._chase_committed
            and u.tag not in bot._focus_committed
        ]
        if chase_stalkers and all_close:
            try_commit_chase(
                bot=bot,
                squad_id=squad.squad_id,
                enemies=all_close,
                squad_stalkers=chase_stalkers,
                squad_position=squad_position,
                can_engage=True,  # Defenders are always engaging
            )
        execute_chase(bot, squad.squad_id, grid, all_close)
        render_chase_debug(bot, squad.squad_id)

        # Ranged micro - weighted scoring handles priority targeting
        for r_unit in ranged:
            # Filter enemies to only those this unit can attack and reach
            unit_enemies = get_attackable_enemies(r_unit, all_close, grid)
            if not unit_enemies:
                continue
            
            # Skip stalkers committed to group snipe, chase, or focus-fire
            if r_unit.type_id == UnitTypeId.STALKER:
                if r_unit.tag in bot._snipe_committed and bot.state.game_loop < bot._snipe_committed[r_unit.tag]:
                    continue
                if r_unit.tag in bot._chase_committed and bot.state.game_loop < bot._chase_committed[r_unit.tag]:
                    continue
                if r_unit.tag in bot._focus_committed and bot.state.game_loop < bot._focus_committed[r_unit.tag]:
                    continue
            
            # Stalkers get blink-aware micro when low health
            if r_unit.type_id == UnitTypeId.STALKER:
                ranged_maneuver = micro_stalker(
                    stalker=r_unit,
                    enemies=unit_enemies,
                    grid=grid,
                    avoid_grid=avoid_grid,
                    aggressive=True,  # Always aggressive when defending base
                    squad_center=squad_position,
                    enemy_center=enemy_center,
                    ranged_units=ranged,
                    all_close=all_close,
                    bot=bot,
                )
            else:
                ranged_maneuver = micro_ranged_unit(
                    unit=r_unit,
                    enemies=unit_enemies,
                    grid=grid,
                    avoid_grid=avoid_grid,
                    aggressive=True  # Always aggressive when defending base
                )
            bot.register_behavior(ranged_maneuver)

            # Debug: show micro state above base defender ranged units
            from bot.combat.unit_micro import _has_melee_threat, _is_targeted_by_ranged
            is_threatened = _has_melee_threat(r_unit, unit_enemies) or _is_targeted_by_ranged(r_unit, unit_enemies)
            render_micro_state_debug(
                bot, r_unit,
                is_threatened=is_threatened,
                aggressive=True,  # Always aggressive when defending base
                weapon_ready=r_unit.weapon_ready,
                in_range=bool(cy_in_attack_range(r_unit, unit_enemies)),
            )
        # Air micro - separate handling for flying combat units
        for a_unit in air:
            unit_enemies = get_attackable_enemies(a_unit, all_close, grid)
            if not unit_enemies:
                continue
            air_maneuver = micro_air_unit(
                unit=a_unit,
                enemies=unit_enemies,
                avoid_grid=bot.mediator.get_air_avoidance_grid,
                aggressive=True,  # Always aggressive when defending base
            )
            bot.register_behavior(air_maneuver)

        # Melee micro - weighted scoring handles priority targeting
        for m_unit in melee:
            # Filter enemies to only those this unit can attack and reach
            unit_enemies = get_attackable_enemies(m_unit, all_close, grid)
            if not unit_enemies:
                continue
            
            melee_maneuver = micro_melee_unit(
                unit=m_unit,
                enemies=unit_enemies,
                avoid_grid=avoid_grid,
                grid=grid,
                fallback_position=threat_position,
                aggressive=True  # Always aggressive when defending base
            )
            bot.register_behavior(melee_maneuver)
        
        # Spellcasters - stay safe, move with defenders
        for caster in spellcasters:
            caster_maneuver = CombatManeuver()
            caster_maneuver.add(KeepUnitSafe(unit=caster, grid=avoid_grid))
            caster_maneuver.add(AMove(unit=caster, target=threat_position))
            bot.register_behavior(caster_maneuver)


def select_defensive_anchor(bot, main_army: Units) -> Point2:
    """
    Selects the best strategic anchor position for army placement when not attacking.
    Uses predefined anchors with 15-second cooldown to prevent oscillation.
    
    Priority cascade:
    1. Mid/late game OR 2+ bases: Position at base closest to enemy
    2. Gatekeeper exists: Position behind gatekeeper (avoids bunching)
    3. Structures at natural: Position forward of natural
    4. Fallback: Position back from main ramp
    """
    ANCHOR_CHANGE_COOLDOWN = 15.0  # Seconds between anchor changes
    NATURAL_STRUCTURE_RADIUS = 400.0  # 20 units radius for structure check
    NATURAL_BATTERY_RADIUS_SQ = 400.0  # 20 units radius for battery-natural check
    
    # Check if we're still in cooldown (prevent rapid switching)
    if bot._current_defensive_anchor is not None:
        if bot.time < bot._anchor_change_time + ANCHOR_CHANGE_COOLDOWN:
            return bot._current_defensive_anchor
    
    # Determine new anchor position
    new_anchor = None
    
    # A Shield Battery at the natural with no Nexus there is still a defensive
    # anchor worth rallying to (e.g. cheese response holding the natural).
    battery_at_natural = bot.structures(UnitTypeId.SHIELDBATTERY).ready.filter(
        lambda b: cy_distance_to_squared(b.position, bot.natural_expansion) < NATURAL_BATTERY_RADIUS_SQ
    )
    nat_has_townhall = bot.townhalls.closer_than(10, bot.natural_expansion).amount > 0
    
    # PHASE 1: Mid/late game OR successful early expansion (>= 2 ready townhalls)
    if bot.game_state >= 1 or len(bot.townhalls.ready) >= 2:
        # If we hold the natural with a townhall, rally to the base closest to enemy
        if bot.townhalls.ready:
            enemy_start = bot.enemy_start_locations[0]
            # Battery at natural with no Nexus there -> rally to the battery instead
            if battery_at_natural and not nat_has_townhall:
                new_anchor = battery_at_natural.closest_to(bot.natural_expansion).position.towards(enemy_start, 3)
            else:
                closest_base = bot.townhalls.ready.closest_to(enemy_start)
                # Offset 3 units towards enemy start for forward positioning
                new_anchor = closest_base.position.towards(enemy_start, 3)
    
    # PHASE 2: Early game single-base positioning
    else:
        # Check for gatekeeper position (best choke control)
        if hasattr(bot, 'gatekeeping_pos') and bot.gatekeeping_pos is not None:
            # Position behind gatekeeper (direction depends on opponent race)
            # PvP (main ramp): position towards start location
            # PvZ/Random (natural choke): position towards natural
            if bot.enemy_race == Race.Protoss:
                new_anchor = bot.gatekeeping_pos.towards(bot.start_location, 4)
            else:
                new_anchor = bot.gatekeeping_pos.towards(bot.natural_expansion, 4)
        
        # Check for any structures at natural (shows operational investment)
        elif bot.structures:
            structures_at_natural = bot.structures.filter(lambda s:
                cy_distance_to_squared(s.position, bot.natural_expansion) < NATURAL_STRUCTURE_RADIUS
            )
            if structures_at_natural:
                # Position forward of natural towards enemy
                enemy_start = bot.enemy_start_locations[0]
                new_anchor = bot.natural_expansion.towards(enemy_start, 3)
        
        # Fallback: safe position back from main ramp
        if new_anchor is None:
            # Position back from main towards start location (safe fallback)
            main_ramp = bot.main_base_ramp.top_center
            new_anchor = main_ramp.towards(bot.start_location, 3)
    
    # Ensure we always have a valid anchor (final fallback)
    if new_anchor is None:
        new_anchor = bot.start_location
    
    # Update anchor if it changed
    if new_anchor != bot._current_defensive_anchor:
        bot._current_defensive_anchor = new_anchor
        bot._anchor_change_time = bot.time
        
        # Sync rally_point to anchor (single source of truth)
        bot.rally_point = new_anchor
        
        # Update all gateway rally points to new anchor
        for gateway in bot.structures(UnitTypeId.GATEWAY).ready:
            gateway(AbilityId.RALLY_BUILDING, new_anchor)
    
    return new_anchor


def manage_defensive_unit_roles(bot) -> None:
    """
    Release BASE_DEFENDER units back to ATTACKING when threats clear.
    
    Uses ARES enemy tracking which has hysteresis (enemies must leave beyond
    24 units to clear). This prevents role oscillation from brief visibility gaps.
    """
    defending_units = bot.mediator.get_units_from_role(role=UnitRole.BASE_DEFENDER)
    
    if not defending_units:
        return
    
    ground_near = bot.mediator.get_ground_enemy_near_bases
    flying_near = bot.mediator.get_flying_enemy_near_bases
    
    # Check if ARES is tracking any enemies near bases
    active_threats = any(ground_near.values()) or any(flying_near.values())
    
    if not active_threats:
        for unit in defending_units:
            bot.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)


