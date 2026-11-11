"""Group blink-snipe and focus-fire helper functions for Stalker micro.

Purpose: Damage math, target evaluation, stalker selection, and coordinated execution.
    Snipe-A: walk-in, blink-out (safe, for targets we outrange or match).
    Snipe-B: blink-in, walk-out (riskier, for targets that outrange us).
    Focus: multi-volley pursuit (blink or walk in, stay on target until dead/abort).
Key Decisions: Reuses existing TACTICAL_BONUS + army_value for target value; uses python-sc2's
    calculate_damage_vs_target for exact damage math; cy_* for all hot-path geometry.
    Skip-set pattern (bot._snipe_committed / bot._focus_committed) prevents per-unit micro
    from overriding group commands.
Limitations: calculate_damage_vs_target doesn't model Guardian Shield on enemies we haven't
    attacked yet (BuffId check). Overkill buffer compensates.
"""

from math import ceil
from typing import Optional, Union

import numpy as np

from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.group import AMoveGroup, GroupUseAbility, PathGroupToTarget, StutterGroupForward
from ares.dicts.unit_data import UNIT_DATA

from cython_extensions import (
    cy_center,
    cy_closer_than,
    cy_distance_to,
    cy_in_attack_range,
    cy_range_vs_target,
    cy_sorted_by_distance_to,
    cy_towards,
)
from cython_extensions.numpy_helper import cy_all_points_below_max_value, cy_point_below_value

from bot.combat.scoring_table import (
    DEFAULT_TYPE_VALUE,
    TYPE_VALUE_SCALE,
    base_unit_type,
    tactical_bonus,
)
from ares.consts import EngagementResult

from bot.constants import (
    SNIPE_MIN_TARGET_VALUE,
    SNIPE_OVERKILL_BUFFER_MOBILE,
    SNIPE_OVERKILL_BUFFER_STATIC,
    SNIPE_EXIT_FRAMES,
    SNIPE_COMMIT_COOLDOWN,
    SNIPE_APPROACH_RANGE_BUFFER,
    STALKER_BLINK_RANGE,
    TACTICAL_ESCAPE_MAX,
    CHASE_TACTICAL_MAX,
    CHASE_BLINK_GAP_THRESHOLD,
    FOCUS_MIN_STALKERS,
)

# Combat sim results that allow snipe commitment
SNIPE_SIM_THRESHOLD = {EngagementResult.TIE, EngagementResult.VICTORY_MARGINAL,
                       EngagementResult.VICTORY_CLOSE, EngagementResult.VICTORY_DECISIVE,
                       EngagementResult.VICTORY_OVERWHELMING, EngagementResult.VICTORY_EMPHATIC}

# Combat sim results that allow focus-fire commitment (more permissive — smaller commitment)
FOCUS_SIM_THRESHOLD = {EngagementResult.TIE, EngagementResult.VICTORY_MARGINAL,
                      EngagementResult.VICTORY_CLOSE, EngagementResult.VICTORY_DECISIVE,
                      EngagementResult.VICTORY_OVERWHELMING, EngagementResult.VICTORY_EMPHATIC}

# Combat sim results that trigger focus-fire abort (tide turned — disengage)
FOCUS_ABORT_SIM = {EngagementResult.LOSS_DECISIVE, EngagementResult.LOSS_OVERWHELMING,
                   EngagementResult.LOSS_EMPHATIC}

# Max grid value along the approach corridor (before firing range).
# Safe cell = 1.0; enemy influence adds cost. ~15 allows walking past
# light threats but blocks marching through a full army.
SNIPE_CORRIDOR_MAX_VALUE = 15.0
SNIPE_CORRIDOR_SAMPLES = 4  # number of evenly-spaced sample points




def effective_value(target: Unit) -> float:
    """Compute the effective tactical value of an enemy unit.

    Combines ARES army_value (economic) with TACTICAL_BONUS (gameplay impact).
    Same formula used in score_target() for consistency. A state variant
    (burrowed Roach, Ghost in nova mode) resolves to its base unit.

    Perf: ~200ns (dict lookups only, no iteration).
    """
    data = UNIT_DATA.get(target.type_id)
    if data is None:
        data = UNIT_DATA.get(base_unit_type(target.type_id))
    base = (data["army_value"] * TYPE_VALUE_SCALE) if data else DEFAULT_TYPE_VALUE
    return base + tactical_bonus(target.type_id)


def damage_per_volley(stalker: Unit, target: Unit) -> float:
    """Compute the damage a single stalker deals to target in one attack cycle.

    Uses python-sc2's calculate_damage_vs_target which accounts for:
    - Upgrades (attack + armor)
    - Bonus damage vs armor types (e.g., Stalker +bonus vs Armored)
    - Guardian Shield buff on target
    - Multi-attack weapons
    - Air/ground targeting

    Returns 0.0 if stalker can't attack the target.

    Perf: ~2µs (python-sc2 property access + weapon iteration).
    """
    dmg, _speed, _range = stalker.calculate_damage_vs_target(target)
    return float(dmg)


def stalkers_needed_to_kill(
    stalker: Unit,
    target: Unit,
) -> int:
    """Calculate the minimum number of stalkers needed to kill target in one volley.

    Uses a representative stalker for damage calculation (all stalkers in a squad
    should have the same upgrades). Adds an overkill buffer: +1 for mobile targets
    (they might dodge), +0 for static targets (sieged tanks, burrowed units).

    Returns 0 if stalker can't attack the target (e.g., can't hit air).

    Perf: ~2µs (one calculate_damage_vs_target call + arithmetic).
    """
    dmg = damage_per_volley(stalker, target)
    if dmg <= 0:
        return 0

    target_hp = target.health + target.shield
    raw_needed = ceil(target_hp / dmg)

    # Static targets (sieged, burrowed, phased) can't dodge — no overkill buffer needed
    buffer = (
        SNIPE_OVERKILL_BUFFER_STATIC
        if target.movement_speed == 0
        else SNIPE_OVERKILL_BUFFER_MOBILE
    )
    return raw_needed + buffer


def find_snipe_candidates(
    stalkers: list[Unit],
    target: Unit,
    needed: int,
    max_range: float = 20.0,
) -> list[Unit]:
    """Select the minimum stalkers needed for a snipe, from those eligible.

    Filters:
    - Blink ready (EFFECT_BLINK_STALKER in abilities)
    - Within max_range of the target

    Weapons are not checked here — stalkers use pure move during approach,
    so any prior cooldowns will expire before they reach firing range.

    Sorts by: distance to target (closest first) to minimize travel time.
    Returns up to `needed` stalkers. Returns empty list if not enough qualify.

    Perf: O(n) filter + O(n log n) sort where n = stalkers in range. Typically n < 20.
    """
    nearby: list[Unit] = cy_closer_than(stalkers, max_range, target.position)

    eligible: list[Unit] = [
        s for s in nearby
        if AbilityId.EFFECT_BLINK_STALKER in s.abilities
    ]

    if len(eligible) < needed:
        return []

    sorted_by_dist: list[Unit] = cy_sorted_by_distance_to(eligible, target.position)
    return sorted_by_dist[:needed]


def is_snipe_a_eligible(stalker_sample: Unit, target: Unit) -> bool:
    """Check if Snipe-A (walk-in, blink-out) is viable for this target.

    Snipe-A requires we can walk into weapon range without being out-ranged.
    True when target's weapon range ≤ stalker range + buffer.

    Perf: ~300ns (two cy_range_vs_target calls).
    """
    our_range = cy_range_vs_target(stalker_sample, target)
    their_range = cy_range_vs_target(target, stalker_sample)
    return their_range <= our_range + SNIPE_APPROACH_RANGE_BUFFER


def _is_corridor_safe(
    bot,
    squad_center: Union[Point2, tuple[float, float]],
    target_pos: Point2,
    stalker_sample: Unit,
    target: Unit,
) -> bool:
    """Check if the approach corridor is safe enough for a snipe walk-in.

    Samples points along the path from squad_center to the edge of firing
    range around the target. The last ~7 tiles near the target (where the
    stalkers will actually shoot) are excluded — danger there is expected.

    If the stalkers are already within firing range, the corridor is empty
    and this always returns True.

    Perf: O(SNIPE_CORRIDOR_SAMPLES) grid lookups (~4 us total).
    """
    firing_range = cy_range_vs_target(stalker_sample, target) + 1.0
    sc = Point2(squad_center) if not isinstance(squad_center, Point2) else squad_center
    total_dist = cy_distance_to(sc, target_pos)

    # Already close enough — no corridor to check
    corridor_length = total_dist - firing_range
    if corridor_length <= 2.0:
        return True

    # Sample evenly-spaced points along the corridor (before firing range)
    grid: np.ndarray = bot.mediator.get_ground_grid
    dx = target_pos.x - sc.x
    dy = target_pos.y - sc.y
    # Normalize direction
    inv_dist = 1.0 / total_dist
    dx *= inv_dist
    dy *= inv_dist

    sample_points: list[tuple[int, int]] = []
    for i in range(1, SNIPE_CORRIDOR_SAMPLES + 1):
        t = (corridor_length * i / (SNIPE_CORRIDOR_SAMPLES + 1))
        px = int(sc.x + dx * t)
        py = int(sc.y + dy * t)
        sample_points.append((px, py))

    return cy_all_points_below_max_value(
        grid, SNIPE_CORRIDOR_MAX_VALUE, sample_points
    )


def is_snipe_b_eligible(stalker_sample: Unit, target: Unit) -> bool:
    """Check if Snipe-B (blink-in, walk-out) is viable for this target.

    Snipe-B is for targets that outrange us — we blink in, fire, and walk out.
    True when target's weapon range > stalker range + buffer (inverse of Snipe-A).
    Also requires the target is within blink range from a reasonable distance.

    Perf: ~300ns (two cy_range_vs_target calls).
    """
    our_range = cy_range_vs_target(stalker_sample, target)
    their_range = cy_range_vs_target(target, stalker_sample)
    if their_range <= our_range + SNIPE_APPROACH_RANGE_BUFFER:
        return False  # Snipe-A handles this
    # Target must have meaningful range (no point blinking into melee-range stuff)
    return their_range > 0


def compute_blink_landing(
    squad_center: Union[Point2, tuple[float, float]],
    target_pos: Point2,
    stalker_range: float,
) -> Point2:
    """Compute the blink landing spot for Snipe-B.

    Landing spot is right on top of the target (~0.5 tiles away).
    This negates splash damage (e.g., siege tanks can't splash without
    hitting themselves) and removes any range advantage the target has.

    Perf: ~200ns (one cy_towards call).
    """
    # Land practically touching the target — inside minimum range of
    # splash units and too close for them to fire effectively
    land_distance = 0.5
    return Point2(cy_towards(target_pos, squad_center, land_distance))


def _is_escape_corridor_safe(
    bot,
    landing: Point2,
    retreat_pos: Point2,
) -> bool:
    """Check if the walk-out corridor is safe enough for Snipe-B retreat.

    Samples points along the path from landing back toward the retreat point.
    Uses the tactical ground grid — high values mean heavy enemy presence.
    The first ~3 tiles near the landing are excluded (danger expected at target).

    Perf: O(SNIPE_CORRIDOR_SAMPLES) grid lookups (~4 us total).
    """
    total_dist = cy_distance_to(landing, retreat_pos)
    if total_dist <= 3.0:
        return True  # Already very close, no corridor to check

    grid: np.ndarray = bot.mediator.get_ground_grid
    dx = retreat_pos.x - landing.x
    dy = retreat_pos.y - landing.y
    inv_dist = 1.0 / total_dist
    dx *= inv_dist
    dy *= inv_dist

    # Skip the first ~3 tiles (near target, expected to be hot)
    # Sample from ~3 tiles out to ~80% of the corridor
    skip_dist = 3.0
    corridor_length = total_dist - skip_dist

    if corridor_length <= 2.0:
        return True

    sample_points: list[tuple[int, int]] = []
    for i in range(1, SNIPE_CORRIDOR_SAMPLES + 1):
        t = skip_dist + (corridor_length * i / (SNIPE_CORRIDOR_SAMPLES + 1))
        px = int(landing.x + dx * t)
        py = int(landing.y + dy * t)
        sample_points.append((px, py))

    return cy_all_points_below_max_value(
        grid, TACTICAL_ESCAPE_MAX, sample_points
    )


def find_best_snipe_target(
    bot,
    enemies: Union[Units, list[Unit]],
    stalker_sample: Unit,
    squad_stalkers: list[Unit],
    squad_position: Point2,
    min_value: float = SNIPE_MIN_TARGET_VALUE,
    sim_enemies: Optional[list] = None,
) -> Optional[tuple[Unit, int, list[Unit], str]]:
    """Find the best target for a group blink snipe (Snipe-A, Snipe-B, or Focus).

    Evaluates all enemies by effective_value. Priority:
    1. Snipe (one-shot): mode "a" or "b" — safest, stalkers retreat after killing.
    2. Focus (multi-volley): mode "focus" — stalkers stay on target until dead/abort.

    For snipe targets, checks damage math + candidate count + combat sim + corridor safety.
    For focus targets, checks isolation + damage math + candidate count + combat sim + corridor safety.

    Returns (target, needed_count, selected_stalkers, mode) or None.
    mode is "a", "b", or "focus".

    Perf: O(e) value scan + O(1) damage calc + O(s) candidate selection + 1 combat sim.
    where e = enemies, s = squad stalkers. Typically e < 30, s < 15.
    """
    # (value, enemy, mode) — mode determined by range/speed comparison
    snipe_targets: list[tuple[float, Unit, str]] = []
    focus_targets: list[tuple[float, Unit, str]] = []

    for enemy in enemies:
        val = effective_value(enemy)
        if val < min_value:
            continue
        if is_snipe_a_eligible(stalker_sample, enemy):
            snipe_targets.append((val, enemy, "a"))
        elif is_snipe_b_eligible(stalker_sample, enemy):
            snipe_targets.append((val, enemy, "b"))
        # Focus-fire: eligible target type + isolated position
        elif is_focus_eligible(bot, stalker_sample, enemy):
            focus_targets.append((val, enemy, "focus"))

    # Sort snipe targets by value descending, then by current HP ascending
    snipe_targets.sort(key=lambda x: (x[0], -(x[1].health + x[1].shield)), reverse=True)
    # Sort focus targets by value descending
    focus_targets.sort(key=lambda x: (x[0], -(x[1].health + x[1].shield)), reverse=True)

    # Priority 1: Try snipe (one-shot) targets first — safer than focus-fire
    for _val, target, mode in snipe_targets:
        needed = stalkers_needed_to_kill(stalker_sample, target)
        if needed <= 0:
            continue

        candidates = find_snipe_candidates(
            squad_stalkers, target, needed,
        )
        if not candidates:
            continue

        # Combat sim gate: can the snipe squad survive against nearby enemies?
        # sim_enemies includes static defenses (cannons, bunkers, spines) that
        # all_close excludes — stalkers must know if blinking into cannon range
        sim_result = bot.mediator.can_win_fight(
            own_units=Units(candidates, bot),
            enemy_units=Units(sim_enemies if sim_enemies is not None else list(enemies), bot),
            workers_do_no_damage=True,
        )
        if sim_result not in SNIPE_SIM_THRESHOLD:
            continue

        squad_center = cy_center([c.position for c in candidates])

        if mode == "a":
            # Walk-in corridor safety
            if not _is_corridor_safe(bot, squad_center, target.position, stalker_sample, target):
                continue
        else:
            # Snipe-B: don't blink blind — landing must be visible
            stalker_range = cy_range_vs_target(stalker_sample, target)
            landing = compute_blink_landing(squad_center, target.position, stalker_range)
            if not bot.is_visible(landing):
                continue
            # Check escape corridor from landing back to squad
            if not _is_escape_corridor_safe(bot, landing, squad_position):
                continue

        return target, needed, candidates, mode

    # Priority 2: Try focus-fire (multi-volley) targets
    for _val, target, mode in focus_targets:
        needed = stalkers_needed_to_focus(stalker_sample, target)
        if needed <= 0:
            continue

        candidates = find_focus_candidates(
            squad_stalkers, target, needed,
        )
        if not candidates:
            continue

        # Combat sim gate: more permissive than snipe (TIE_OR_BETTER)
        # sim_enemies includes static defenses for survival evaluation
        sim_result = bot.mediator.can_win_fight(
            own_units=Units(candidates, bot),
            enemy_units=Units(sim_enemies if sim_enemies is not None else list(enemies), bot),
            workers_do_no_damage=True,
        )
        if sim_result not in FOCUS_SIM_THRESHOLD:
            continue

        squad_center = cy_center([c.position for c in candidates])

        # Corridor safety: approach corridor only (we're staying, no escape corridor)
        if not _is_corridor_safe(bot, squad_center, target.position, stalker_sample, target):
            continue

        # Blink-in targets: landing must be visible
        blink_in = should_blink_in(stalker_sample, target)
        if blink_in:
            stalker_range = cy_range_vs_target(stalker_sample, target)
            landing = compute_blink_landing(squad_center, target.position, stalker_range)
            if not bot.is_visible(landing):
                continue

        return target, needed, candidates, mode

    return None


# ===== SNIPE STATE MACHINE =====

def try_commit_snipe(
    bot,
    squad_id: str,
    enemies: Union[Units, list[Unit]],
    squad_stalkers: list[Unit],
    squad_position: Point2,
    sim_enemies: Optional[list] = None,
) -> bool:
    """Evaluate and commit a snipe (A or B) for this squad.

    Called once per squad per frame from control_main_army, BEFORE per-unit micro.
    If a snipe is committed, adds tags to bot._snipe_committed and creates
    a _snipe_state entry. Returns True if a snipe was committed this frame.

    Gate checks applied here:
    - Cooldown: no snipe if squad committed within SNIPE_COMMIT_COOLDOWN frames
    - No existing snipe in progress for this squad
    - Target found via find_best_snipe_target (value + range + damage math + combat sim)

    Perf: O(e + s) where e = enemies, s = stalkers. Typically < 50 units total.
    """
    game_loop = bot.state.game_loop

    # Cooldown check: don't re-commit too soon after a previous snipe
    last_snipe_commit = bot._snipe_squad_cooldown.get(squad_id, 0)
    if game_loop - last_snipe_commit < SNIPE_COMMIT_COOLDOWN:
        return False

    # Don't stack — only one active snipe or focus per squad
    if squad_id in bot._snipe_state:
        return False
    if squad_id in bot._focus_state:
        return False

    # Need at least one stalker with blink to even evaluate
    if not squad_stalkers:
        return False

    stalker_sample = squad_stalkers[0]

    result = find_best_snipe_target(
        bot=bot,
        enemies=enemies,
        stalker_sample=stalker_sample,
        squad_stalkers=squad_stalkers,
        squad_position=squad_position,
        sim_enemies=sim_enemies,
    )
    if result is None:
        return False

    target, needed, candidates, mode = result
    candidate_tags = {s.tag for s in candidates}

    if mode == "focus":
        # Focus-fire: multi-volley pursuit — different state dict and skip-set
        # TTL is generous — _cleanup_focus releases stalkers when done
        expiry = game_loop + 800  # ~36s safety net
        for tag in candidate_tags:
            bot._focus_committed[tag] = expiry

        # Precompute blink landing if target outranges or is faster
        blink_in = should_blink_in(stalker_sample, target)
        landing = None
        if blink_in:
            stalker_range = cy_range_vs_target(stalker_sample, target)
            squad_center = cy_center([c.position for c in candidates])
            landing = compute_blink_landing(squad_center, target.position, stalker_range)

        bot._focus_state[squad_id] = {
            "target_tag": target.tag,
            "stalker_tags": candidate_tags,
            "retreat_point": squad_position,
            "min_to_kill": needed,
            "mode": mode,
            "landing": landing,
            "blink_in": blink_in,
        }
    else:
        # Snipe (A or B): one-shot — original state dict and skip-set
        # Raw kill count (without overkill buffer) — minimum stalkers to one-shot
        dmg = damage_per_volley(stalker_sample, target)
        raw_needed = ceil((target.health + target.shield) / dmg) if dmg > 0 else needed

        # Commit: register in skip-set with TTL
        expiry = game_loop + SNIPE_EXIT_FRAMES + 200  # generous TTL for full lifecycle
        for tag in candidate_tags:
            bot._snipe_committed[tag] = expiry

        # For Snipe-B, precompute landing spot
        landing = None
        if mode == "b":
            stalker_range = cy_range_vs_target(stalker_sample, target)
            squad_center = cy_center([c.position for c in candidates])
            landing = compute_blink_landing(squad_center, target.position, stalker_range)

        # Record snipe state for this squad
        bot._snipe_state[squad_id] = {
            "target_tag": target.tag,
            "stalker_tags": candidate_tags,
            "retreat_point": squad_position,
            "commit_frame": game_loop,
            "min_to_kill": raw_needed,
            "mode": mode,
            "landing": landing,  # only used by Snipe-B
        }
        bot._snipe_squad_cooldown[squad_id] = game_loop

    return True


def execute_snipe_a(bot, squad_id: str, grid: np.ndarray) -> None:
    """Execute Snipe-A (walk-in, blink-out) for an active snipe.

    Each frame reads unit state and decides:
      not in range        → PathGroupToTarget (pure move, weapons recharge)
      in range, shooting  → AMoveGroup (volley)
      all weapons cooling → blink retreat (shots are out)
      target dead/lost    → blink retreat (job done or abort)
      stalkers all dead   → cleanup only (nobody to blink)
    """
    if squad_id not in bot._snipe_state:
        return

    info = bot._snipe_state[squad_id]
    target_tag = info["target_tag"]
    stalker_tags: set[int] = info["stalker_tags"]
    retreat = info["retreat_point"]
    game_loop = bot.state.game_loop

    # Resolve live stalkers
    stalkers = [u for u in bot.units if u.tag in stalker_tags]
    if not stalkers:
        _cleanup_snipe(bot, squad_id)
        return

    tags = {s.tag for s in stalkers}

    # Resolve target — if dead/lost, blink retreat immediately
    target: Optional[Unit] = None
    for u in bot.all_enemy_units:
        if u.tag == target_tag:
            target = u
            break

    if target is None or not target.is_visible:
        maneuver = CombatManeuver()
        maneuver.add(GroupUseAbility(
            ability=AbilityId.EFFECT_BLINK_STALKER,
            group=stalkers,
            group_tags=tags,
            target=retreat,
            sync_command=True,
        ))
        bot.register_behavior(maneuver)
        _cleanup_snipe(bot, squad_id)
        return

    in_range = [s for s in stalkers if cy_in_attack_range(s, [target], bonus_distance=0.5)]
    # "Enough" = at least the raw number needed to one-shot the target
    enough_in_range = len(in_range) >= info["min_to_kill"]

    maneuver = CombatManeuver()

    # --- Approach: pure move (no attacking) so weapons recharge en route ---
    if not enough_in_range:
        stalker_center = Point2(cy_center(stalkers))
        maneuver.add(PathGroupToTarget(
            start=stalker_center,
            group=stalkers,
            group_tags=tags,
            grid=grid,
            target=target.position,
            success_at_distance=2.0,
        ))
        bot.register_behavior(maneuver)
        return

    # All weapons on cooldown = all shots have left the barrel → blink
    # Only check in-range stalkers; stragglers blink regardless
    if all(not s.weapon_ready for s in in_range):
        maneuver.add(GroupUseAbility(
            ability=AbilityId.EFFECT_BLINK_STALKER,
            group=stalkers,
            group_tags=tags,
            target=retreat,
            sync_command=True,
        ))
        bot.register_behavior(maneuver)
        _cleanup_snipe(bot, squad_id)
        return

    # Weapons still ready / firing — keep attacking
    maneuver.add(AMoveGroup(group=in_range, group_tags={s.tag for s in in_range}, target=target))
    bot.register_behavior(maneuver)


def execute_snipe_b(bot, squad_id: str, grid: np.ndarray) -> None:
    """Execute Snipe-B (blink-in, walk-out) for an active snipe.

    Each frame reads unit state and decides:
      not blinked in yet  → GroupUseAbility blink to landing spot
      not in range yet    → AMoveGroup toward target (closing after blink)
      in range, shooting  → AMoveGroup (volley)
      all weapons cooling → PathGroupToTarget walk out (shots are out)
      target dead/lost    → PathGroupToTarget walk out (no blink to escape)
      stalkers all dead   → cleanup only
    """
    if squad_id not in bot._snipe_state:
        return

    info = bot._snipe_state[squad_id]
    target_tag = info["target_tag"]
    stalker_tags: set[int] = info["stalker_tags"]
    retreat = info["retreat_point"]
    landing = info["landing"]
    game_loop = bot.state.game_loop

    # Resolve live stalkers
    stalkers = [u for u in bot.units if u.tag in stalker_tags]
    if not stalkers:
        _cleanup_snipe(bot, squad_id)
        return

    tags = {s.tag for s in stalkers}

    # Resolve target
    target: Optional[Unit] = None
    for u in bot.all_enemy_units:
        if u.tag == target_tag:
            target = u
            break

    # --- Blink approach (first frame only) ---
    if not info.get("_blinked_in"):
        info["_blinked_in"] = True
        maneuver = CombatManeuver()
        maneuver.add(GroupUseAbility(
            ability=AbilityId.EFFECT_BLINK_STALKER,
            group=stalkers,
            group_tags=tags,
            target=landing,
            sync_command=True,
        ))
        bot.register_behavior(maneuver)
        return

    # --- Target dead/lost → walk out immediately (no blink available) ---
    if target is None or not target.is_visible:
        maneuver = CombatManeuver()
        stalker_center = Point2(cy_center(stalkers))
        maneuver.add(PathGroupToTarget(
            start=stalker_center,
            group=stalkers,
            group_tags=tags,
            grid=grid,
            target=retreat,
            success_at_distance=3.0,
        ))
        bot.register_behavior(maneuver)
        _cleanup_snipe(bot, squad_id)
        return

    in_range = [s for s in stalkers if cy_in_attack_range(s, [target], bonus_distance=0.5)]
    enough_in_range = len(in_range) >= info["min_to_kill"]

    maneuver = CombatManeuver()

    # --- Not in range yet: close the gap after blink (shouldn't take long) ---
    if not enough_in_range:
        maneuver.add(AMoveGroup(group=stalkers, group_tags=tags, target=target))
        bot.register_behavior(maneuver)
        return

    # --- All weapons on cooldown → walk out ---
    if all(not s.weapon_ready for s in in_range):
        stalker_center = Point2(cy_center(stalkers))
        maneuver.add(PathGroupToTarget(
            start=stalker_center,
            group=stalkers,
            group_tags=tags,
            grid=grid,
            target=retreat,
            success_at_distance=3.0,
        ))
        bot.register_behavior(maneuver)
        _cleanup_snipe(bot, squad_id)
        return

    # Weapons still ready / firing — keep attacking
    maneuver.add(AMoveGroup(group=in_range, group_tags={s.tag for s in in_range}, target=target))
    bot.register_behavior(maneuver)


def _cleanup_snipe(bot, squad_id: str) -> None:
    """Remove all snipe state for a squad, releasing stalkers back to normal micro."""
    if squad_id not in bot._snipe_state:
        return
    state_info = bot._snipe_state[squad_id]
    for tag in state_info["stalker_tags"]:
        bot._snipe_committed.pop(tag, None)
    del bot._snipe_state[squad_id]


# ===== FOCUS-FIRE FUNCTIONS =====

def stalkers_needed_to_focus(
    stalker: Unit,
    target: Unit,
) -> int:
    """Calculate the number of stalkers needed to focus-fire a target.

    Scales naturally with target HP: enough stalkers that each volley deals
    roughly half the target's HP. Low-HP targets need fewer stalkers,
    high-HP targets need more. No hard cap — the combat sim gate decides
    whether the commitment is viable.

    Returns 0 if stalker can't attack the target.
    Returns at least FOCUS_MIN_STALKERS.

    Perf: ~2µs (one calculate_damage_vs_target call + arithmetic).
    """
    dmg = damage_per_volley(stalker, target)
    if dmg <= 0:
        return 0

    target_hp = target.health + target.shield
    # Each volley should deal ~50% of target HP — scales naturally with durability
    raw_needed = ceil(target_hp / dmg / 2)

    # Mobile targets might dodge — add small buffer
    buffer = SNIPE_OVERKILL_BUFFER_MOBILE if target.movement_speed > 0 else SNIPE_OVERKILL_BUFFER_STATIC
    needed = raw_needed + buffer

    if needed < FOCUS_MIN_STALKERS:
        needed = FOCUS_MIN_STALKERS

    return needed


def find_focus_candidates(
    stalkers: list[Unit],
    target: Unit,
    needed: int,
    max_range: float = 20.0,
) -> list[Unit]:
    """Select stalkers for a focus-fire, from those eligible.

    Filters:
    - Blink ready (EFFECT_BLINK_STALKER in abilities) — required for gap-close
    - Within max_range of the target

    Sorts by: distance to target (closest first) to minimize travel time.
    Returns up to `needed` stalkers. Returns empty list if not enough qualify.

    Perf: O(n) filter + O(n log n) sort where n = stalkers in range. Typically n < 20.
    """
    nearby: list[Unit] = cy_closer_than(stalkers, max_range, target.position)

    eligible: list[Unit] = [
        s for s in nearby
        if AbilityId.EFFECT_BLINK_STALKER in s.abilities
    ]

    if len(eligible) < needed:
        return []

    sorted_by_dist: list[Unit] = cy_sorted_by_distance_to(eligible, target.position)
    return sorted_by_dist[:needed]


def is_focus_eligible(
    bot,
    stalker_sample: Unit,
    target: Unit,
) -> bool:
    """Check if focus-fire is eligible for this target.

    Focus-fire requires the target to be isolated (low tactical grid value
    at target position). The value system (TACTICAL_BONUS + army_value) already
    filters out low-value targets. The approach method (blink vs walk) is
    determined by should_blink_in() based on range and speed.

    Returns True if the target is isolated enough for a focus-fire commitment.

    Perf: ~1µs (grid lookup).
    """
    # Isolation check: grid value at target position must be low enough
    grid: np.ndarray = bot.mediator.get_ground_grid
    target_pos = target.position.rounded
    if not cy_point_below_value(grid, target_pos, CHASE_TACTICAL_MAX):
        return False

    return True


def should_blink_in(stalker_sample: Unit, target: Unit) -> bool:
    """Determine whether focus-fire should blink in or walk in.

    Blinks in if the target outranges us OR is faster than us.
    Walks in if we outrange AND are faster (or equal speed).

    Perf: ~300ns (two cy_range_vs_target calls + one movement_speed comparison).
    """
    our_range = cy_range_vs_target(stalker_sample, target)
    their_range = cy_range_vs_target(target, stalker_sample)
    target_outranges = their_range > our_range + SNIPE_APPROACH_RANGE_BUFFER
    target_is_faster = target.movement_speed > stalker_sample.movement_speed
    return target_outranges or target_is_faster


# ===== FOCUS-FIRE STATE MACHINE =====

def execute_focus(
    bot,
    squad_id: str,
    grid: np.ndarray,
    nearby_enemies: Union[Units, list[Unit]],
) -> None:
    """Execute focus-fire (multi-volley pursuit) for an active focus operation.

    State machine:
      APPROACH (first frame, blink-in): GroupUseAbility blink toward target
      APPROACH (first frame, walk-in): PathGroupToTarget toward target
      ATTACK (subsequent frames):
        Target gone/invisible → cleanup
        Sim LOSS_DECISIVE_OR_WORSE → cleanup (tide turned)
        Target far + blink ok → GroupUseAbility blink toward target (gap-close)
        Else → StutterGroupForward toward target (keep attacking)

    nearby_enemies is the same all_close used by the combat loop — scoped
    to UNIT_ENEMY_DETECTION_RANGE. Used for combat sim abort check.
    """
    if squad_id not in bot._focus_state:
        return

    info = bot._focus_state[squad_id]
    target_tag = info["target_tag"]
    stalker_tags: set[int] = info["stalker_tags"]
    retreat = info["retreat_point"]
    blink_in = info.get("blink_in", False)

    # Resolve live stalkers
    stalkers = [u for u in bot.units if u.tag in stalker_tags]
    if not stalkers:
        _cleanup_focus(bot, squad_id)
        return

    tags = {s.tag for s in stalkers}

    # Resolve target from nearby enemies only (not global)
    target: Optional[Unit] = None
    for u in nearby_enemies:
        if u.tag == target_tag:
            target = u
            break

    # --- Target gone/invisible → cleanup ---
    if target is None or not target.is_visible:
        _cleanup_focus(bot, squad_id)
        return

    # --- Combat sim abort: tide turned against us ---
    enemy_list = list(nearby_enemies) if nearby_enemies else []
    if enemy_list:
        sim_result = bot.mediator.can_win_fight(
            own_units=Units(stalkers, bot),
            enemy_units=Units(enemy_list, bot),
            workers_do_no_damage=True,
        )
        if sim_result in FOCUS_ABORT_SIM:
            _cleanup_focus(bot, squad_id)
            return

    # --- Blink approach (first frame only, if target outranges or is faster) ---
    if not info.get("_approached"):
        info["_approached"] = True
        maneuver = CombatManeuver()

        if blink_in:
            # Blink to close the gap (like Snipe-B)
            landing = info.get("landing")
            if landing is not None:
                maneuver.add(GroupUseAbility(
                    ability=AbilityId.EFFECT_BLINK_STALKER,
                    group=stalkers,
                    group_tags=tags,
                    target=landing,
                    sync_command=True,
                ))
            else:
                # Fallback: blink toward target position
                maneuver.add(GroupUseAbility(
                    ability=AbilityId.EFFECT_BLINK_STALKER,
                    group=stalkers,
                    group_tags=tags,
                    target=target.position,
                    sync_command=True,
                ))
        else:
            # Walk in (like Snipe-A) — target is in range or slower
            stalker_center = Point2(cy_center(stalkers))
            maneuver.add(PathGroupToTarget(
                start=stalker_center,
                group=stalkers,
                group_tags=tags,
                grid=grid,
                target=target.position,
                success_at_distance=2.0,
            ))

        bot.register_behavior(maneuver)
        return

    # --- Ongoing attack: StutterGroupForward to stay on target ---
    stalker_center = Point2(cy_center(stalkers))
    dist_to_target = cy_distance_to(stalker_center, target.position)

    maneuver = CombatManeuver()

    # Blink gap-close: if target is pulling away beyond range + threshold
    # Land right on top of the target (same as initial blink-in) so we're
    # practically touching them — negates range advantage and splash.
    stalker_range = cy_range_vs_target(stalkers[0], target)
    if dist_to_target > stalker_range + CHASE_BLINK_GAP_THRESHOLD:
        has_blink = any(
            AbilityId.EFFECT_BLINK_STALKER in s.abilities for s in stalkers
        )
        if has_blink:
            blink_stalkers = [
                s for s in stalkers
                if AbilityId.EFFECT_BLINK_STALKER in s.abilities
            ]
            blink_tags = {s.tag for s in blink_stalkers}
            landing = compute_blink_landing(stalker_center, target.position, stalker_range)
            maneuver.add(GroupUseAbility(
                ability=AbilityId.EFFECT_BLINK_STALKER,
                group=blink_stalkers,
                group_tags=blink_tags,
                target=landing,
                sync_command=True,
            ))
            bot.register_behavior(maneuver)
            return

    # Normal pursuit: stutter forward toward target
    maneuver.add(StutterGroupForward(
        group=stalkers,
        group_tags=tags,
        group_position=stalker_center,
        target=target.position,
        enemies=Units(nearby_enemies, bot) if nearby_enemies else Units([], bot),
    ))
    bot.register_behavior(maneuver)


def _cleanup_focus(bot, squad_id: str) -> None:
    """Remove all focus-fire state for a squad, releasing stalkers back to normal micro."""
    if squad_id not in bot._focus_state:
        return
    state_info = bot._focus_state[squad_id]
    for tag in state_info["stalker_tags"]:
        bot._focus_committed.pop(tag, None)
    del bot._focus_state[squad_id]
