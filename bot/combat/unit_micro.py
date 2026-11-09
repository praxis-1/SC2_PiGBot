"""Unit-level micro management functions.

Purpose: Reusable micro control logic for individual units in combat
Key Decisions: Pure functions that create and return CombatManeuver instances;
    Stalker blink uses concave-aware positioning when available
Limitations: Assumes individual behavior pattern (one CombatManeuver per unit)
"""

from typing import List, Optional
import numpy as np

from sc2.unit import Unit
from sc2.units import Units
from sc2.position import Point2
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.data import Race

from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    KeepUnitSafe, StutterUnitBack, AMove, ShootTargetInRange, PathUnitToTarget,
    UseAOEAbility, UseAbility,
)
from cython_extensions import cy_closest_to, cy_in_attack_range, cy_distance_to, cy_find_units_center_mass, cy_towards, cy_is_facing
from bot.combat.scoring_table import select_target
from bot.constants import (
    DISRUPTOR_SQUAD_FOLLOW_DISTANCE, DISRUPTOR_SQUAD_TARGET_DISTANCE,
    HT_SQUAD_FOLLOW_DISTANCE, HT_SQUAD_TARGET_DISTANCE,
    HT_STORM_ENERGY_COST, HT_STORM_MIN_TARGETS,
    HT_FEEDBACK_ENERGY_COST, HT_FEEDBACK_RANGE, HT_FEEDBACK_MIN_ENEMY_ENERGY,
    FEEDBACK_TARGET_TYPES, HT_MERGE_ENERGY_THRESHOLD,
    HT_MERGE_COUNT_THRESHOLD, HT_MERGE_COUNT_THRESHOLD_PVP,
    STALKER_BLINK_HEALTH_THRESHOLD, STALKER_BLINK_RANGE,
    STALKER_LOCKON_BREAK_DISTANCE, STALKER_WIDOWMINE_DODGE_RADIUS,
    STALKER_FUNGAL_DODGE_RADIUS, FUNGAL_GROWTH_ENERGY_COST,
    FUNGAL_GROWTH_IMPACT_RADIUS,
    MELEE_RANGE_THRESHOLD, MELEE_THREAT_BUFFER, ACTIVE_ENGAGE_ANGLE,
)

# Per-mine tracking: tag → True if we already dodged this mine's shot.
# Prevents re-dodging the same mine shot across multiple frames.
_mine_dodged: dict = {}

# Per-frame cache: maps mine tag → targeted stalker tag.
# Computed once per frame so each stalker doesn't re-scan all mines.
_mine_target_cache: dict = {}  # {mine_tag: stalker_tag}
_mine_target_frame: int = -1   # game_loop when cache was built

# Per-infestor energy tracking: tag → (energy, frame) for detecting casts.
# Used to detect Fungal Growth casts via energy drop.
_infestor_energy_cache: dict = {}  # {infestor_tag: (energy, frame)}

# Per-frame cache: maps infestor tag → list of stalker tags in impact zone.
# Computed once per frame so each stalker doesn't re-scan all infestors.
_fungal_target_cache: dict = {}  # {infestor_tag: [stalker_tag, ...]}
_fungal_target_frame: int = -1   # game_loop when cache was built

# Per-infestor tracking: tag → True if we already dodged this fungal cast.
# Prevents re-dodging the same cast across multiple frames.
_fungal_dodged: dict = {}


def _build_mine_target_cache(bot) -> None:
    """Build a cache of which stalker each fired mine is targeting.

    Called once per frame. For each WIDOWMINEBURROWED with cba=True
    and facing!=0, finds the closest stalker within the facing cone
    (cy_is_facing with angle_error=0.5). This avoids each stalker
    independently iterating all mines.

    Also prunes stale entries from _mine_dodged to prevent unbounded
    growth across long games.
    """
    global _mine_target_cache, _mine_target_frame
    current_frame = bot.state.game_loop
    if current_frame == _mine_target_frame:
        return  # Already built this frame
    _mine_target_frame = current_frame
    _mine_target_cache = {}

    # Prune _mine_dodged: remove entries for mines no longer on the field
    alive_mine_tags = {u.tag for u in bot.enemy_units}
    stale = [k for k in _mine_dodged if k not in alive_mine_tags]
    for k in stale:
        del _mine_dodged[k]

    stalkers = bot.units(UnitTypeId.STALKER)
    if not stalkers:
        return

    for mine in bot.enemy_units:
        if mine.type_id != UnitTypeId.WIDOWMINEBURROWED:
            continue
        if not mine.can_be_attacked or mine.facing == 0.0:
            continue
        if _mine_dodged.get(mine.tag, False):
            continue
        # Find the stalker this mine is targeting (closest in facing cone)
        best_stalker = None
        best_dist = float('inf')
        for s in stalkers:
            if cy_distance_to(s.position, mine.position) > STALKER_WIDOWMINE_DODGE_RADIUS:
                continue
            if cy_is_facing(mine, s, angle_error=0.5):
                d = cy_distance_to(s.position, mine.position)
                if d < best_dist:
                    best_dist = d
                    best_stalker = s
        if best_stalker is not None:
            _mine_target_cache[mine.tag] = best_stalker.tag


def _build_fungal_target_cache(bot) -> None:
    """Build a cache of which stalkers are in the facing cone of casting infestors.

    Called once per frame. For each INFESTOR with energy drop >= 75 (Fungal cast),
    finds all stalkers within cast range (12 tiles) that are in the infestor's
    facing cone (cy_is_facing with angle_error=0.5). This identifies which
    stalkers the infestor targeted, since Fungal Growth is cast in the direction
    the infestor faces.

    Also prunes stale entries from _fungal_dodged to prevent unbounded growth.
    """
    global _fungal_target_cache, _fungal_target_frame, _infestor_energy_cache
    current_frame = bot.state.game_loop
    if current_frame == _fungal_target_frame:
        return  # Already built this frame
    _fungal_target_frame = current_frame
    _fungal_target_cache = {}

    # Prune _fungal_dodged: remove entries for infestors no longer on the field
    alive_infestor_tags = {u.tag for u in bot.enemy_units}
    stale = [k for k in _fungal_dodged if k not in alive_infestor_tags]
    for k in stale:
        del _fungal_dodged[k]

    # Prune old energy cache entries (older than 5 seconds)
    stale_energy = [tag for tag, (energy, frame) in _infestor_energy_cache.items()
                    if current_frame - frame > 112]  # ~5 seconds at 22.4 fps
    for tag in stale_energy:
        del _infestor_energy_cache[tag]

    stalkers = bot.units(UnitTypeId.STALKER)
    if not stalkers:
        return

    for infestor in bot.enemy_units:
        if infestor.type_id != UnitTypeId.INFESTOR:
            continue

        # Get previous energy to detect casts
        prev_energy, prev_frame = _infestor_energy_cache.get(infestor.tag, (None, -1))
        _infestor_energy_cache[infestor.tag] = (infestor.energy, current_frame)

        if prev_energy is None:
            continue  # First time seeing this infestor

        # Check for energy drop indicating Fungal Growth cast (75 energy)
        energy_drop = prev_energy - infestor.energy
        if energy_drop < FUNGAL_GROWTH_ENERGY_COST - 5:  # Allow 5 energy tolerance
            continue  # Not a fungal cast

        if _fungal_dodged.get(infestor.tag, False):
            continue  # Already dodged this cast

        # Find all stalkers in the infestor's facing cone within cast range.
        # Fungal Growth has 9 range + 2 radius, so stalkers up to ~11 tiles
        # away in the facing direction could be hit. Use cy_is_facing to
        # identify which stalkers the infestor targeted.
        targeted_stalkers = []
        for s in stalkers:
            dist = cy_distance_to(s.position, infestor.position)
            if dist > STALKER_FUNGAL_DODGE_RADIUS:
                continue
            # Check if infestor is facing this stalker (angle_error=0.5 ~28 degrees)
            if cy_is_facing(infestor, s, angle_error=0.5):
                targeted_stalkers.append(s.tag)

        if targeted_stalkers:
            _fungal_target_cache[infestor.tag] = targeted_stalkers


def micro_stalker(
    stalker: Unit,
    enemies: List[Unit],
    grid: np.ndarray,
    avoid_grid: np.ndarray,
    aggressive: bool = True,
    squad_center: Optional[Point2] = None,
    enemy_center: Optional[Point2] = None,
    ranged_units: Optional[List[Unit]] = None,
    all_close: Optional[Units] = None,
    bot=None,
) -> CombatManeuver:
    """Stalker micro with threat-dodge and blink-when-low behavior.

    Priority order:
    1. Dodge incoming threats (Cyclone lock-on, Widow Mine, Fungal Growth)
       — blink away from the threat source/impact point.
    2. Low-health blink along the concave line to a safer position.
    3. Standard ranged micro (stutter-step via micro_ranged_unit).

    Threat detection methods:
    - Cyclone Lock-on: check our stalker's buffs for BuffId.LOCKON
    - Widow Mine: track can_be_attacked transitions on WIDOWMINEBURROWED
      (False→True = just fired) + facing angle to identify targeted stalker
    - Fungal Growth: detect via INFESTOR energy drop (75 energy) + facing
      angle to identify stalkers in the 2-tile impact radius

    Args:
        stalker: The Stalker unit to control
        enemies: All enemy units in range (pre-filtered to attackable/reachable)
        grid: Ground grid for pathfinding
        avoid_grid: Avoidance grid for safety checks
        aggressive: Whether to fight aggressively or defensively
        squad_center: Center of the squad (for concave direction)
        enemy_center: Center of nearby enemies (for concave direction)
        ranged_units: Other ranged units in the squad (for concave edge calculation)
        all_close: All nearby enemy units (for finding Cyclone position)
        bot: Bot instance (for finding projectile units)

    Returns:
        CombatManeuver with appropriate behaviors added
    """
    blink_ready = AbilityId.EFFECT_BLINK_STALKER in stalker.abilities

    # --- Priority 1: Threat-dodge blink ---
    if blink_ready and all_close is not None and bot is not None:
        dodge_target = _check_threat_dodge(stalker, all_close, bot)
        if dodge_target is not None:
            maneuver = CombatManeuver()
            maneuver.add(UseAbility(
                ability=AbilityId.EFFECT_BLINK_STALKER,
                unit=stalker,
                target=dodge_target,
            ))
            maneuver.add(KeepUnitSafe(stalker, avoid_grid))
            return maneuver

    # --- Priority 2: Low-health concave blink ---
    health_ratio = stalker.shield_health_percentage
    if blink_ready and health_ratio < STALKER_BLINK_HEALTH_THRESHOLD and enemies:
        blink_target = _compute_blink_target(
            stalker, enemies, squad_center, enemy_center, ranged_units
        )

        if blink_target is not None:
            maneuver = CombatManeuver()
            maneuver.add(UseAbility(
                ability=AbilityId.EFFECT_BLINK_STALKER,
                unit=stalker,
                target=blink_target,
            ))
            maneuver.add(KeepUnitSafe(stalker, avoid_grid))
            return maneuver

    # --- Priority 3: Standard ranged micro ---
    return micro_ranged_unit(
        unit=stalker,
        enemies=enemies,
        grid=grid,
        avoid_grid=avoid_grid,
        aggressive=aggressive,
    )


def _check_threat_dodge(
    stalker: Unit,
    all_close: Units,
    bot,
) -> Optional[Point2]:
    """Check if the stalker should blink to dodge an incoming threat.

    Detection methods (enemy orders/engaged_target_tag/weapon_cooldown
    are NOT populated for enemies; WIDOWMINEWEAPON/FUNGALGROWTHMISSILE
    projectile units are NOT visible to opponents):

    1. Cyclone Lock-on: our stalker gets BuffId.LOCKON while locked.
       Blink away from the Cyclone to break the 15-tile tether.
    2. Widow Mine: track can_be_attacked transitions on WIDOWMINEBURROWED.
       When a mine flips from can_be_attacked=False → True, it just fired.
       The mine's .facing angle points at its target — blink stalkers
       in that direction away from the mine.
    3. Fungal Growth: detect via INFESTOR energy drop (75 energy).
       When an infestor's energy drops by ~75, it just cast Fungal.
       Blink stalkers in the impact zone (2 tile radius) away from
       the infestor's position.

    Args:
        stalker: The Stalker that might need to dodge
        all_close: All nearby enemy units (for finding Cyclone position)
        bot: Bot instance (for finding enemy mines/infestors)

    Returns:
        Point2 blink target if a threat is detected, None otherwise
    """
    pos = stalker.position

    # --- Cyclone Lock-on ---
    # Our stalker gets the LOCKON buff when a Cyclone locks onto it.
    # Blink away from the Cyclone to break the tether (must exceed 15 tiles).
    if BuffId.LOCKON in stalker.buffs:
        cyclones = [e for e in all_close if e.type_id == UnitTypeId.CYCLONE]
        if cyclones:
            closest_cyclone = cy_closest_to(pos, cyclones)
            away = Point2(cy_towards(closest_cyclone.position, pos, STALKER_BLINK_RANGE))
            return away

    # --- Widow Mine (can_be_attacked + facing) ---
    # A WIDOWMINEBURROWED with can_be_attacked=True and facing != 0.0
    # has just fired — the facing angle points at the target.
    # We pre-compute which stalker each mine targets (once per frame),
    # then each stalker just checks "am I the target?"
    _build_mine_target_cache(bot)
    for mine_tag, stalker_tag in _mine_target_cache.items():
        if stalker_tag != stalker.tag:
            continue  # This mine isn't targeting us
        # We're the target — blink away from the mine
        _mine_dodged[mine_tag] = True
        # Find the mine position for blink direction
        mine = None
        for e in bot.enemy_units:
            if e.tag == mine_tag:
                mine = e
                break
        if mine is None:
            continue
        dist = cy_distance_to(pos, mine.position)
        away = Point2(cy_towards(mine.position, pos, STALKER_BLINK_RANGE))
        return away

    # --- Fungal Growth (energy drop detection) ---
    # When an INFESTOR's energy drops by ~75, it just cast Fungal Growth.
    # The fungal targets wherever the infestor is facing (9 range, 2 radius).
    # We pre-compute which stalkers are in the facing cone (once per frame),
    # then each stalker checks "am I in the impact zone?"
    _build_fungal_target_cache(bot)
    for infestor_tag, stalker_tags in _fungal_target_cache.items():
        if stalker.tag not in stalker_tags:
            continue  # This fungal isn't targeting us
        # We're in the impact zone — blink away from the infestor
        _fungal_dodged[infestor_tag] = True
        # Find the infestor position for blink direction
        infestor = None
        for e in bot.enemy_units:
            if e.tag == infestor_tag:
                infestor = e
                break
        if infestor is None:
            continue
        away = Point2(cy_towards(infestor.position, pos, STALKER_BLINK_RANGE))
        return away

    return None


def _compute_blink_target(
    stalker: Unit,
    enemies: List[Unit],
    squad_center: Optional[Point2],
    enemy_center: Optional[Point2],
    ranged_units: Optional[List[Unit]],
) -> Optional[Point2]:
    """Compute where a low-health Stalker should blink to.

    Strategy: blink along the concave line (perpendicular to approach
    direction) toward the far edge of the formation. This keeps the
    Stalker contributing DPS from a safer position at the concave edge
    rather than retreating entirely.

    Falls back to blinking away from the closest enemy when concave
    geometry isn't available.

    Args:
        stalker: The Stalker unit that will blink
        enemies: Nearby enemy units
        squad_center: Center of the squad (for approach direction)
        enemy_center: Center of nearby enemies (for approach direction)
        ranged_units: Other ranged units in the squad (for edge calculation)

    Returns:
        Point2 blink target, or None if no valid target found
    """
    pos = stalker.position

    # Determine approach direction (squad → enemy) for concave line
    if squad_center is not None and enemy_center is not None:
        length = cy_distance_to(squad_center, enemy_center)

        if length > 0.01:
            # Approach direction unit vector
            dx = enemy_center.x - squad_center.x
            dy = enemy_center.y - squad_center.y
            approach_x = dx / length
            approach_y = dy / length

            # Perpendicular axis (left/right relative to approach)
            perp_x = -approach_y
            perp_y = approach_x

            # Find which side of the concave the stalker is on,
            # then blink toward the far edge of the formation
            if ranged_units and len(ranged_units) >= 2:
                # Compute lateral offsets for all ranged units
                offsets = []
                for u in ranged_units:
                    rel_x = u.position.x - squad_center.x
                    rel_y = u.position.y - squad_center.y
                    lateral = rel_x * perp_x + rel_y * perp_y
                    offsets.append(lateral)

                # The stalker's own lateral offset
                stalker_lateral = (pos.x - squad_center.x) * perp_x + (pos.y - squad_center.y) * perp_y
                # Stalker's depth along the approach axis (positive = toward enemy)
                stalker_depth = (pos.x - squad_center.x) * approach_x + (pos.y - squad_center.y) * approach_y

                # Blink toward the far edge of the formation from the stalker's position
                # If stalker is on the left side, blink further left; if right, further right
                min_offset = min(offsets)
                max_offset = max(offsets)

                if stalker_lateral >= 0:
                    # Stalker is on the right side — blink toward right edge
                    target_lateral = max_offset
                else:
                    # Stalker is on the left side — blink toward left edge
                    target_lateral = min_offset

                # Target point: on the concave edge, at the stalker's depth,
                # shifted slightly back (away from enemy) for safety
                retreat_tiles = 2.0
                target_x = squad_center.x + perp_x * target_lateral + approach_x * (stalker_depth - retreat_tiles)
                target_y = squad_center.y + perp_y * target_lateral + approach_y * (stalker_depth - retreat_tiles)
                blink_target = Point2((target_x, target_y))

                # Clamp to blink range from current position
                dist = cy_distance_to(pos, blink_target)
                if dist > STALKER_BLINK_RANGE:
                    blink_target = Point2(cy_towards(pos, blink_target, STALKER_BLINK_RANGE))

                return blink_target

    # Fallback: blink directly away from closest enemy
    closest_enemy = cy_closest_to(pos, enemies)
    if closest_enemy is not None:
        # Point away from enemy at blink range
        away_point = Point2((
            2 * pos.x - closest_enemy.position.x,
            2 * pos.y - closest_enemy.position.y,
        ))
        return Point2(cy_towards(pos, away_point, STALKER_BLINK_RANGE))

    return None


def _has_melee_threat(unit: Unit, enemies: List[Unit]) -> bool:
    """Check if any melee enemy is facing toward and close to this unit.

    Uses cy_is_facing to confirm the melee unit is actively charging
    rather than retreating past or moving elsewhere.
    """
    for enemy in enemies:
        if enemy.ground_range > MELEE_RANGE_THRESHOLD:
            continue
        threat_range = enemy.ground_range + enemy.radius + unit.radius + MELEE_THREAT_BUFFER
        if cy_distance_to(unit.position, enemy.position) <= threat_range:
            if cy_is_facing(enemy, unit, angle_error=ACTIVE_ENGAGE_ANGLE):
                return True
    return False


def _is_targeted_by_ranged(unit: Unit, enemies: List[Unit]) -> bool:
    """Check if any ranged enemy is actively targeting this unit.

    Uses engaged_target_tag from the SC2 protocol to detect which
    unit an enemy has locked onto. Conservative: only returns True
    when we have positive confirmation (tag match), not on 0/unknown.
    """
    unit_tag = unit.tag
    for enemy in enemies:
        if enemy.ground_range <= MELEE_RANGE_THRESHOLD:
            continue
        if enemy.engaged_target_tag == unit_tag:
            return True
    return False


def micro_air_unit(
    unit: Unit,
    enemies: List[Unit],
    avoid_grid: np.ndarray,
    aggressive: bool = True,
) -> CombatManeuver:
    """Placeholder micro for air combat units. Shoot in range, AMove toward target.

    Purpose: Basic air unit behavior until dedicated air micro is implemented.
    Key Decisions: No stutter logic — air units don't kite like ground units.
    Limitations: No specialized behavior for Void Rays, Phoenixes, Carriers, etc.
    """
    maneuver = CombatManeuver()
    maneuver.add(KeepUnitSafe(unit, avoid_grid))

    if in_attack_range := cy_in_attack_range(unit, enemies):
        shoot_target = select_target(unit, in_attack_range)
        maneuver.add(ShootTargetInRange(unit=unit, targets=[shoot_target]))
    elif aggressive:
        best_target = select_target(unit, enemies)
        maneuver.add(AMove(unit=unit, target=best_target.position))

    return maneuver


def micro_ranged_unit(
    unit: Unit,
    enemies: List[Unit],
    grid: np.ndarray,
    avoid_grid: np.ndarray,
    aggressive: bool = True
) -> CombatManeuver:
    """Create micro behaviors for a ground ranged unit.

    Purpose: Smart stutter-step micro for ground ranged combat units (Stalkers, Immortals, etc.).
    Key Decisions: Stutter-back only when threatened (melee facing us or ranged targeting us),
        not blindly on every weapon cooldown. Air units use micro_air_unit instead.
    Limitations: Not suitable for air units (different kiting dynamics).

    Stutter logic:
    - Melee threat nearby (facing us + in range): always stutter back on cooldown
    - Ranged enemy targeting us (engaged_target_tag match): stutter back on cooldown
    - Neither threat: AMove toward target on cooldown (keep advancing, shoot when ready)

    Args:
        unit: The ground ranged unit to control
        enemies: All enemy units (pre-filtered to attackable/reachable)
        grid: Ground grid for pathfinding
        avoid_grid: Avoidance grid for safety checks
        aggressive: Whether to fight aggressively or defensively

    Returns:
        CombatManeuver with appropriate behaviors added
    """
    maneuver = CombatManeuver()
    closest_enemy = cy_closest_to(unit.position, enemies)
    best_target = select_target(unit, enemies)

    # ALWAYS add KeepUnitSafe FIRST with avoidance grid
    # This ensures units dodge dangerous abilities (disruptor shots, banelings, etc.)
    maneuver.add(KeepUnitSafe(unit, avoid_grid))

    # Determine if unit should stutter back (threatened) or hold/advance (safe)
    is_threatened = _has_melee_threat(unit, enemies) or _is_targeted_by_ranged(unit, enemies)

    if not aggressive:
        # Defensive mode: always kite back (retreat from fight).
        # Smart stutter is for aggressive mode only — when retreating,
        # the priority is leaving the battle, not conditional kiting.
        maneuver.add(StutterUnitBack(unit, target=closest_enemy, grid=grid))
        # Still shoot if weapon ready and in range while retreating
        if unit.weapon_ready and (in_attack_range := cy_in_attack_range(unit, enemies)):
            shoot_target = select_target(unit, in_attack_range)
            maneuver.add(ShootTargetInRange(unit=unit, targets=[shoot_target]))
        return maneuver

    # Aggressive mode: smart stutter based on threat type
    if not unit.weapon_ready:
        if is_threatened:
            # Confirmed threat: kite back from closest enemy
            maneuver.add(StutterUnitBack(unit, target=closest_enemy, grid=grid))
        else:
            # No confirmed threat but enemies nearby — stutter back conservatively.
            # engaged_target_tag is unreliable (often 0), and cy_is_facing can miss
            # enemies mid-attack. Only AMove forward if truly no enemies in range.
            if cy_in_attack_range(unit, enemies):
                maneuver.add(StutterUnitBack(unit, target=closest_enemy, grid=grid))
            else:
                maneuver.add(AMove(unit=unit, target=best_target.position))
    else:
        # Weapon ready: shoot if in range, else advance
        if in_attack_range := cy_in_attack_range(unit, enemies):
            shoot_target = select_target(unit, in_attack_range)
            maneuver.add(ShootTargetInRange(unit=unit, targets=[shoot_target]))
        else:
            maneuver.add(AMove(unit=unit, target=best_target.position))

    return maneuver


def micro_melee_unit(
    unit: Unit,
    enemies: List[Unit],
    avoid_grid: np.ndarray,
    grid: np.ndarray,
    fallback_position: Optional[Point2] = None,
    aggressive: bool = True
) -> CombatManeuver:
    """
    Create micro behaviors for a melee unit.
    
    Uses weighted scoring (select_target) for target selection.
    Priority targeting is handled by score weights, not a separate list.
    
    Args:
        unit: The melee unit to control
        enemies: All enemy units in range (pre-filtered to attackable/reachable)
        avoid_grid: Avoidance grid for safety (dodge disruptor shots, etc.)
        grid: Ground grid for pathing (used in defensive retreat)
        fallback_position: Position to retreat to when defensive and no enemies in range
        aggressive: Whether to advance or hold position
        
    Returns:
        CombatManeuver with appropriate behaviors added
    """
    maneuver = CombatManeuver()
    
    # ALWAYS add KeepUnitSafe FIRST with avoidance grid
    # This ensures melee units dodge dangerous abilities (disruptor shots, banelings, etc.)
    maneuver.add(KeepUnitSafe(unit, avoid_grid))
    
    # Attack best scored target in range (both aggressive and defensive)
    if in_attack_range := cy_in_attack_range(unit, enemies):
        shoot_target = select_target(unit, in_attack_range)
        maneuver.add(ShootTargetInRange(unit=unit, targets=[shoot_target]))
    elif aggressive:
        # Aggressive mode: advance toward best scored target
        best_target = select_target(unit, enemies)
        maneuver.add(AMove(unit=unit, target=best_target.position))
    elif fallback_position:
        # Defensive mode: path toward fallback (base/squad target) using ground grid
        # to avoid running through enemy influence. AMove would charge back in.
        maneuver.add(PathUnitToTarget(unit=unit, grid=grid, target=fallback_position, success_at_distance=3.0))
    # Defensive mode with nothing in range and no fallback: KeepUnitSafe handles dodging
    
    return maneuver


def micro_disruptor(
    disruptor: Unit,
    enemies: List[Unit],
    friendly_units: List[Unit],
    avoid_grid: np.ndarray,
    grid: np.ndarray,
    bot,
    nova_manager,
    squad_position: Point2,
    ranged_center: Optional[Point2] = None,
) -> bool:
    """Handle disruptor nova firing and movement to stay with ground army.

    Disruptors are special: they can't attack, only fire novas.
    - Nova ready + enemies: let nova system take FULL control (no interference)
    - Nova on cooldown: stay safe in backlines, follow ranged line

    Uses two KeepUnitSafe layers like ranged/melee units:
      1. avoid_grid — dodge immediate danger (biles, disruptor shots, etc.)
      2. grid (ground influence) — kite away from enemy influence zones

    During combat, follows ranged_center (center of ranged units) so the
    disruptor mirrors the ranged line's movement. Falls back to squad_position
    when no ranged center is available.

    Args:
        disruptor: The Disruptor unit to control
        enemies: Enemy units (filtered for disruptor targeting)
        friendly_units: Friendly units (to avoid friendly fire)
        avoid_grid: Avoidance grid for immediate danger
        grid: Ground grid with enemy influence (for kiting away from enemy zones)
        bot: Bot instance (for use_disruptor_nova access)
        nova_manager: NovaManager instance for coordination
        squad_position: Center of the full squad (fallback when no ranged_center)
        ranged_center: Center of ranged units during combat (preferred follow target)

    Returns:
        True if nova fired or behavior registered, False otherwise
    """

    # Check if nova is ready AND enemies are present
    nova_ready = AbilityId.EFFECT_PURIFICATIONNOVA in disruptor.abilities

    if nova_ready and enemies:
        # Nova system takes FULL control - don't register any ARES maneuver
        # This avoids conflict between KeepUnitSafe and nova targeting
        try:
            result = bot.use_disruptor_nova.execute(
                disruptor, enemies, friendly_units, nova_manager
            )
            if result:  # Nova fired successfully
                return True
        except Exception as e:
            print(f"DEBUG ERROR in disruptor nova firing: {e}")

    # Nova on cooldown OR no enemies - disruptor should stay safe in backlines
    maneuver = CombatManeuver()

    # Layer 1: Dodge immediate danger (biles, disruptor shots, storms, etc.)
    maneuver.add(KeepUnitSafe(disruptor, avoid_grid))

    # Layer 2: Kite away from enemy influence zones (same pattern as ranged/melee units)
    maneuver.add(KeepUnitSafe(disruptor, grid))

    # Follow the ranged center during combat, squad center otherwise
    follow_target = ranged_center if ranged_center is not None else squad_position
    distance_to_follow = cy_distance_to(disruptor.position, follow_target)
    if distance_to_follow > DISRUPTOR_SQUAD_FOLLOW_DISTANCE:
        maneuver.add(PathUnitToTarget(
            unit=disruptor,
            grid=bot.mediator.get_ground_grid,
            target=follow_target,
            success_at_distance=DISRUPTOR_SQUAD_TARGET_DISTANCE,
        ))

    bot.register_behavior(maneuver)
    return True


def merge_high_templars(bot) -> None:
    """Merge HT pairs into Archons via two independent triggers (OR logic).

    Trigger 1 — Energy: Spent HTs (energy < threshold) merge regardless
    of total count. These HTs have done their job and should become Archons.

    Trigger 2 — Count: When we have more HTs than the count threshold,
    merge one closest pair (lowest energy first) to convert surplus into
    Archons. Only one pair per frame to avoid over-merging.

    Count threshold is per-matchup (lower in PvP where HTs are primarily
    Archon-in-waiting). The composition scorer picks the army mix; the count
    gate is what keeps the merge loop from running away here.

    Gas gate: If the active profile has archon_merge_gas_requirement > 0,
    merging is blocked until enough gas geysers are built. This prevents
    premature Archon creation when the economy can't sustain HT production.

    Pairs are selected by proximity: sort candidates by energy (lowest
    first), then find the closest neighbor. This produces Archons that
    arrive at the fight faster than arbitrary pairing.

    Uses request_archon_morph for a proper paired command so both HTs
    get a single merge order (no order conflicts).

    Call once per frame from the combat loop (not macro — HT lifecycle
    belongs with HT micro logic).
    """
    # Gas gate: block merging until we have enough gas infrastructure
    # to sustain HT production. Profiles with requirement=0 skip this check.
    from bot.constants import get_active_profile, _resolve
    profile = get_active_profile(bot)
    gas_requirement = _resolve(profile.archon_merge_gas_requirement, bot)
    if gas_requirement > 0:
        gas_geysers = bot.structures(UnitTypeId.ASSIMILATOR).amount + bot.structures(UnitTypeId.ASSIMILATORRICH).amount
        if gas_geysers < gas_requirement:
            return  # Not enough gas to sustain HT→Archon production yet

    all_hts = bot.units(UnitTypeId.HIGHTEMPLAR).ready
    if bot.enemy_race == Race.Protoss:
        # PvP: if enemy has Carriers, hold HTs for Storm instead of merging
        enemy_has_carriers = any(
            u.type_id == UnitTypeId.CARRIER
            for u in (bot.mediator.get_cached_enemy_army or [])
        )
        count_threshold = HT_MERGE_COUNT_THRESHOLD if enemy_has_carriers else HT_MERGE_COUNT_THRESHOLD_PVP
    else:
        count_threshold = HT_MERGE_COUNT_THRESHOLD

    # Track which HTs are already being merged to avoid double-merging
    merging_tags: set[int] = set()

    # --- Trigger 1: Energy (spent HTs merge regardless of count) ---
    low_energy_hts = [
        ht for ht in all_hts
        if ht.energy < HT_MERGE_ENERGY_THRESHOLD
        and not (ht.orders and ht.orders[0].ability.id == AbilityId.MORPH_ARCHON)
    ]
    low_energy_hts.sort(key=lambda ht: ht.energy)
    while len(low_energy_hts) >= 2:
        ht_a = low_energy_hts.pop(0)
        closest_idx, closest_dist = 0, float("inf")
        for i, ht_b in enumerate(low_energy_hts):
            dist = cy_distance_to(ht_a.position, ht_b.position)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i
        ht_b = low_energy_hts.pop(closest_idx)
        bot.request_archon_morph([ht_a, ht_b])
        merging_tags.add(ht_a.tag)
        merging_tags.add(ht_b.tag)

    # --- Trigger 2: Count (over-stocked → merge one closest pair) ---
    if len(all_hts) > count_threshold:
        # Only consider HTs not already merging from trigger 1
        surplus_hts = [
            ht for ht in all_hts
            if ht.tag not in merging_tags
            and not (ht.orders and ht.orders[0].ability.id == AbilityId.MORPH_ARCHON)
        ]
        if len(surplus_hts) >= 2:
            surplus_hts.sort(key=lambda ht: ht.energy)
            ht_a = surplus_hts[0]
            closest_idx, closest_dist = 1, float("inf")
            for i, ht_b in enumerate(surplus_hts[1:], start=1):
                dist = cy_distance_to(ht_a.position, ht_b.position)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx = i
            ht_b = surplus_hts[closest_idx]
            bot.request_archon_morph([ht_a, ht_b])


def micro_high_templar(
    ht: Unit,
    enemies: List[Unit],
    avoid_grid: np.ndarray,
    grid: np.ndarray,
    bot,
    squad_position: Point2,
    ranged_center: Optional[Point2] = None,
) -> bool:
    """Handle High Templar abilities and safe army-following.

    Priority: Psi Storm > Feedback > stay safe + follow army.
    HTs are fragile casters that should stay behind the army.
    Skips HTs that are currently merging to archon (MORPH_ARCHON order).

    Uses two KeepUnitSafe layers like ranged/melee units:
      1. avoid_grid — dodge immediate danger (biles, disruptor shots, etc.)
      2. grid (ground influence) — kite away from enemy influence zones

    During combat, follows ranged_center (center of ranged units) so the
    HT mirrors the ranged line's movement. Falls back to squad_position
    when no ranged center is available.

    Args:
        ht: The High Templar unit
        enemies: Nearby enemy units (all visible)
        avoid_grid: Avoidance grid for immediate danger
        grid: Ground grid with enemy influence (for kiting away from enemy zones)
        bot: Bot instance
        squad_position: Center of the full squad (fallback when no ranged_center)
        ranged_center: Center of ranged units during combat (preferred follow target)

    Returns:
        True if behavior registered
    """
    # Skip if HT is currently merging to archon
    if ht.orders and ht.orders[0].ability.id == AbilityId.MORPH_ARCHON:
        return False

    # Priority 1: Psi Storm — ARES UseAOEAbility finds optimal cast position,
    # avoids friendly fire and duplicate storms automatically
    if ht.energy >= HT_STORM_ENERGY_COST and enemies:
        storm_behavior = UseAOEAbility(
            unit=ht,
            ability_id=AbilityId.PSISTORM_PSISTORM,
            targets=enemies,
            min_targets=HT_STORM_MIN_TARGETS,
            avoid_own_ground=True,
            avoid_own_flying=True,
        )
        if storm_behavior.execute(bot, bot.config, bot.mediator):
            return True

    # Priority 2: Feedback — only on high-value caster types
    if ht.energy >= HT_FEEDBACK_ENERGY_COST and enemies:
        feedback_targets = [
            e for e in enemies
            if e.type_id in FEEDBACK_TARGET_TYPES
            and e.energy >= HT_FEEDBACK_MIN_ENEMY_ENERGY
            and cy_distance_to(ht.position, e.position) <= HT_FEEDBACK_RANGE
        ]
        if feedback_targets:
            best_target = max(feedback_targets, key=lambda e: e.energy)
            ht(AbilityId.FEEDBACK_FEEDBACK, best_target)
            return True

    # No cast opportunity — stay safe and follow army
    maneuver = CombatManeuver()

    # Layer 1: Dodge immediate danger (biles, disruptor shots, storms, etc.)
    maneuver.add(KeepUnitSafe(ht, avoid_grid))

    # Layer 2: Kite away from enemy influence zones (same pattern as ranged/melee units)
    maneuver.add(KeepUnitSafe(ht, grid))

    # Follow the ranged center during combat, squad center otherwise
    follow_target = ranged_center if ranged_center is not None else squad_position
    distance_to_follow = cy_distance_to(ht.position, follow_target)
    if distance_to_follow > HT_SQUAD_FOLLOW_DISTANCE:
        maneuver.add(PathUnitToTarget(
            unit=ht,
            grid=bot.mediator.get_ground_grid,
            target=follow_target,
            success_at_distance=HT_SQUAD_TARGET_DISTANCE,
        ))

    bot.register_behavior(maneuver)
    return True


def micro_sentry(
    sentry: Unit,
    friendly_units: List[Unit],
    enemies: List[Unit],
    avoid_grid: np.ndarray,
    grid: np.ndarray,
    bot,
    squad_position: Point2,
    ranged_center: Optional[Point2] = None,
    ff_assignments: Optional[dict[int, list[Point2]]] = None,
    gs_approved: bool = False,
    gs_target: Optional[Point2] = None,
) -> bool:
    """Handle Sentry abilities and safe army-following.

    Priority order:
      1. Force Field split (if this sentry was assigned FF positions)
      2. Guardian Shield (if squad-level assignment approved this sentry)
      3. Stay safe (two-layer: avoid grid + ground grid) and follow army

    Force Field split is computed once per squad (in combat.py) across all
    sentries pooling their energy, then each sentry executes its assigned
    casts. This ensures coordinated FF placement that a single sentry
    couldn't achieve alone.

    Guardian Shield assignment is computed once per squad (in combat.py) via
    _compute_guardian_shield_assignments(), which uses a Sentry-centric
    influence map: each Sentry casts where it is if there are uncovered
    friendlies nearby, rather than being sent to a distant pocket. A Sentry
    standing on an active shield (overlap) is assigned the nearest uncovered
    pocket so it can reposition and cast next frame — mirroring the Disruptor
    Nova exclusion-mask pattern without scattering Sentries across the army.

    Uses two KeepUnitSafe layers like ranged/melee units:
      1. avoid_grid — dodge immediate danger (biles, disruptor shots, etc.)
      2. grid (ground influence) — kite away from enemy influence zones

    During combat, non-casting Sentries follow ranged_center (center of
    ranged units) so they mirror the ranged line's movement instead of the
    full squad center (which includes melee units at the front). Falls back
    to squad_position when no ranged center is available.

    Args:
        sentry: The Sentry unit
        friendly_units: Nearby friendly units to potentially cover
        enemies: Nearby enemy units (already filtered for combat-relevant types)
        avoid_grid: Avoidance grid for immediate danger (biles, disruptor, etc.)
        grid: Ground grid with enemy influence (for kiting away from enemy zones)
        bot: Bot instance
        squad_position: Center of the full squad (fallback when no ranged_center)
        ranged_center: Center of ranged units during combat (preferred follow target
            for non-casting Sentries)
        ff_assignments: Dict mapping sentry tag → list of FF positions to cast.
            Pre-computed by compute_ff_split() in combat.py. None = no split.
        gs_approved: Whether this sentry was approved by the squad-level
            Guardian Shield assignment to cast. If False, skip casting.
        gs_target: The densest uncovered pocket this Sentry was assigned to
            cover (from the influence map). When set, the Sentry paths here
            instead of ranged_center. None = not approved or no peak assigned.

    Returns:
        True if behavior registered
    """
    from bot.constants import (
        SENTRY_SQUAD_FOLLOW_DISTANCE,
        SENTRY_SQUAD_TARGET_DISTANCE,
        GUARDIAN_SHIELD_RADIUS,
    )
    from sc2.ids.buff_id import BuffId

    # Priority 1: Force Field split — cast assigned FFs if available
    if ff_assignments and sentry.tag in ff_assignments:
        positions = ff_assignments[sentry.tag]
        if positions:
            # Cast all assigned FFs for this sentry
            # (energy was already validated in compute_ff_split)
            # First cast is immediate (queue=False), subsequent casts are queued
            # so SC2 executes them in sequence instead of overwriting each other
            for i, pos in enumerate(positions):
                sentry(AbilityId.FORCEFIELD_FORCEFIELD, pos, queue=(i > 0))
            return True

    # Priority 2: Guardian Shield
    # Sentry-centric: the assignment (combat.py) approves a Sentry to cast
    # only if there are uncovered friendlies within GS radius of the Sentry's
    # current position. The peak IS the Sentry's position, so the arrival
    # gate is trivially satisfied. For overlapping Sentries (not approved,
    # assigned a repositioning peak), gs_target is the nearest uncovered
    # pocket — they path there and cast next frame once they arrive.
    if gs_approved and not sentry.has_buff(BuffId.GUARDIANSHIELD):
        if gs_target is not None:
            # Cast only once we're within GS radius of the assigned peak.
            # Until then, fall through to the movement block to path there.
            dist_to_peak = cy_distance_to(sentry.position, gs_target)
            if dist_to_peak <= GUARDIAN_SHIELD_RADIUS:
                sentry(AbilityId.GUARDIANSHIELD_GUARDIANSHIELD)
                return True
        else:
            # No specific peak (e.g. single Sentry, already at squad center)
            sentry(AbilityId.GUARDIANSHIELD_GUARDIANSHIELD)
            return True

    # No cast opportunity — stay safe and follow army
    maneuver = CombatManeuver()

    # Layer 1: Dodge immediate danger (biles, disruptor shots, storms, etc.)
    maneuver.add(KeepUnitSafe(sentry, avoid_grid))

    # Layer 2: Kite away from enemy influence zones (same pattern as ranged/melee units)
    maneuver.add(KeepUnitSafe(sentry, grid))

    # Movement: if this Sentry has a gs_target, path toward it. For approved
    # Sentries this is their current position (cast in place — minimal movement).
    # For overlapping Sentries (not approved, but assigned a repositioning
    # peak), this moves them away from the active shield to the nearest
    # uncovered pocket so they can cast next frame. Sentries with no gs_target
    # follow ranged_center / squad_position as before.
    follow_target = gs_target if gs_target is not None else (
        ranged_center if ranged_center is not None else squad_position
    )
    success_dist = GUARDIAN_SHIELD_RADIUS if gs_target is not None else SENTRY_SQUAD_TARGET_DISTANCE
    # Sentries with a gs_target need to close to GS radius (to cast or to
    # clear the overlap). Others use the standard follow distance.
    follow_threshold = GUARDIAN_SHIELD_RADIUS if gs_target is not None else SENTRY_SQUAD_FOLLOW_DISTANCE
    distance_to_follow = cy_distance_to(sentry.position, follow_target)
    if distance_to_follow > follow_threshold:
        maneuver.add(PathUnitToTarget(
            unit=sentry,
            grid=bot.mediator.get_ground_grid,
            target=follow_target,
            success_at_distance=success_dist,
        ))

    bot.register_behavior(maneuver)
    return True
