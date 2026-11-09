import math

from sc2.data import Race
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId

from cython_extensions import cy_unit_pending, cy_structure_pending_ares, cy_distance_to

from sc2.position import Point2
from sc2.units import Units

from ares.behaviors.macro import (
    ProductionController,
    SpawnController,
    MacroPlan,
    AutoSupply,
    BuildWorkers,
    ExpansionController,
    GasBuildingController,
    UpgradeController,
    BuildStructure,
    TechUp,
)
from ares.consts import UnitRole, WORKER_TYPES, LOSS_DECISIVE_OR_WORSE
from ares.consts import ID, TARGET

from bot.utilities.performance_monitor import get_economy_state, get_resource_pressure
from bot.managers.reactions import assess_threat
from bot.utilities.debug import render_detection_cannon_debug
from bot.utilities.telemetry import log_transition
from bot.constants import (
    MEMORY_EXPIRY_TIME,
    FREEFLOW_INCOME_RATIO_THRESHOLD,
    THREAT_BLOCK_EXPANSION_LEVEL,
    EXPANSION_INTEL_URGENCY_BLOCK,
    EXPANSION_PASSIVE_ENEMY_TIME,
    BuildProfile,
    BUILD_PROFILES,
    PVT_STANDARD_2023_PROFILE,
    PVT_STALKER_2021_PROFILE,
    PVZ_STANDARD_PROFILE,
    PVZ_SENTRY_PROFILE,
    PVP_2GATE_PROFILE,
    _resolve,
    get_active_profile,
    _needs_detection_cannons,
    DETECTION_CANNON_RANGE,
    PYLON_POWER_RANGE,
    MINERAL_CLEARANCE,
)
from ares.dicts.cost_dict import COST_DICT


def get_freeflow_mode(bot) -> bool:
    """
    Freeflow calculation: detects when SpawnController should `continue` past
    unaffordable units instead of `break`ing the loop.
    
    Uses a layered approach:
      1. Early-game pressure checks (one base, PvP)
      2. Bank-imbalance checks (always active, no PM dependency)
      3. Income-aware checks (when PerformanceMonitor data is available)
    
    Args:
        bot: The bot instance
        
    Returns:
        bool: True if should use freeflow mode
    """
    minerals = bot.minerals
    vespene = bot.vespene
    
    # --- Bank-imbalance checks (always active, catches resource skew) ---
    # Both resources high — just spend
    if minerals > 500 and vespene > 500:
        return True
    
    # Significant bank with clear imbalance — one resource is 3x+ the other
    # and the dominant resource exceeds 400 (enough that we should be spending it)
    if minerals > 400 and minerals > 3 * max(vespene, 1):
        return True
    if vespene > 400 and vespene > 3 * max(minerals, 1):
        return True
    
    # --- Income-aware checks (more precise, needs PM data) ---
    pm = getattr(bot, 'performance_monitor', None)
    if pm and pm.tracking_enabled and pm.samples_taken >= 5:
        income_min = pm.avg_income_minerals
        income_gas = pm.avg_income_vespene
        
        # Severe income imbalance: one resource accumulates faster than we can
        # spend it. Trigger freeflow to keep production flowing with whatever
        # resources we DO have. No bank threshold — the imbalance itself is
        # the signal, and our Layer 1 priority reorder handles what gets built.
        if income_gas > 0 and income_min / (income_gas + 1) > FREEFLOW_INCOME_RATIO_THRESHOLD:
            return True
        if income_min > 0 and income_gas / (income_min + 1) > FREEFLOW_INCOME_RATIO_THRESHOLD:
            return True
    
    # Normal proportional production
    return False


def get_optimal_gas_workers(bot) -> int:
    """
    Dynamically adjust gas workers based on economy state.
    Prevents gas flooding and mineral starvation.
    
    Args:
        bot: The bot instance
        
    Returns:
        int: Number of workers per gas (0-3)
    """
    # Get current gatherers (workers actually mining)
    gatherers = bot.mediator.get_units_from_role(role=UnitRole.GATHERING)
    
    # Critical mineral starvation - stop gas completely
    if (bot.minerals < 100 
        and bot.vespene > 300 
        and bot.supply_used < 84):
        return 0
    
    # Early game with few workers - stop gas temporarily
    if ((bot.reaction_manager.is_cheese_response and len(gatherers) < 21)
        or len(gatherers) < 12):
        return 0
    
    # Vespene flooding - toggle off gas mining
    if bot._gas_worker_toggle and bot.vespene > 1500 and bot.minerals < 100:
        bot._gas_worker_toggle = False
        return 0
    
    # Vespene starved - toggle gas mining back on
    if bot.vespene < 400 or bot.minerals > 1200:
        bot._gas_worker_toggle = True
    
    # Low gas - max workers
    if bot.vespene < 100:
        return 3
    
    return 3 if bot._gas_worker_toggle else 0

# Army composition constants 
STANDARD_ARMY_0 = {
    UnitTypeId.IMMORTAL: {"proportion": 0.15, "priority": 1},
    UnitTypeId.COLOSSUS: {"proportion": 0.15, "priority": 4},
    UnitTypeId.HIGHTEMPLAR: {"proportion": 0.4, "priority": 0},
    UnitTypeId.DISRUPTOR: {"proportion": 0.1, "priority": 5},
    UnitTypeId.STALKER: {"proportion": 0.0, "priority": 2},
    UnitTypeId.ZEALOT: {"proportion": 0.2, "priority": 3},
}
STANDARD_ARMY_1 = {
    UnitTypeId.IMMORTAL: {"proportion": 0.35, "priority": 1},
    UnitTypeId.COLOSSUS: {"proportion": 0.2, "priority": 4},
    UnitTypeId.DISRUPTOR: {"proportion": 0.1, "priority": 3},
    UnitTypeId.STALKER: {"proportion": 0.0, "priority": 2},
    UnitTypeId.ZEALOT: {"proportion": 0.35, "priority": 0},
}

# Parting PvZ Soul Train: Immortal/Sentry/Zealot all-in with Warp Prism.
# No HT — build has no Templar Archive; gas goes to Robo units instead.
# army_0 = main all-in push. Proportions from guide: 10 Zealots, 6 Sentries, 3 Immortals.
PVZ_SENTRY_ARMY_0 = {
    UnitTypeId.IMMORTAL: {"proportion": 0.16, "priority": 0},
    UnitTypeId.SENTRY: {"proportion": 0.32, "priority": 1},
    UnitTypeId.ZEALOT: {"proportion": 0.53, "priority": 2},
}
# army_1 = recovery comp if the all-in fails. Adds Disruptor/Colossus via RoboBay.
PVZ_SENTRY_ARMY_1 = {
    UnitTypeId.STALKER: {"proportion": 0.25, "priority": 0},
    UnitTypeId.IMMORTAL: {"proportion": 0.20, "priority": 1},
    UnitTypeId.COLOSSUS: {"proportion": 0.20, "priority": 2},
    UnitTypeId.DISRUPTOR: {"proportion": 0.15, "priority": 3},
    UnitTypeId.ZEALOT: {"proportion": 0.15, "priority": 4},
    UnitTypeId.SENTRY: {"proportion": 0.05, "priority": 5},
}

PVP_ARMY_0 = {
    UnitTypeId.STALKER: {"proportion": 0.3, "priority": 0},
    UnitTypeId.IMMORTAL: {"proportion": 0.15, "priority": 1},
    UnitTypeId.DISRUPTOR: {"proportion": 0.2, "priority": 2},
    UnitTypeId.HIGHTEMPLAR: {"proportion": 0.2, "priority": 3},
    UnitTypeId.COLOSSUS: {"proportion": 0.15, "priority": 4},
}

PVP_ARMY_1 = {
    UnitTypeId.STALKER: {"proportion": 0.35, "priority": 0},
    UnitTypeId.DISRUPTOR: {"proportion": 0.3, "priority": 1},
    UnitTypeId.COLOSSUS: {"proportion": 0.15, "priority": 2},
    UnitTypeId.IMMORTAL: {"proportion": 0.2, "priority": 3},
}

CHEESE_DEFENSE_ARMY = {
    UnitTypeId.ADEPT: {"proportion": 0.25, "priority": 2},
    UnitTypeId.STALKER: {"proportion": 0.15, "priority": 0},
    UnitTypeId.ZEALOT: {"proportion": 0.6, "priority": 1},
    
    
}

# 2021 PvT Stalker-Centric army: pure Stalker + Zealot, no Robo units
PVT_STALKER_2021_ARMY = {
    UnitTypeId.STALKER: {"proportion": 0.65, "priority": 0},
    UnitTypeId.ZEALOT: {"proportion": 0.35, "priority": 1},
}

# 2021 PvT Stalker-Centric moderate economy army: adds HT + Sentry for Storm/Feedback
# Proportions: Stalker/Zealot scaled down to 80% of base, HT 15%, Sentry 5%
PVT_STALKER_2021_ARMY_MODERATE = {
    UnitTypeId.STALKER: {"proportion": 0.52, "priority": 0},
    UnitTypeId.HIGHTEMPLAR: {"proportion": 0.13, "priority": 1},
    UnitTypeId.ZEALOT: {"proportion": 0.33, "priority": 2},
    UnitTypeId.SENTRY: {"proportion": 0.02, "priority": 3},
}

# 2021 PvT Stalker-Centric post-Archon army: HTs have merged into Archons,
# redistribute HT proportion to Stalker/Zealot. Keep small HT production for Storm.
# Stop heavy HT production, let existing HTs merge, keep a small HT line for
# Storm/Feedback. The scorer picks this mix when it outscores the others.
PVT_STALKER_2021_ARMY_FULL = {
    UnitTypeId.STALKER: {"proportion": 0.50, "priority": 0},
    UnitTypeId.ZEALOT: {"proportion": 0.35, "priority": 1},
    UnitTypeId.HIGHTEMPLAR: {"proportion": 0.10, "priority": 2},
    UnitTypeId.SENTRY: {"proportion": 0.05, "priority": 3},
}

# Wire army composition dicts into BuildProfile instances
# (profiles are defined in constants.py with empty dicts to avoid circular imports)
PVT_STANDARD_2023_PROFILE.army_composition_0 = STANDARD_ARMY_0
PVT_STANDARD_2023_PROFILE.army_composition_1 = STANDARD_ARMY_1
PVT_STALKER_2021_PROFILE.army_composition_0 = PVT_STALKER_2021_ARMY
PVT_STALKER_2021_PROFILE.army_composition_1 = PVT_STALKER_2021_ARMY_MODERATE  # Scored candidate
PVT_STALKER_2021_PROFILE.army_composition_2 = PVT_STALKER_2021_ARMY_FULL  # Scored candidate
PVZ_STANDARD_PROFILE.army_composition_0 = STANDARD_ARMY_0  # Same robo-centric comp as PvT
PVZ_STANDARD_PROFILE.army_composition_1 = STANDARD_ARMY_1
PVZ_SENTRY_PROFILE.army_composition_0 = PVZ_SENTRY_ARMY_0
PVZ_SENTRY_PROFILE.army_composition_1 = PVZ_SENTRY_ARMY_1
PVP_2GATE_PROFILE.army_composition_0 = PVP_ARMY_0
PVP_2GATE_PROFILE.army_composition_1 = PVP_ARMY_1

# Wire conditional upgrades (reactive, game-state dependent)
# These predicates reference helper functions defined below, so they're set up here.
_thermal_lance_predicate = lambda bot: bool(
    bot.structures(UnitTypeId.ROBOTICSBAY)
    and (bot.units(UnitTypeId.COLOSSUS).amount or cy_unit_pending(bot, UnitTypeId.COLOSSUS))
)
_storm_predicate = lambda bot: bool(
    bot.structures(UnitTypeId.TEMPLARARCHIVE)
    and (
        (bot.enemy_race in {Race.Terran, Race.Zerg} and _count_enemy_anti_air(bot) >= 2)
        or (bot.enemy_race == Race.Protoss and _count_enemy_carriers(bot) >= 1)
    )
)
PVT_STANDARD_2023_PROFILE.conditional_upgrades = [
    (UpgradeId.EXTENDEDTHERMALLANCE, _thermal_lance_predicate),
    (UpgradeId.PSISTORMTECH, _storm_predicate),
]
PVZ_STANDARD_PROFILE.conditional_upgrades = [
    (UpgradeId.EXTENDEDTHERMALLANCE, _thermal_lance_predicate),
    (UpgradeId.PSISTORMTECH, _storm_predicate),
]
PVZ_SENTRY_PROFILE.conditional_upgrades = [
    (UpgradeId.EXTENDEDTHERMALLANCE, _thermal_lance_predicate),
]
PVP_2GATE_PROFILE.conditional_upgrades = [
    (UpgradeId.EXTENDEDTHERMALLANCE, _thermal_lance_predicate),
    (UpgradeId.PSISTORMTECH, _storm_predicate),
]
# 2021 profile: conditional upgrades are economy-gated (Weapons/Armor 2-3)

# ===== PRODUCTION NUDGING =====
# The composition competition lives in bot/managers/composition_scoring.py: the
# scoring table is the core base score there, not a post-decision nudge. The
# only nudge left in this module is the production-mechanics priority reorder
# (reorder_priorities_by_resources), which keeps the SpawnController alive.

ANTI_COLOSSUS_TYPES = {
    UnitTypeId.VIKINGFIGHTER, UnitTypeId.VIKINGASSAULT, UnitTypeId.CORRUPTOR,
    UnitTypeId.PHOENIX, UnitTypeId.CARRIER, UnitTypeId.TEMPEST,
}
"""Enemy unit types that hard-counter Colossus. Drives nudging in all matchups."""


def _count_enemy_anti_air(bot) -> int:
    """Count enemy anti-Colossus air units from cached intel."""
    return sum(1 for u in _get_enemy_combat_units(bot) if u.type_id in ANTI_COLOSSUS_TYPES)


def _count_enemy_carriers(bot) -> int:
    """Count enemy Carriers from cached intel. Used for PvP Storm trigger."""
    return sum(1 for u in _get_enemy_combat_units(bot) if u.type_id == UnitTypeId.CARRIER)


def _get_enemy_combat_units(bot) -> list:
    """Get filtered enemy combat units for production analysis.

    Excludes workers, structures, and expired ghost units (age >= MEMORY_EXPIRY_TIME).
    """
    cached = bot.mediator.get_cached_enemy_army or []
    return [
        u for u in cached
        if u.type_id not in WORKER_TYPES
        and not u.is_structure
        and u.age < MEMORY_EXPIRY_TIME
    ]


def _get_gas_ratio(unit_type: UnitTypeId) -> float:
    """Get the gas fraction of a unit's total cost using ARES COST_DICT.
    
    Returns 0.0 for mineral-only units, 1.0 for gas-only units,
    ~0.5 for balanced cost units. Uses dynamic lookup — no hardcoded costs.
    
    Perf note: dict lookup, called once per unit type per macro cycle.
    """
    cost = COST_DICT.get(unit_type)
    if cost is None:
        return 0.5  # Unknown unit, assume balanced
    total = cost.minerals + cost.vespene
    if total == 0:
        return 0.0
    return cost.vespene / total


def reorder_priorities_by_resources(composition: dict, bot) -> dict:
    """Layer 4: Push unaffordable units behind affordable ones in priority order.

    Only reorders units that can't currently be afforded — those are the ones
    that would cause the SpawnController's non-freeflow `break` to kill the
    entire production loop. Affordable units keep the effectiveness-based
    priority the scorer gave them, so the strategic choice of
    "build the most effective unit first" is preserved.

    When balanced (no resource pressure): return composition unchanged.
    When gas-rich/mineral-starved: unaffordable high-gas units stay near front
    (they might be affordable next frame when gas accumulates), unaffordable
    low-gas units go to back (we're waiting for minerals, not gas).
    When mineral-rich/gas-starved: the reverse.

    The composition dict is never mutated — returns a new dict.

    Perf note: O(k log k) sort where k = unit types in comp (~5). Negligible.

    Args:
        composition: Army composition dict {UnitTypeId: {"proportion": ..., "priority": ...}}
        bot: Bot instance for resource access

    Returns:
        New composition dict with reordered priorities (proportions unchanged)
    """
    # Detect imbalance direction via shared economy signal (instantaneous bank)
    pressure = get_resource_pressure(bot, sustained=False)
    mineral_rich = pressure == "GAS_STARVED"
    gas_rich = pressure == "MINERAL_STARVED"

    if not mineral_rich and not gas_rich:
        return composition  # Balanced — keep original priorities

    # Split units into affordable (keep effectiveness priority) and
    # unaffordable (sort by gas ratio to push them to the back).
    # This prevents the SpawnController break while preserving the
    # scorer's effectiveness-based priority for units we can build.
    affordable: list = []
    unaffordable: list = []
    for unit_type in composition:
        if bot.can_afford(unit_type):
            affordable.append(unit_type)
        else:
            unaffordable.append(unit_type)

    # No unaffordable units — nothing to reorder
    if not unaffordable:
        return composition

    # Sort unaffordable by gas_ratio: ascending if mineral-rich (cheap-gas
    # first — they're closer to affordable), descending if gas-rich
    unaffordable.sort(
        key=lambda ut: _get_gas_ratio(ut),
        reverse=gas_rich,
    )

    # Affordable units keep their existing priority (from the scorer).
    # Unaffordable units get pushed behind all affordable ones.
    # Within each group, preserve relative order (stable sort).
    affordable.sort(key=lambda ut: composition[ut]["priority"])
    ordered = affordable + unaffordable

    reordered: dict = {}
    for new_priority, unit_type in enumerate(ordered):
        info = composition[unit_type]
        reordered[unit_type] = {
            "proportion": info["proportion"],
            "priority": new_priority,
        }

    return reordered


# No static upgrade list - use get_desired_upgrades() instead

def calculate_optimal_worker_count(bot) -> int:
    """
    Calculates the optimal number of workers based on available resources.
    Checks mineral contents and vespene contents to determine if mining positions are viable.
    
    Returns:
        int: The optimal number of workers to build
    """
    mineral_spots = 0
    for townhall in bot.townhalls:
        nearby_minerals = bot.mineral_field.closer_than(10, townhall)
        for mineral in nearby_minerals:
            if mineral.mineral_contents > 100:  
                mineral_spots += 2
    
    gas_spots = 0
    for gas_building in bot.gas_buildings:
        if gas_building.vespene_contents > 100:
            gas_spots += 3
    
    # Add a small buffer for worker replacements
    worker_count = mineral_spots + gas_spots + 4
    
    return worker_count


def is_base_depleted(bot) -> bool:
    """Accurate base depletion using ARES data + python-sc2 mineral contents"""
    
    # Method 1: Check mineral contents directly (python-sc2)
    depleted_bases = 0
    for townhall in bot.townhalls:
        nearby_minerals = bot.mineral_field.closer_than(10, townhall)
        if nearby_minerals:
            depleted_patches = len([m for m in nearby_minerals if m.mineral_contents < 300])
            if depleted_patches / len(nearby_minerals) > 0.5:
                depleted_bases += 1
    
    # Method 2: Use ARES worker assignment data
    mineral_assignments = bot.mediator.get_mineral_patch_to_list_of_workers
    available_patches = bot.mediator.get_num_available_mineral_patches
    
    # Count oversaturated patches (3+ workers per patch)
    oversaturated_count = sum(1 for workers in mineral_assignments.values() 
                             if len(workers) >= 3)
    
    gas_workers = len(bot.mediator.get_worker_to_vespene_dict)
    mining_workers = bot.workers.amount - gas_workers
    
    return (
        depleted_bases > 0                          # Some bases have depleted patches
        or available_patches < 4                    # Very few available patches  
        or (mining_workers > 0 and available_patches == 0)  # Workers but no patches
        or oversaturated_count > 2                  # Too many oversaturated patches
    )


def _can_expand_natural_under_attack(bot) -> bool:
    """
    Simple defender's advantage check: allow the natural expansion during an
    attack if we have a shield battery near the natural and haven't taken it yet.

    The natural is behind our wall — the defender's advantage (wall, battery,
    high ground) lets a smaller army hold while the Nexus builds. The
    ExpansionController's own ground-grid safety check is the final gate: if
    enemies are physically at the natural, it won't place the Nexus.

    This is the simple version — see plan/expansion_improvement_plan.md #6 for
    the enhanced version (wall building check, threat-level thresholds, etc.).
    """
    # Only applies when expanding to the natural (1 base → 2)
    if len(bot.townhalls) >= 2:
        return False

    # Must have a completed shield battery near the natural
    nat_pos = bot.mediator.get_own_nat
    battery_at_nat = bot.structures(UnitTypeId.SHIELDBATTERY).ready.filter(
        lambda b: cy_distance_to(b.position, nat_pos) < 20.0
    )
    return bool(battery_at_nat)


def _has_map_control(bot) -> bool:
    """
    Determines whether the bot has enough map control to safely expand.

    Based on pro expansion principles (race-neutral signals only):
      - Expand behind aggression (commenced attack = map control)
      - Don't expand under direct attack or combat threats near bases
      - Don't expand blind when intel is very stale (enemy could be timing)
      - Expand if opponent is passive/turtling (army not seen in a long time)
      - Expand if no enemy army has ever been seen (no known threat)
      - Defender's advantage: allow natural expansion behind a battery
        even when under attack (simple version; see plan for enhancement)

    The ExpansionController independently verifies the expansion location is
    safe via the ground grid, so this function only answers: "is it generally
    safe to send a worker out?"

    Race-specific signals (opponent base-count mirroring, all-in detection,
    worker economy comparison) are deferred — see plan/expansion_improvement_plan.md.
    """
    # ── HARD GATES: block expansion entirely ──────────────────────────

    # Gate 1: Under direct attack — defense first, never expand.
    # Exception: defender's advantage allows the natural behind a battery.
    if bot._under_attack:
        if not _can_expand_natural_under_attack(bot):
            return False

    # Gate 2: Combat threat near our bases — defense system is engaged
    # Uses the threat system (assess_threat) for consistency with defense logic.
    # Only blocks for real combat threats, not harassment handled by defenders.
    ground_near = bot.mediator.get_ground_enemy_near_bases
    flying_near = bot.mediator.get_flying_enemy_near_bases
    all_threat_tags: set[int] = set()
    for enemy_tags in ground_near.values():
        all_threat_tags.update(enemy_tags)
    for enemy_tags in flying_near.values():
        all_threat_tags.update(enemy_tags)

    if all_threat_tags:
        threatening_units = bot.enemy_units.tags_in(all_threat_tags)
        if threatening_units:
            threat_info = assess_threat(bot, threatening_units, bot.own_army, return_details=True)
            assert isinstance(threat_info, dict), "assess_threat with return_details=True should return dict"
            if threat_info.get("threat_level", 0) >= THREAT_BLOCK_EXPANSION_LEVEL:
                return False

    # ── POSITIVE SIGNALS: expand when any is true ──────────────────────

    # Signal 1: "Expand behind aggression" — army has commenced attack.
    # The opponent is forced to defend, buying time for the expansion.
    # Reuses handle_attack_toggles' global can_win_fight decision (proper
    # unit filtering + intel-quality gating) instead of a second sim.
    if bot._commenced_attack:
        return True

    # Signal 2: No enemy army ever seen — no known threat exists.
    # The ExpansionController still verifies the location via ground grid.
    if not bot._enemy_army_ever_seen:
        return True

    # Signal 3: "Opponent is turtling/passive" — we've seen the enemy army
    # before, but haven't seen it in a long time AND no threats are near our
    # bases. If the opponent isn't attacking, we can safely invest in economy.
    # Race-neutral: a passive opponent isn't pressuring us regardless of race.
    time_since_seen = bot.time - bot._last_enemy_army_visible_time
    if time_since_seen > EXPANSION_PASSIVE_ENEMY_TIME:
        return True

    # ── CAUTIOUS GATES: grey zone (seen enemy recently, not attacking) ──

    # We've seen the enemy recently but our army hasn't commenced attack.
    # This is the danger zone: the enemy could be setting up a timing attack.
    # Gate on intel freshness: if intel is very stale (urgency high), we're
    # blind to the enemy's current position — don't expand blind.
    # If intel is fresh enough, we have enough recent info to risk expanding.
    if bot._intel_urgency >= EXPANSION_INTEL_URGENCY_BLOCK:
        return False

    # Intel is fresh enough: we've seen the enemy recently and have a
    # reasonable picture of their army. The threat system didn't flag a
    # blocking threat. The ExpansionController's location safety check
    # is the final gate.
    return True


def expansion_checker(bot, main_army) -> int:
    """
    Evaluates multiple factors to determine when to expand:
    1. Map control (using army's global assessment + enemy proximity)
    2. Resource starvation (production idle due to lack of income)
    3. Base depletion (worker saturation with declining income)
    4. Spending efficiency using python-sc2 score metrics

    Returns the recommended expansion count.
    """
    current_bases = len(bot.townhalls)
    current_workers = bot.workers.amount
    optimal_workers = calculate_optimal_worker_count(bot)
    worker_saturation = current_workers / optimal_workers if optimal_workers > 0 else 0

    mineral_collection_rate = bot.state.score.collection_rate_minerals
    idle_production_time = bot.state.score.idle_production_time
    expansion_count = current_bases

    # Calculate spending efficiency (SQ-style)
    current_unspent = bot.minerals
    spending_efficiency = mineral_collection_rate / (current_unspent + 1) if mineral_collection_rate > 0 else 0

    # Map control gate: if we don't have map control, don't add new expansions.
    # Unlike the old army_safe early-return, this doesn't bypass the game-state
    # fallback below — it only prevents *new* expansion_count increments.
    has_map_control = _has_map_control(bot)

    resource_starved = (
        idle_production_time > 30.0  # Production buildings idle for 30+ seconds
        and current_unspent < 500    # Low mineral bank
        and mineral_collection_rate > 0  # But we do have some income
    )

    bases_depleting = is_base_depleted(bot)

    # Spending efficiency thresholds (dynamic by game state)
    if bot.game_state == 0:      # Early: should spend quickly
        efficiency_threshold = 1.5
    elif bot.game_state == 1:    # Mid: more complex economy
        efficiency_threshold = 1.0
    else:                        # Late: can bank for big investments
        efficiency_threshold = 0.5

    # Low spending efficiency = banking too much relative to income
    inefficient_spending = spending_efficiency < efficiency_threshold

    if has_map_control:
        if resource_starved:
            expansion_count = current_bases + 1
        elif bases_depleting:
            expansion_count = current_bases + 1
        elif inefficient_spending and worker_saturation > 0.7:
            expansion_count = current_bases + 1

    # Game state-based fallback expansions (minimal safety net).
    # Gated behind map control: don't expand while under pressure even if
    # the game-state timer says we "should" have more bases by now.
    # The ExpansionController independently verifies the location is safe.
    if has_map_control:
        if bot.game_state >= 1 and current_bases < 2:
            expansion_count = max(expansion_count, 2)
        elif bot.game_state >= 2 and current_bases < 3:
            expansion_count = max(expansion_count, 3)

    # Debug snapshot — read by render_expansion_debug() in debug.py
    triggered_by = "none"
    if has_map_control:
        if resource_starved:
            triggered_by = "starved"
        elif bases_depleting:
            triggered_by = "depleting"
        elif inefficient_spending and worker_saturation > 0.7:
            triggered_by = "inefficient"
        elif bot.game_state >= 1 and current_bases < 2:
            triggered_by = "gamestate_mid"
        elif bot.game_state >= 2 and current_bases < 3:
            triggered_by = "gamestate_late"

    bot._expansion_debug = {
        "current_bases": current_bases,
        "target_bases": expansion_count,
        "wants_expand": expansion_count > current_bases,
        "has_map_control": has_map_control,
        "triggered_by": triggered_by,
        "resource_starved": resource_starved,
        "bases_depleting": bases_depleting,
        "inefficient_spending": inefficient_spending,
        "worker_saturation": worker_saturation,
        "spending_efficiency": spending_efficiency,
        "efficiency_threshold": efficiency_threshold,
    }

    return expansion_count


def _safe_to_invest_in_tech(bot) -> bool:
    """Is it safe to spend minerals on tech buildings/upgrades right now?

    Uses two dimensions:
      1. Resource pressure: can the bank absorb the cost?
         (same signal as GasBuildingController and nudge pipeline)
      2. Situation severity: are we in a fight we might lose?
         Combines _under_attack / is_early_defensive with the combat sim
         margin. Being under attack alone doesn't block tech — winning
         while under attack is exactly when tech investment pays off.
         Only block when we're losing badly enough that every mineral
         needs to go to army.
    """
    # Bank is drained — can't afford tech buildings without starving army
    if get_resource_pressure(bot, sustained=False) != "BALANCED":
        return False

    # Check if we're in a threatened situation at all
    under_pressure = (
        getattr(bot, '_under_attack', False)
        or bot.reaction_manager.is_early_defensive
    )

    if not under_pressure:
        return True  # Safe macro — resource pressure is the only concern

    # We're under pressure. Check the combat sim to see if we're holding.
    # The sim accounts for positioning (good_positioning=True), so a
    # defender's advantage is reflected in the result.
    try:
        own_combat = [u for u in bot.own_army if u.type_id not in WORKER_TYPES]
        enemy_combat = [u for u in bot.enemy_army if u.type_id not in WORKER_TYPES and not u.is_structure]
        if not own_combat or not enemy_combat:
            return True  # No fight to sim — safe to invest

        fight_result = bot.mediator.can_win_fight(
            own_units=own_combat,
            enemy_units=enemy_combat,
            timing_adjust=True,
            good_positioning=True,
            workers_do_no_damage=True,
        )
        # TIE or better: we're holding — tech investment will pay off
        # when the attack breaks. LOSS_DECISIVE or worse: every mineral
        # needs to go to army production.
        if fight_result in LOSS_DECISIVE_OR_WORSE:
            return False
        return True
    except Exception:
        return True  # Sim failed — don't block upgrades on uncertainty


def get_desired_upgrades(bot) -> list[UpgradeId]:
    """
    Returns dynamic upgrade list based on the active BuildProfile and game state.
    UpgradeController will automatically build required tech structures.
    """
    profile = get_active_profile(bot)
    upgrades: list[UpgradeId] = []
    
    # Static upgrade order from profile
    for upgrade in profile.upgrade_order:
        if not bot.pending_or_complete_upgrade(upgrade):
            upgrades.append(upgrade)
    
    # Conditional upgrades (reactive, game-state dependent)
    for upgrade, predicate in profile.conditional_upgrades:
        if not bot.pending_or_complete_upgrade(upgrade) and predicate(bot):
            upgrades.append(upgrade)
    
    # Gate 1: Economy can't sustain investment long-term (income too low)
    economy_state = get_economy_state(bot)
    if economy_state in ("recovery", "reduced"):
        return upgrades
    
    # Gate 2: Army too small — need units before upgrades
    if bot.supply_army < 15:
        return upgrades
    
    # Gate 3: Situational safety — can we afford to divert minerals from
    # army production right now? Checks bank balance, threat state, and
    # combat sim margin. Being under attack doesn't block tech — only
    # blocks when losing badly (LOSS_DECISIVE or worse). Prevents the
    # post-build-runner tech cascade from draining the mineral bank.
    if not _safe_to_invest_in_tech(bot):
        return []
    
    return upgrades


def _train_warp_prism(bot) -> None:
    """Train Warp Prisms to target count from the active BuildProfile.
    Supports dynamic warp_prism_target (e.g., 2023 build returns 1 when tech/supply met, 0 otherwise).
    Returns early if profile says 0 warp prisms.
    """
    profile = get_active_profile(bot)
    target_count = _resolve(profile.warp_prism_target, bot)
    if target_count == 0:
        return

    total_prisms = (bot.units(UnitTypeId.WARPPRISM).amount +
                    bot.units(UnitTypeId.WARPPRISMPHASING).amount +
                    bot.already_pending(UnitTypeId.WARPPRISM))
    if total_prisms >= target_count:
        return
    if not bot.can_afford(UnitTypeId.WARPPRISM):
        return

    robotics_facilities = bot.structures(UnitTypeId.ROBOTICSFACILITY).ready.idle
    for facility in robotics_facilities:
        facility.train(UnitTypeId.WARPPRISM)
        return


def require_shield_battery(bot) -> bool:
    """
    Returns True if shield battery should be built after cheese response.
    Limits to one battery total. Tech requirements checked by BuildStructure.
    """
    total_batteries = (bot.structures(UnitTypeId.SHIELDBATTERY).amount + 
                      cy_structure_pending_ares(bot, UnitTypeId.SHIELDBATTERY))
    
    return bot.reaction_manager.is_cheese_response and total_batteries == 0


def get_shield_battery_base_location(bot) -> Point2:
    """
    Determines which base location to build shield battery at:
    - At natural if buildings or townhall exist there
    - At main base otherwise (uses static_defence placement)
    """
    natural_location = bot.mediator.get_own_nat
    
    natural_has_base = bot.townhalls.closer_than(10, natural_location).amount > 0
    natural_has_structures = bot.structures.closer_than(10, natural_location).amount > 0
    
    if natural_has_base or natural_has_structures:
        return natural_location
    else:
        # Use main base - BuildStructure with static_defence=True will find defensive spot
        return bot.start_location


def get_desired_gateway_count(bot) -> int:
    """
    Returns desired gateway/warpgate count based on the active BuildProfile.
    Walks gateway_thresholds in reverse (highest nexus req first), returns first match.
    """
    profile = get_active_profile(bot)
    bases = len(bot.townhalls)
    for nexus_req, gate_count in reversed(profile.gateway_thresholds):
        if bases >= nexus_req:
            return gate_count
    return profile.gateway_thresholds[0][1]  # fallback to first entry


def get_desired_forge_count(bot) -> int:
    """
    Returns desired forge count based on the active BuildProfile.
    Supports int (static) or callable (dynamic) values.
    """
    profile = get_active_profile(bot)
    return _resolve(profile.forge_count, bot)


def select_army_composition(bot) -> dict:
    """Pick the composition to build — the scorer's winner, priority-fixed.

    The competition itself lives in bot/managers/composition_scoring.py: the
    profile compositions, the hold option and the counter-driven variants are
    scored against the believed enemy army on the macro cadence, and the winner
    builds. This function adapts that winner to production and reports it.

    Steps:
        1. Ask the scorer for the winning composition (cached between re-scores)
        2. Priority reorder — production mechanics, not strategy
        3. Emit the choice transition + cache it for the debug overlay

    Returns:
        dict: The selected army composition dictionary
    """
    profile = get_active_profile(bot)
    selection = bot.composition_scorer.select(bot)
    candidate = selection.candidate
    # No candidate (empty ballot) — fall back to the profile's base build.
    composition = candidate.composition if candidate else profile.army_composition_0

    # Priority reorder — push unaffordable units behind affordable ones.
    # Only touches units we can't afford right now; affordable units keep the
    # effectiveness-based priority the scorer gave them. Prevents SpawnController
    # break in non-freeflow mode without overriding the strategic choice.
    composition = reorder_priorities_by_resources(composition, bot)

    _report_composition_choice(bot, selection)

    # Cache for debug overlay
    bot._last_base_comp = profile.army_composition_0
    bot._last_selected_comp = composition
    # Resource pressure state for debug display
    bot._resource_pressure = get_resource_pressure(bot, sustained=False)

    return composition


def _report_composition_choice(bot, selection) -> None:
    """Emit the composition transition — the new telemetry gate for macro.

    A transition event fires only when the winning candidate changes, so the
    event marks decisions instead of the macro cycle. The winner's score and
    the runner-up's ride along: the gap is what the commitment bonus is buying.
    """
    ranking = selection.ranking
    winner = ranking[0] if ranking else None
    runner_up = ranking[1] if len(ranking) > 1 else None
    log_transition(
        subsystem="macro",
        action="composition_choice",
        reason="scored_competition",
        key="composition_choice",
        value=winner.choice.name if winner else "none",
        winner_score=round(winner.score, 4) if winner else 0.0,
        runner_up_score=round(runner_up.score, 4) if runner_up else 0.0,
        candidates=len(selection.candidates),
        _ts=bot.time,
    )


def _train_observers(bot) -> None:
    """Train observers to target count from the active BuildProfile.
    Supports dynamic observer_target (e.g., 2021 build returns 1 when detection needed, 0 otherwise).
    Returns early if profile says 0 observers and no detection trigger is active.

    Uses already_pending() instead of cy_unit_pending() because cy_unit_pending only
    inspects orders[0] on each production structure. When an Observer is queued behind
    another unit (e.g. Immortal in slot 0), cy_unit_pending returns 0 and the function
    would re-issue an observer train order every step until the front unit completes —
    producing the "3 observers popping out at once" pile-up. already_pending iterates
    all queued orders via _abilities_count_and_build_progress, so it sees the whole queue.
    """
    profile = get_active_profile(bot)
    target_count = _resolve(profile.observer_target, bot)
    if target_count == 0:
        return

    observer_count = (
        bot.units(UnitTypeId.OBSERVER).amount
        + bot.units(UnitTypeId.OBSERVERSIEGEMODE).amount
    )
    pending = bot.already_pending(UnitTypeId.OBSERVER)
    total_observers = observer_count + pending

    if pending or not bot.can_afford(UnitTypeId.OBSERVER):
        return

    robotics_facilities = bot.structures(UnitTypeId.ROBOTICSFACILITY).ready
    if total_observers < 1:
        for facility in robotics_facilities:
            facility.train(UnitTypeId.OBSERVER)
            return

    if bot.game_state == 0:
        return

    # Non-urgent: only train on idle Robos to avoid queuing behind other units.
    if total_observers < target_count:
        for facility in robotics_facilities.idle:
            facility.train(UnitTypeId.OBSERVER)
            return


def build_detection_cannons(bot) -> None:
    """Build Pylon + Photon Cannon behind mineral lines at each owned base when
    harass threats are seen (Banshee/DT/Oracle/Widow Mine). Uses a per-base
    state machine to track progress: needs_pylon → pylon_pending → needs_cannon
    → cannon_pending → complete.

    Purpose: Provide anti-harass detection coverage for all builds.
    Key Decisions: Per-base state machine prevents duplicate orders; proximity-based worker
        selection naturally picks probes from the target base. All bases build in parallel
        — no sequential blocking between bases. Mineral clearance ensures structures don't
        block probe pathing.
    Limitations: Only builds 1 cannon per base; relies on Forge being built by macro.
    """
    # Gate: need a completed Forge to build cannons
    if not bot.structures(UnitTypeId.FORGE).ready.exists:
        return

    # Prune state entries for dead Nexuses
    alive_tags = {nexus.tag for nexus in bot.townhalls.ready}
    bot._detection_cannon_state = {
        tag: state for tag, state in bot._detection_cannon_state.items()
        if tag in alive_tags
    }

    for nexus in bot.townhalls.ready:
        # Get behind-mineral-line positions: [left, center, right]
        behind_mineral = bot.mediator.get_behind_mineral_positions(
            th_pos=nexus.position
        )
        if not behind_mineral or len(behind_mineral) < 2:
            continue

        # Get mineral fields at this base for clearance checks
        mineral_fields = bot.mineral_field.closer_than(10, nexus.position)

        mineral_center = behind_mineral[1]  # Center position behind mineral line
        state = bot._detection_cannon_state.get(nexus.tag, "needs_pylon")

        # --- State: needs_pylon ---
        if state == "needs_pylon":
            # Check if a Pylon already exists near this base's mineral line
            existing_pylons = bot.structures(UnitTypeId.PYLON).closer_than(
                DETECTION_CANNON_RANGE, mineral_center
            )
            if existing_pylons.exists:
                bot._detection_cannon_state[nexus.tag] = "needs_cannon"
                continue

            # Check if a Pylon is already pending for THIS specific base
            # (use building tracker to avoid blocking other bases)
            pending_for_this_base = False
            building_tracker = bot.mediator.get_building_tracker_dict
            for worker_tag, info in building_tracker.items():
                if info.get(ID) == UnitTypeId.PYLON:
                    target = info.get(TARGET)
                    if target and cy_distance_to(target.position, mineral_center) < DETECTION_CANNON_RANGE:
                        pending_for_this_base = True
                        break
            if pending_for_this_base:
                continue

            # Find a valid placement position for the Pylon
            placement = _find_pylon_placement(bot, behind_mineral, nexus.position, mineral_fields)

            if placement is None:
                continue

            if not bot.can_afford(UnitTypeId.PYLON):
                continue

            worker = bot.mediator.select_worker(
                target_position=placement, force_close=True
            )
            if worker is None:
                continue

            success = bot.mediator.build_with_specific_worker(
                worker=worker,
                structure_type=UnitTypeId.PYLON,
                pos=placement,
            )
            if success:
                bot._detection_cannon_state[nexus.tag] = "pylon_pending"

        # --- State: pylon_pending ---
        elif state == "pylon_pending":
            existing_pylons = bot.structures(UnitTypeId.PYLON).closer_than(
                DETECTION_CANNON_RANGE, mineral_center
            )
            if existing_pylons.exists:
                ready_pylons = existing_pylons.ready
                if ready_pylons.exists:
                    bot._detection_cannon_state[nexus.tag] = "needs_cannon"
                # Pylon exists but not ready yet — stay in pylon_pending
            else:
                # No pylon at all — build may have been cancelled, retry
                bot._detection_cannon_state[nexus.tag] = "needs_pylon"

        # --- State: needs_cannon ---
        elif state == "needs_cannon":
            # Check if a Cannon already exists near this base's mineral line
            existing_cannons = bot.structures(UnitTypeId.PHOTONCANNON).closer_than(
                DETECTION_CANNON_RANGE, mineral_center
            )
            if existing_cannons.exists:
                bot._detection_cannon_state[nexus.tag] = "complete"
                continue

            # Find the pylon that will power the cannon
            existing_pylons = bot.structures(UnitTypeId.PYLON).closer_than(
                DETECTION_CANNON_RANGE, mineral_center
            )
            if not existing_pylons.ready.exists:
                # Pylon not ready yet — go back to waiting
                bot._detection_cannon_state[nexus.tag] = "pylon_pending"
                continue

            pylon = existing_pylons.ready.closest_to(mineral_center)

            # Find a valid placement position for the Cannon
            placement = _find_cannon_placement(bot, behind_mineral, pylon, mineral_center, nexus.position, mineral_fields)

            if placement is None:
                continue

            if not bot.can_afford(UnitTypeId.PHOTONCANNON):
                continue

            worker = bot.mediator.select_worker(
                target_position=placement, force_close=True
            )
            if worker is None:
                continue

            success = bot.mediator.build_with_specific_worker(
                worker=worker,
                structure_type=UnitTypeId.PHOTONCANNON,
                pos=placement,
            )
            if success:
                bot._detection_cannon_state[nexus.tag] = "cannon_pending"

        # --- State: cannon_pending ---
        elif state == "cannon_pending":
            existing_cannons = bot.structures(UnitTypeId.PHOTONCANNON).closer_than(
                DETECTION_CANNON_RANGE, mineral_center
            )
            if existing_cannons.exists:
                bot._detection_cannon_state[nexus.tag] = "complete"
            else:
                # Check if a cannon is still pending for THIS specific base
                pending_for_this_base = False
                building_tracker = bot.mediator.get_building_tracker_dict
                for worker_tag, info in building_tracker.items():
                    if info.get(ID) == UnitTypeId.PHOTONCANNON:
                        target = info.get(TARGET)
                        if target and cy_distance_to(target.position, mineral_center) < DETECTION_CANNON_RANGE:
                            pending_for_this_base = True
                            break
                if not pending_for_this_base:
                    # No cannon and nothing pending for this base — cancelled, retry
                    bot._detection_cannon_state[nexus.tag] = "needs_cannon"

        # --- State: complete ---
        elif state == "complete":
            # Verify cannon still exists; if destroyed, rebuild
            existing_cannons = bot.structures(UnitTypeId.PHOTONCANNON).closer_than(
                DETECTION_CANNON_RANGE, mineral_center
            )
            if not existing_cannons.exists:
                # Cannon destroyed — check if pylon still exists
                existing_pylons = bot.structures(UnitTypeId.PYLON).closer_than(
                    DETECTION_CANNON_RANGE, mineral_center
                )
                if existing_pylons.exists:
                    bot._detection_cannon_state[nexus.tag] = "needs_cannon"
                else:
                    bot._detection_cannon_state[nexus.tag] = "needs_pylon"

    # Render debug overlay (no-op when bot.debug is False)
    render_detection_cannon_debug(bot)


def _is_clear_of_minerals(pos: Point2, mineral_fields, clearance: float = MINERAL_CLEARANCE) -> bool:
    """Check if a position has enough clearance from mineral fields for probe pathing.

    Args:
        pos: Position to check
        mineral_fields: Units collection of mineral fields at this base
        clearance: Minimum distance from mineral field center (default MINERAL_CLEARANCE)

    Returns:
        True if position is far enough from all mineral fields for probe pathing.
    """
    if not mineral_fields:
        return True
    for mf in mineral_fields:
        if cy_distance_to(pos, mf.position) < clearance:
            return False
    return True


def _find_pylon_placement(bot, behind_mineral: list, nexus_pos: Point2, mineral_fields) -> Point2 | None:
    """Find a valid placement for a Pylon, trying behind-mineral-line first,
    then falling back to positions near the nexus. Ensures clearance from
    mineral fields for probe pathing.

    Args:
        bot: Bot instance for placement validation
        behind_mineral: 3 positions behind mineral line [left, center, right]
        nexus_pos: Nexus position for fallback search
        mineral_fields: Mineral fields at this base (for clearance checks)

    Returns:
        Valid placement Point2, or None if no position found
    """
    # Priority 1: Behind mineral line positions (with mineral clearance)
    for pos in behind_mineral:
        if bot.mediator.can_place_structure(
            position=pos, structure_type=UnitTypeId.PYLON
        ) and _is_clear_of_minerals(pos, mineral_fields):
            return pos

    # Priority 2: Between nexus and mineral center (offset toward minerals)
    mineral_center = behind_mineral[1]
    for offset in (3.0, 4.0, 5.0):
        pos = Point2(nexus_pos.towards(mineral_center, offset))
        if bot.mediator.can_place_structure(
            position=pos, structure_type=UnitTypeId.PYLON
        ) and _is_clear_of_minerals(pos, mineral_fields):
            return pos

    # Priority 3: Around the nexus at increasing radii
    for radius in (4.0, 5.0, 6.0):
        for angle_offset in range(0, 360, 45):
            angle = math.radians(angle_offset)
            pos = Point2((
                nexus_pos.x + radius * math.cos(angle),
                nexus_pos.y + radius * math.sin(angle),
            ))
            if bot.mediator.can_place_structure(
                position=pos, structure_type=UnitTypeId.PYLON
            ) and _is_clear_of_minerals(pos, mineral_fields):
                return pos

    return None


def _find_cannon_placement(
    bot, behind_mineral: list, pylon, mineral_center: Point2, nexus_pos: Point2, mineral_fields
) -> Point2 | None:
    """Find a valid placement for a Photon Cannon within pylon power range.
    Ensures clearance from mineral fields for probe pathing.

    Tries behind-mineral positions first, then positions near the pylon,
    then fallback positions around the nexus.

    Args:
        bot: Bot instance for placement validation
        behind_mineral: 3 positions behind mineral line [left, center, right]
        pylon: The powered Pylon that will supply the cannon
        mineral_center: Center position behind mineral line
        nexus_pos: Nexus position for fallback search
        mineral_fields: Mineral fields at this base (for clearance checks)

    Returns:
        Valid placement Point2, or None if no position found
    """
    # Priority 1: Behind mineral line within pylon power range (with mineral clearance)
    for pos in behind_mineral:
        dist_to_pylon = cy_distance_to(pos, pylon.position)
        if dist_to_pylon <= PYLON_POWER_RANGE:
            if bot.mediator.can_place_structure(
                position=pos, structure_type=UnitTypeId.PHOTONCANNON
            ) and _is_clear_of_minerals(pos, mineral_fields):
                return pos

    # Priority 2: Between mineral center and pylon
    for offset in (1.0, 2.0, 3.0):
        pos = Point2(mineral_center.towards(pylon.position, offset))
        dist_to_pylon = cy_distance_to(pos, pylon.position)
        if dist_to_pylon <= PYLON_POWER_RANGE:
            if bot.mediator.can_place_structure(
                position=pos, structure_type=UnitTypeId.PHOTONCANNON
            ) and _is_clear_of_minerals(pos, mineral_fields):
                return pos

    # Priority 3: Around the pylon at increasing radii (within power range)
    for radius in (2.0, 3.0, 4.0, 5.0):
        for angle_offset in range(0, 360, 45):
            angle = math.radians(angle_offset)
            pos = Point2((
                pylon.position.x + radius * math.cos(angle),
                pylon.position.y + radius * math.sin(angle),
            ))
            dist_to_pylon = cy_distance_to(pos, pylon.position)
            if dist_to_pylon <= PYLON_POWER_RANGE:
                if bot.mediator.can_place_structure(
                    position=pos, structure_type=UnitTypeId.PHOTONCANNON
                ) and _is_clear_of_minerals(pos, mineral_fields):
                    return pos

    # Priority 4: Around the nexus (cannon may be far from minerals but still useful)
    for radius in (4.0, 5.0, 6.0):
        for angle_offset in range(0, 360, 45):
            angle = math.radians(angle_offset)
            pos = Point2((
                nexus_pos.x + radius * math.cos(angle),
                nexus_pos.y + radius * math.sin(angle),
            ))
            # Check if within pylon power range
            dist_to_pylon = cy_distance_to(pos, pylon.position)
            if dist_to_pylon <= PYLON_POWER_RANGE:
                if bot.mediator.can_place_structure(
                    position=pos, structure_type=UnitTypeId.PHOTONCANNON
                ) and _is_clear_of_minerals(pos, mineral_fields):
                    return pos

    return None


async def handle_macro(
    bot,
    main_army: Units,
    warp_prism: Units,
    freeflow: bool,
) -> None:
    """
    Main macro logic: builds units, keeps supply up, reacts to cheese, 
    or transitions to late game. Call this in your bot's on_step.
    """
    spawn_location = bot.natural_expansion
    production_location = bot.start_location
    
    profile = get_active_profile(bot)
    worker_limit = _resolve(profile.worker_cap, bot)
    # Cheese reaction caps probe production until the reaction transitions
    # (keeps the build runner's ConstantWorkerProductionTill and post-build
    #  BuildWorkers in sync — previously BuildWorkers would jump straight
    #  to the standard profile cap, ignoring CategoryConfig.probe_cap)
    cheese_probe_cap = bot.reaction_manager.category_config.probe_cap
    if bot.reaction_manager.is_cheese_response and cheese_probe_cap is not None:
        worker_limit = min(worker_limit, cheese_probe_cap)
    optimal_worker_count = min(calculate_optimal_worker_count(bot), worker_limit)
    
    economy_state = get_economy_state(bot)
    
    # Select army composition based on reaction state
    # ReactionManager handles the one-way transition from cheese → standard
    if not bot.reaction_manager.is_cheese_response:
        army_composition = select_army_composition(bot)
        expansion_count = expansion_checker(bot, main_army)
    else:
        army_composition = CHEESE_DEFENSE_ARMY
        expansion_count = 3
    
    if require_shield_battery(bot):
        base_location = get_shield_battery_base_location(bot)
        bot.register_behavior(
            BuildStructure(
                base_location=base_location,
                structure_id=UnitTypeId.SHIELDBATTERY,
                static_defence=True
            )
        )
    
    # Scale gateways to desired count (from active BuildProfile)
    # Allow in reduced+ economy so production capacity ramps before the moderate transition,
    # preventing a burst of 2+ gateways when economy recovers
    if economy_state in ("reduced", "moderate", "full"):
        desired_gates = get_desired_gateway_count(bot)
        current_gates = (bot.structures(UnitTypeId.GATEWAY).amount + 
                         bot.structures(UnitTypeId.WARPGATE).amount + 
                         cy_structure_pending_ares(bot, UnitTypeId.GATEWAY))
        
        if current_gates < desired_gates and bot.can_afford(UnitTypeId.GATEWAY):
            bot.register_behavior(BuildStructure(production_location, UnitTypeId.GATEWAY))
    
    # Scale forges to desired count (0→1→2) - moderate+ only (lower priority than gates)
    if economy_state in ("moderate", "full"):
        desired_forges = get_desired_forge_count(bot)
        current_forges = (bot.structures(UnitTypeId.FORGE).amount + 
                         cy_structure_pending_ares(bot, UnitTypeId.FORGE))
        
        if current_forges < desired_forges and bot.can_afford(UnitTypeId.FORGE):
            bot.register_behavior(BuildStructure(production_location, UnitTypeId.FORGE))
    
    # Reactive structures from BuildProfile (e.g., Robo for detection in Stalker build)
    for structure_type, predicate in profile.conditional_structures:
        if predicate(bot) and bot.can_afford(structure_type):
            bot.register_behavior(BuildStructure(production_location, structure_type))
    
    # Detection cannons behind mineral lines (all profiles)
    # Sticky trigger: once a harass threat (Banshee/DT/Oracle/Widow Mine/Dark Shrine)
    # is seen, the system stays active for ALL bases (including new expansions)
    # until every base has a completed cannon. This prevents the harass unit from
    # simply flying to an unprotected base.
    if _needs_detection_cannons(bot):
        bot._detection_cannon_triggered = True
    if bot._detection_cannon_triggered:
        build_detection_cannons(bot)
    
    macro_plan: MacroPlan = MacroPlan()
    
    # Determine if we should bank minerals for an expansion this frame.
    #
    # Root cause of the original bug: ExpansionController was at the END of
    # the MacroPlan. MacroPlan.execute() returns True on the first successful
    # behavior, so BuildWorkers (50m), AutoSupply (100m), and GasBuildingController
    # (75m) would succeed and return True before ExpansionController ever ran.
    # The bot never accumulated the 400m for a Nexus — it hovered at 200-350m,
    # perpetually spending on probes/pylons/assimilators.
    #
    # Fix: when banking, add ExpansionController(prioritize=True) at the TOP
    # of the MacroPlan. prioritize=True makes it return True even when it
    # can't afford the Nexus yet, blocking all lower-priority spending.
    #
    # The banking condition stays active while a worker is en route to the
    # expansion (cy_structure_pending_ares > 0) but the Nexus isn't physically
    # under construction yet. This prevents SpawnController from spending
    # minerals before the building manager can place the Nexus.
    #
    # We use expansion_count (from expansion_checker) instead of counting
    # townhalls. expansion_checker already evaluates resource starvation,
    # base depletion, and spending efficiency to decide if we need more
    # bases. This works for the natural (2 bases) AND for late-game
    # expansions when existing bases are mining out.
    nexus_under_construction = bot.structures(UnitTypeId.NEXUS).not_ready.amount > 0
    wants_to_expand = expansion_count > len(bot.townhalls)
    # Defender's advantage: allow banking for the natural during an attack
    # if we have a shield battery to hold behind (see _can_expand_natural_under_attack)
    blocked_by_attack = bot._under_attack and not _can_expand_natural_under_attack(bot)
    banking_for_expansion = (
        economy_state == "reduced"
        and not blocked_by_attack
        and not nexus_under_construction
        and not bot.reaction_manager.is_cheese_response
        and wants_to_expand
    )
    
    if banking_for_expansion:
        # prioritize=True blocks all lower-priority spending until the
        # Nexus is placed. On frame 1, it sends a worker and returns True.
        # On subsequent frames, expansion_pending is True so the controller
        # returns False — but we still skip SpawnController below, so only
        # BuildWorkers (50m) and AutoSupply (100m) can spend. Minerals
        # accumulate and the building manager places the Nexus when affordable.
        macro_plan.add(ExpansionController(to_count=expansion_count, max_pending=1, prioritize=True))

    # Update expansion debug snapshot with banking/build state
    if hasattr(bot, '_expansion_debug'):
        bot._expansion_debug["banking"] = banking_for_expansion
        bot._expansion_debug["blocked_by_attack"] = blocked_by_attack
        bot._expansion_debug["economy_state"] = economy_state
        bot._expansion_debug["cheese_response"] = bot.reaction_manager.is_cheese_response
    
    # Always: workers and supply (all economy states)
    # These run after ExpansionController, so they only execute if
    # ExpansionController didn't block (i.e. we can afford the Nexus
    # or already have enough bases).
    macro_plan.add(BuildWorkers(to_count=optimal_worker_count))
    macro_plan.add(AutoSupply(base_location=production_location))
    
    if economy_state == "recovery":
        pass
    else:
        # Reduced+: gas buildings and spawn from existing production
        gas_target = _resolve(profile.gas_target, bot)
        # Demand-based gas: only build new assimilators when existing gas
        # capacity is fully saturated — that's the throughput manager's
        # (get_optimal_gas_workers) signal that we need more capacity.
        # When it throttles workers below 3, existing geysers are
        # underutilized and building more wastes 75m that could go to army.
        if get_optimal_gas_workers(bot) < 3:
            gas_target = min(gas_target, bot.gas_buildings.amount)
        macro_plan.add(GasBuildingController(to_count=gas_target, max_pending=2))
        
        spawn_target = warp_prism[0].position if warp_prism else spawn_location
        # Freeflow prevents SpawnController from breaking on unaffordable high-priority
        # units. Use it when: cheese defense active, get_freeflow_mode says yes,
        # or economy is not reduced (moderate/full can spend freely).
        spawn_freeflow = (
            bot.reaction_manager.is_cheese_response
            or freeflow
            or economy_state != "reduced"
        )
        
        # When banking for expansion, SpawnController is skipped entirely
        # so minerals accumulate for the Nexus. Otherwise, produce army.
        if banking_for_expansion:
            pass  # No SpawnController — minerals go to ExpansionController
        else:
            macro_plan.add(SpawnController(army_composition, spawn_target=spawn_target, freeflow_mode=spawn_freeflow))
        
        # Cheese response: ProductionController and UpgradeController both use
        # TechUp to auto-build missing tech, but they're gated to moderate/full
        # economy. Cheese keeps economy in reduced/recovery, so TechUp never
        # fires. This ensures tech buildings (e.g. Cyber Core for Stalkers)
        # get built even when the build runner was force-completed early.
        if bot.reaction_manager.is_cheese_response:
            for unit_type in army_composition:
                if not bot.tech_ready_for_unit(unit_type):
                    macro_plan.add(TechUp(unit_type, base_location=production_location))
        
        # Expansion logic: moderate+ gets full expansion, reduced gets safety net to 2 bases
        # Skip expansions when under attack (unless defender's advantage applies)
        # No expansions during cheese response — focus entirely on defense
        # Note: when banking_for_expansion, ExpansionController was already
        # added at the top with prioritize=True, so we skip adding it again here.
        if not banking_for_expansion and not blocked_by_attack and not bot.reaction_manager.is_cheese_response:
            if economy_state in ("moderate", "full"):
                macro_plan.add(ExpansionController(to_count=expansion_count, max_pending=1))
            elif wants_to_expand:
                # Reduced economy but expansion_checker says we need more bases.
                # This covers both the natural (early) and late-game expansions
                # when existing bases are mining out.
                macro_plan.add(ExpansionController(to_count=expansion_count, max_pending=1))
        
        if economy_state in ("moderate", "full"):
            # Moderate+: upgrades
            macro_plan.add(UpgradeController(get_desired_upgrades(bot), base_location=production_location))
            
            if economy_state == "full":
                # Full: add new production buildings
                macro_plan.add(ProductionController(army_composition, base_location=production_location, add_production_at_bank=(450,450), ignore_below_proportion=0.3))
    
    bot.register_behavior(macro_plan)
    _train_observers(bot)
    _train_warp_prism(bot)
