"""Combat debug visualization utilities.

Purpose: Centralized debug rendering for combat system, controlled by bot.debug flag
Key Decisions: All debug calls gated by bot.debug, minimal performance impact when disabled
Limitations: Requires bot.debug=True in bot.py to activate
"""

import math

import numpy as np
from sc2.position import Point2, Point3
from sc2.units import Units
from sc2.ids.unit_typeid import UnitTypeId
from ares.consts import UnitRole

from bot.intel import get_enemy_intel_quality, get_sim_static_defense
from bot.constants import FRESH_INTEL_THRESHOLD, STALE_INTEL_THRESHOLD, MEMORY_EXPIRY_TIME
from cython_extensions import cy_find_units_center_mass, cy_distance_to

# Worker types to filter from combat sim
WORKER_TYPES = {UnitTypeId.SCV, UnitTypeId.PROBE, UnitTypeId.DRONE, UnitTypeId.MULE}


def _get_unit_type_summary(units, max_types: int = 3) -> str:
    """Get a compact summary of unit types (e.g., '5Stlk 3Immo 2Colos').
    
    Args:
        units: List or Units collection to summarize
        max_types: Maximum number of unit types to show
        
    Returns:
        Compact string like '5Stlk 3Immo 2Colo'
    """
    from collections import Counter
    
    if not units:
        return "none"
    
    # Count unit types
    type_counts = Counter(u.type_id for u in units)
    
    # Get top N by count
    top_types = type_counts.most_common(max_types)
    
    # Abbreviate unit names (first 4 chars of name, capitalize)
    parts = []
    for unit_type, count in top_types:
        # Get short name from type_id
        name = unit_type.name[:4].title()
        parts.append(f"{count}{name}")
    
    return " ".join(parts)


def render_combat_state_overlay(bot, main_army: Units, enemy_threat_level: int, is_early_defensive_mode: bool) -> None:
    """
    Render on-screen debug text showing combat state, roles, and squad information.
    
    Only renders if bot.debug is True.
    
    Args:
        bot: Bot instance
        main_army: Main army units
        enemy_threat_level: Current enemy threat assessment
        is_early_defensive_mode: Whether in early defensive mode
    """
    if not bot.debug:
        return
    
    # --- Unified y-position layout (0.018 spacing) ---
    # All 2D debug lines use sequential y to avoid overlaps
    _y = 0.220
    _step = 0.018

    # Combat state overlay
    bot.client.debug_text_2d(
        f"Attack: {bot._commenced_attack} Threat: {enemy_threat_level} Under Attack: {bot._under_attack}", 
        Point2((0.1, _y)), None, 14
    )
    _y += _step
    bot.client.debug_text_2d(
        f"EarlyDefMode: {is_early_defensive_mode} Cheese: {bot.reaction_manager.is_cheese_response}", 
        Point2((0.1, _y)), None, 14
    )
    _y += _step

    # Mass Recall — always compute gates fresh each frame
    from bot.constants import MASS_RECALL_MIN_OWN_SUPPLY, MASS_RECALL_RETREAT_SEARCH_RADIUS, MASS_RECALL_ENERGY_COST
    from ares.consts import LOSS_DECISIVE_OR_WORSE
    from cython_extensions.general_utils import cy_in_pathing_grid_ma
    from cython_extensions.numpy_helper import cy_point_below_value
    from cython_extensions import cy_distance_to

    recall_cd = max(0.0, bot._mass_recall_last_cast_time + 130.0 - bot.time)
    recall_status = f"CD:{recall_cd:.0f}s" if recall_cd > 0 else "READY"

    # Compute gates fresh
    own_supply = sum(bot.calculate_supply_cost(u.type_id) for u in main_army) if main_army else 0
    army_center = main_army.center if main_army else None
    nearest_base = bot.townhalls.closest_to(army_center) if army_center and bot.townhalls else None
    base_position = nearest_base.position if nearest_base else None

    # Gate 1: fight result
    fight_result = None
    try:
        cached_enemy = bot.mediator.get_cached_enemy_army or []
        combat_enemies = [
            u for u in cached_enemy
            if u.type_id not in WORKER_TYPES and u.age < MEMORY_EXPIRY_TIME
        ]
        # Match the real gate: static D from enemy_structures (the cache never
        # holds structures — the `not u.is_structure` filter below was a no-op)
        combat_enemies += get_sim_static_defense(bot)
        fight_result = bot.mediator.can_win_fight(
            own_units=main_army, enemy_units=combat_enemies, workers_do_no_damage=True,
        )
    except Exception:
        pass

    # Gate 3: retreat blocked (escape zone with ground_grid — matches _is_retreat_blocked)
    retreat_blocked = False
    blocked_reason = ""
    if army_center and base_position:
        ground_grid = bot.mediator.get_ground_grid
        if not cy_point_below_value(ground_grid, army_center):
            army_dist = cy_distance_to(army_center, base_position)
            escape_distance = min(20.0, army_dist * 0.5)
            dx = base_position.x - army_center.x
            dy = base_position.y - army_center.y
            path_len = max(math.sqrt(dx * dx + dy * dy), 0.01)
            dx_norm = dx / path_len
            dy_norm = dy / path_len
            num_samples = 6
            blocked_count = 0
            for i in range(1, num_samples + 1):
                dist = escape_distance * i / num_samples
                sample = Point2((army_center.x + dx_norm * dist, army_center.y + dy_norm * dist))
                if not cy_in_pathing_grid_ma(ground_grid, sample) or not cy_point_below_value(ground_grid, sample):
                    blocked_count += 1
            if blocked_count >= num_samples // 2:
                retreat_blocked = True
                blocked_reason = f"esc:{blocked_count}/{num_samples}"
            else:
                safe_spot = bot.mediator.find_closest_safe_spot(
                    from_pos=army_center, grid=ground_grid, radius=MASS_RECALL_RETREAT_SEARCH_RADIUS,
                )
                if safe_spot is None:
                    retreat_blocked = True
                    blocked_reason = "no_safe"
                elif cy_distance_to(safe_spot, base_position) >= army_dist:
                    retreat_blocked = True
                    blocked_reason = "safe_away"

    g1 = "Y" if fight_result in LOSS_DECISIVE_OR_WORSE else "N"
    g2 = "Y" if own_supply >= MASS_RECALL_MIN_OWN_SUPPLY else "N"
    g3 = "Y" if retreat_blocked else "N"
    fr_name = fight_result.name if fight_result and hasattr(fight_result, 'name') else "N/A"
    blk_detail = f"({blocked_reason})" if blocked_reason else ""
    # Show trapped squad size from actual per-squad check (stored by try_mass_recall)
    gates = getattr(bot, '_mass_recall_gates', {})
    sq_size = gates.get("trapped_squad_size", 0)
    sq_info = f" sq:{sq_size}" if sq_size else ""
    gate_str = f"loss{g1} supply{g2}({own_supply:.0f}) blocked{g3}{blk_detail}{sq_info}"

    # Nexus energy check — shows why recall didn't cast even when gates pass
    nexus_energy_info = ""
    ready_nexuses = list(bot.structures(UnitTypeId.NEXUS).ready)
    if ready_nexuses:
        max_energy = max(n.energy for n in ready_nexuses)
        nexus_energy_info = f" NxMaxE:{max_energy:.0f}/{MASS_RECALL_ENERGY_COST}"
    else:
        nexus_energy_info = " NoNexus"

    bot.client.debug_text_2d(
        f"Recall: {recall_status} Pend:{bot._mass_recall_pending} | {gate_str} [{fr_name}]{nexus_energy_info}",
        Point2((0.1, _y)), None, 12
    )
    _y += _step

    # Role counts
    attacking_units = bot.mediator.get_units_from_role(role=UnitRole.ATTACKING)
    defending_units = bot.mediator.get_units_from_role(role=UnitRole.DEFENDING) 
    base_defender_units = bot.mediator.get_units_from_role(role=UnitRole.BASE_DEFENDER)
    
    bot.client.debug_text_2d(
        f"ROLES: ATK:{len(attacking_units)} DEF:{len(defending_units)} BASE:{len(base_defender_units)}", 
        Point2((0.1, _y)), None, 12
    )
    _y += _step

    # Squad counts (use constants for squad radii)
    from bot.constants import ATTACKING_SQUAD_RADIUS, DEFENDER_SQUAD_RADIUS
    attacking_squads = bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=ATTACKING_SQUAD_RADIUS)
    defending_squads = bot.mediator.get_squads(role=UnitRole.DEFENDING, squad_radius=ATTACKING_SQUAD_RADIUS)
    base_defender_squads = bot.mediator.get_squads(role=UnitRole.BASE_DEFENDER, squad_radius=DEFENDER_SQUAD_RADIUS)
    
    bot.client.debug_text_2d(
        f"SQUADS: ATK:{len(attacking_squads)} DEF:{len(defending_squads)} BASE:{len(base_defender_squads)}", 
        Point2((0.1, _y)), None, 12
    )
    _y += _step

    # Pass current y to sub-renderers so they continue from here
    bot._debug_y = _y
    
    # Composition overlay: shows the profile base vs the selected composition
    _render_composition_overlay(bot)
    
    # Combat simulator results
    _render_combat_sim_overlay(bot, main_army)
    
    # Visual markers for targeting
    render_target_markers(bot, main_army)


# Short labels for unit types in the composition overlay
_UNIT_SHORT_NAMES: dict[UnitTypeId, str] = {
    UnitTypeId.ZEALOT: "ZEA",
    UnitTypeId.STALKER: "STK",
    UnitTypeId.IMMORTAL: "IMM",
    UnitTypeId.COLOSSUS: "COL",
    UnitTypeId.DISRUPTOR: "DIS",
    UnitTypeId.HIGHTEMPLAR: "HT",
    UnitTypeId.ARCHON: "ARC",
    UnitTypeId.ADEPT: "ADP",
}


def _render_composition_overlay(bot) -> None:
    """Render the composition overlay: profile base vs the scored selection."""
    base = getattr(bot, '_last_base_comp', {})
    selected = getattr(bot, '_last_selected_comp', {})
    if not base:
        return

    _y = min(getattr(bot, '_debug_y', 0.30), 0.95)
    _step = 0.018

    # Show proportions with deltas, sorted by final priority
    sorted_types = sorted(
        selected.keys() if selected else base.keys(),
        key=lambda ut: (selected or base).get(ut, {}).get("priority", 99),
    )
    parts = []
    for unit_type in sorted_types:
        name = _UNIT_SHORT_NAMES.get(unit_type, unit_type.name[:3])
        base_pct = base.get(unit_type, {}).get("proportion", 0)
        base_pri = base.get(unit_type, {}).get("priority", "?")
        chosen_info = selected.get(unit_type, base.get(unit_type, {}))
        chosen_pct = chosen_info.get("proportion", base_pct) if selected else base_pct
        chosen_pri = chosen_info.get("priority", base_pri)
        delta = chosen_pct - base_pct
        # Show priority change (e.g. [2->1]) or just current (e.g. [1])
        pri_str = f"{base_pri}->{chosen_pri}" if selected and chosen_pri != base_pri else f"{chosen_pri}"
        if abs(delta) > 0.005:
            arrow = "+" if delta > 0 else ""
            parts.append(f"{name}[{pri_str}] {base_pct:.0%}->{chosen_pct:.0%}({arrow}{delta:.0%})")
        else:
            parts.append(f"{name}[{pri_str}] {chosen_pct:.0%}")
    
    # Resource pressure tag
    pressure = getattr(bot, '_resource_pressure', 'BALANCED')
    pressure_tag = f" [{pressure}]" if pressure != "BALANCED" else ""
    
    label = f"Comp{pressure_tag}: " + " ".join(parts)
    bot.client.debug_text_2d(label, Point2((0.1, _y)), None, 12)
    bot._debug_y = _y + _step


def _render_combat_sim_overlay(bot, main_army: Units) -> None:
    """Render combat simulator results (global and squad-level)."""
    _y = min(getattr(bot, '_debug_y', 0.30), 0.95)
    _step = 0.018

    # Get enemy army (filter workers and expired ghosts) - use cached enemy
    cached_enemy = bot.mediator.get_cached_enemy_army or []
    combat_enemies = [
        u for u in cached_enemy
        if u.type_id not in WORKER_TYPES and u.age < MEMORY_EXPIRY_TIME
    ]
    # Match the real gate: static D from enemy_structures (cache never holds
    # structures — the old `not u.is_structure` filter was a no-op)
    combat_enemies += get_sim_static_defense(bot)
    
    # Scout status (any unit with SCOUTING role)
    scout_tags = bot.mediator.get_unit_role_dict.get(UnitRole.SCOUTING, set())
    valid_tags = {u.tag for u in bot.all_own_units}
    live_scout_tags = scout_tags & valid_tags
    
    if live_scout_tags:
        scouts = bot.all_own_units.tags_in(live_scout_tags)
        scout = scouts.first
        scout_type = scout.type_id.name
        scout_hp = f"{int(scout.health_percentage * 100)}%"
        bot.client.debug_text_2d(
            f"Scout: {scout_type} #{scout.tag} HP:{scout_hp} ({len(scouts)} total)", 
            Point2((0.1, _y)), None, 12
        )
    elif bot._worker_scout_sent_this_stale_period:
        bot.client.debug_text_2d(
            f"Scout: Dead/Missing (flag set)", 
            Point2((0.1, _y)), None, 12
        )
    _y += _step

    # Intel quality display — uses composition belief freshness when enabled
    use_belief = bot.config.get("Belief", {}).get("enable_composition", False)
    if use_belief:
        comp = bot.belief_state.composition
        freshness = comp.freshness
        weighted = comp.get_weighted_army(
            exclude_workers=True, exclude_structures=True, exclude_ignored=True,
            min_confidence=0.01,
        )
        visible_count = sum(1 for wu in weighted if wu.confidence >= 0.9)
        memory_count = len(weighted) - visible_count
        total_weighted = comp.total_weighted_count
        expected_types = list(comp.get_expected_unit_types().keys())
        exp_str = f" Exp:[{','.join(t.name[:4] for t in expected_types[:4])}]" if expected_types else ""
        freshness_bar = "\u2588" * int(freshness * 10) + "\u2591" * (10 - int(freshness * 10))
        intel_status = "FRESH" if freshness >= FRESH_INTEL_THRESHOLD else ("STALE" if freshness >= STALE_INTEL_THRESHOLD else "BLIND")
        bot.client.debug_text_2d(
            f"Intel: [{freshness_bar}] {intel_status} ({visible_count}vis/{memory_count}mem W:{total_weighted:.1f}{exp_str})",
            Point2((0.1, _y)), None, 12
        )
    else:
        intel = get_enemy_intel_quality(bot)
        freshness_bar = "\u2588" * int(intel["freshness"] * 10) + "\u2591" * (10 - int(intel["freshness"] * 10))
        intel_status = "FRESH" if intel["freshness"] >= FRESH_INTEL_THRESHOLD else ("STALE" if intel["freshness"] >= STALE_INTEL_THRESHOLD else "BLIND")
        expired = intel.get('expired_count', 0)
        exp_str = f" +{expired}exp" if expired else ""
        bot.client.debug_text_2d(
            f"Intel: [{freshness_bar}] {intel_status} ({intel['visible_count']}vis/{intel['memory_count']}mem{exp_str})",
            Point2((0.1, _y)), None, 12
        )
    _y += _step

    # Intel urgency indicator (shows response thresholds)
    urgency = getattr(bot, '_intel_urgency', 0.0)
    urgency_bar = "\u2588" * int(urgency * 10) + "\u2591" * (10 - int(urgency * 10))
    urgency_status = ""
    if urgency >= 0.7:
        urgency_status = " \u2192UPGRADE"
    elif urgency >= 0.5:
        urgency_status = " \u2192+OBS"
    elif urgency >= 0.3:
        urgency_status = " \u2192PRIORITY"
    bot.client.debug_text_2d(
        f"Urgency: [{urgency_bar}]{urgency_status}", 
        Point2((0.1, _y)), None, 12
    )
    _y += _step

    # Strategy belief display (when enabled)
    if (bot.config.get("Belief", {}).get("enable_strategy", True)
            and bot.belief_state.strategy is not None):
        pred = bot.belief_state.strategy.last_prediction
        if pred is not None:
            strat_bar = f"{pred.label.value}({pred.source})"
            strat_probs = f"C:{pred.p_cheese:.0%} A:{pred.p_all_in:.0%} T:{pred.p_timing:.0%} M:{pred.p_macro:.0%}"
            bot.client.debug_text_2d(
                f"Strat: {strat_bar} {strat_probs}",
                Point2((0.1, _y)), None, 12
            )
            _y += _step
            if pred.evidence:
                ev = pred.evidence
                ev_str = f"pool={ev.get('pool_bin','?')} bases={ev.get('bases_bin','?')} rax={ev.get('rax_bin','?')}"
                bot.client.debug_text_2d(
                    f"  {ev_str}",
                    Point2((0.1, _y)), None, 10
                )
                _y += _step

    # Scout VOI staleness display (when enabled)
    if bot.config.get("Belief", {}).get("enable_scout_voi", False):
        location_last_seen = getattr(bot, "_location_last_seen", {})
        if location_last_seen:
            game_time = bot.time
            staleness_parts = []
            for key in sorted(location_last_seen.keys()):
                staleness = game_time - location_last_seen[key]
                staleness_parts.append(f"{key}:{staleness:.0f}s")
            voi_str = " ".join(staleness_parts[:5])
            bot.client.debug_text_2d(
                f"VOI: {voi_str}",
                Point2((0.1, _y)), None, 10
            )
            _y += _step

    # Global fight result
    try:
        global_result = bot.mediator.can_win_fight(
            own_units=main_army,
            enemy_units=combat_enemies,
            workers_do_no_damage=True,
        )
        bot.client.debug_text_2d(
            f"Global Fight: {global_result.name}", 
            Point2((0.1, _y)), None, 14
        )
    except Exception:
        bot.client.debug_text_2d(
            "Global Fight: N/A", 
            Point2((0.1, _y)), None, 14
        )
    _y += _step

    # Army compositions
    own_types = _get_unit_type_summary(main_army or [])
    enemy_types = _get_unit_type_summary(combat_enemies)
    
    bot.client.debug_text_2d(
        f"Own: {len(main_army) if main_army else 0}u [{own_types}]", 
        Point2((0.1, _y)), None, 12
    )
    _y += _step
    bot.client.debug_text_2d(
        f"Enemy: {len(combat_enemies)}u [{enemy_types}]", 
        Point2((0.1, _y)), None, 12
    )
    _y += _step

    # Squad engagement tracker summary
    tracker = getattr(bot, '_squad_engagement_tracker', {})
    if tracker:
        engaged = sum(1 for v in tracker.values() if v.get("can_engage", False))
        total = len(tracker)
        bot.client.debug_text_2d(
            f"Squad Engage: {engaged}/{total} squads", 
            Point2((0.1, _y)), None, 14
        )
    _y += _step

    # 3D squad labels at squad positions
    _render_squad_labels(bot, tracker)

    # Pass y to next renderer
    bot._debug_y = _y


def _render_squad_labels(bot, tracker: dict) -> None:
    """Render 3D labels at each attacking squad's position showing engagement status."""
    from bot.constants import ATTACKING_SQUAD_RADIUS
    
    attacking_squads = bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=ATTACKING_SQUAD_RADIUS)
    
    for squad in attacking_squads:
        squad_id = squad.squad_id
        can_engage = tracker.get(squad_id, {}).get("can_engage", True)
        
        # Color: green if engaging, red if retreating
        color = (0, 255, 0) if can_engage else (255, 0, 0)
        status = "ATK" if can_engage else "RTR"
        label = f"S{squad_id[-4:]}:{status}"  # Short squad ID + status
        
        # Draw at squad center
        pos = squad.squad_position
        z = bot.get_terrain_z_height(pos)
        bot.client.debug_text_world(
            label,
            Point3((pos.x, pos.y, z + 1.5)),
            color=color,
            size=12,
        )


def render_target_markers(bot, main_army: Units) -> None:
    """
    Render 3D debug spheres showing current attack target and army center.
    
    Only renders if bot.debug is True.
    
    Args:
        bot: Bot instance
        main_army: Main army units
    """
    if not bot.debug:
        return
    
    _y = min(getattr(bot, '_debug_y', 0.40), 0.95)
    _step = 0.018

    # Current attack target marker (red sphere)
    if hasattr(bot, 'current_attack_target') and bot.current_attack_target:
        bot.client.debug_text_2d(
            f"Current Target: {bot.current_attack_target}", 
            Point2((0.1, _y)), None, 14
        )
        _y += _step
        target_3d = Point3((
            bot.current_attack_target.x, 
            bot.current_attack_target.y, 
            bot.get_terrain_z_height(bot.current_attack_target)
        ))
        bot.client.debug_sphere_out(target_3d, 2, Point3((255, 0, 0)))
    
    # Army center marker (green sphere)
    if main_army:
        army_center = main_army.center
        bot.client.debug_text_2d(
            f"Army Center: {army_center}", 
            Point2((0.1, _y)), None, 14
        )
        _y += _step
        army_center_3d = Point3((
            army_center.x, 
            army_center.y, 
            bot.get_terrain_z_height(army_center)
        ))
        bot.client.debug_sphere_out(army_center_3d, 3, Point3((0, 255, 0)))

    bot._debug_y = _y


def render_expansion_debug(bot) -> None:
    """Render expansion decision debug overlay.

    Reads bot._expansion_debug (set by expansion_checker + handle_macro in macro.py).
    Shows: current vs target base count, map control gate, which trigger fired,
    banking state, and economy state. Lets you verify the expansion logic is
    working correctly during replays.
    """
    if not bot.debug:
        return

    info = getattr(bot, '_expansion_debug', None)
    if info is None:
        return

    _y = min(getattr(bot, '_debug_y', 0.52), 0.95)
    _step = 0.018

    # Line 1: base counts + map control + trigger
    wants = info.get("wants_expand", False)
    mc = info.get("has_map_control", False)
    trigger = info.get("triggered_by", "none")
    status_color = (0, 255, 0) if wants and mc else ((255, 165, 0) if not mc else (255, 255, 255))
    bot.client.debug_text_2d(
        f"Expand: {info.get('current_bases', 0)}->{info.get('target_bases', 0)} "
        f"MC:{'Y' if mc else 'N'} Trig:{trigger}",
        Point2((0.1, _y)), status_color, 12
    )
    _y += _step

    # Line 2: banking + blockers + economy
    banking = info.get("banking", False)
    blocked = info.get("blocked_by_attack", False)
    econ = info.get("economy_state", "?")
    cheese = info.get("cheese_response", False)
    flags = []
    if banking:
        flags.append("BANKING")
    if blocked:
        flags.append("ATK_BLOCKED")
    if cheese:
        flags.append("CHEESE")
    flag_str = f" [{','.join(flags)}]" if flags else ""
    bot.client.debug_text_2d(
        f"  Econ:{econ}{flag_str}",
        Point2((0.1, _y)), None, 12
    )
    _y += _step

    # Line 3: economy metrics (saturation, spending efficiency)
    sat = info.get("worker_saturation", 0)
    eff = info.get("spending_efficiency", 0)
    eff_thresh = info.get("efficiency_threshold", 0)
    bot.client.debug_text_2d(
        f"  Sat:{sat:.0%} SpendEff:{eff:.1f}/{eff_thresh:.1f} "
        f"Starved:{'Y' if info.get('resource_starved') else 'N'} "
        f"Depl:{'Y' if info.get('bases_depleting') else 'N'}",
        Point2((0.1, _y)), None, 11
    )
    _y += _step

    bot._debug_y = _y


def render_mass_recall_debug(bot, main_army: Units) -> None:
    """Render 3D Mass Recall debug: escape zone samples, trapped/influenced units, recall target.

    Reads per-frame data stored by _is_retreat_blocked and _find_recall_target in combat.py.
    Shows escape zone sample points (green=safe, red=blocked), unit influence state,
    and recall target with radius circle when all gates pass.
    """
    if not bot.debug or not main_army:
        return

    from bot.constants import MASS_RECALL_RADIUS

    # --- Escape zone samples (stored by _is_retreat_blocked) ---
    escape_samples = getattr(bot, '_recall_escape_samples', [])
    for sample_pos, is_blocked in escape_samples:
        z = bot.get_terrain_z_height(sample_pos)
        color = Point3((255, 50, 50)) if is_blocked else Point3((50, 255, 50))
        bot.client.debug_sphere_out(
            Point3((sample_pos.x, sample_pos.y, z + 0.4)), 0.4, color
        )

    # --- Trapped / influenced unit labels (stored by _find_recall_target) ---
    trapped_tags = getattr(bot, '_recall_trapped_tags', set())
    influenced_tags = getattr(bot, '_recall_influenced_tags', set())
    if trapped_tags or influenced_tags:
        for u in main_army:
            if u.tag in trapped_tags:
                z = bot.get_terrain_z_height(u.position)
                bot.client.debug_text_world(
                    "TRAPPED", Point3((u.position.x, u.position.y, z + 1.2)),
                    color=(255, 50, 50), size=10,
                )
            elif u.tag in influenced_tags:
                z = bot.get_terrain_z_height(u.position)
                bot.client.debug_text_world(
                    "INFL", Point3((u.position.x, u.position.y, z + 1.2)),
                    color=(255, 165, 0), size=10,
                )

    # --- Recall target marker + radius circle (only when gates passed) ---
    recall_target = getattr(bot, '_recall_target_pos', None)
    if recall_target is not None:
        z = bot.get_terrain_z_height(recall_target)
        # Purple sphere at target center
        bot.client.debug_sphere_out(
            Point3((recall_target.x, recall_target.y, z + 0.5)), 1.0,
            Point3((200, 0, 255)),
        )
        bot.client.debug_text_world(
            "RECALL TARGET", Point3((recall_target.x, recall_target.y, z + 2.0)),
            color=(200, 0, 255), size=14,
        )
        # Draw recall radius circle on ground
        num_circle = 24
        for i in range(num_circle):
            a1 = 2 * math.pi * i / num_circle
            a2 = 2 * math.pi * (i + 1) / num_circle
            p1 = Point2((recall_target.x + MASS_RECALL_RADIUS * math.cos(a1),
                         recall_target.y + MASS_RECALL_RADIUS * math.sin(a1)))
            p2 = Point2((recall_target.x + MASS_RECALL_RADIUS * math.cos(a2),
                         recall_target.y + MASS_RECALL_RADIUS * math.sin(a2)))
            z1 = bot.get_terrain_z_height(p1)
            z2 = bot.get_terrain_z_height(p2)
            bot.client.debug_line_out(
                Point3((p1.x, p1.y, z1 + 0.2)),
                Point3((p2.x, p2.y, z2 + 0.2)),
                color=Point3((200, 0, 255)),
            )


def render_disruptor_labels(bot) -> None:
    """Render 3D labels on Disruptors showing ability status and target info."""
    if not bot.debug:
        return
    
    from sc2.ids.ability_id import AbilityId
    
    disruptors = bot.units(UnitTypeId.DISRUPTOR)
    
    for disruptor in disruptors:
        # Check if Nova ability is ready
        can_nova = AbilityId.EFFECT_PURIFICATIONNOVA in disruptor.abilities
        
        # Build label
        if can_nova:
            label = "NOVA READY"
            color = (0, 255, 0)  # Green
        else:
            label = "CD"
            color = (128, 128, 128)  # Gray
        
        # Draw at unit position
        pos = disruptor.position
        z = bot.get_terrain_z_height(pos)
        bot.client.debug_text_world(
            label,
            Point3((pos.x, pos.y, z + 2.0)),
            color=color,
            size=12,
        )


def render_nova_labels(bot, nova_manager) -> None:
    """Render 3D labels on active Novas showing frames left and target."""
    if not bot.debug or nova_manager is None:
        return
    
    for nova in nova_manager.active_novas:
        if nova.unit is None:
            continue
        
        # Find the nova unit in current game state
        nova_unit = None
        for unit in bot.units:
            if unit.tag == nova.unit.tag:
                nova_unit = unit
                break
        
        if nova_unit is None:
            continue
        
        # Build label with frames left and target
        frames = nova.frames_left
        target = nova.best_target_pos
        
        if target:
            dist = cy_distance_to(nova_unit.position, target)
            label = f"F:{frames} D:{dist:.1f}"
        else:
            label = f"F:{frames}"
        
        # Color based on frames left (green->yellow->red)
        if frames > 30:
            color = (0, 255, 0)
        elif frames > 15:
            color = (255, 255, 0)
        else:
            color = (255, 0, 0)
        
        pos = nova_unit.position
        z = bot.get_terrain_z_height(pos)
        bot.client.debug_text_world(
            label,
            Point3((pos.x, pos.y, z + 1.0)),
            color=color,
            size=10,
        )
        
        # Draw line to target
        if target:
            target_z = bot.get_terrain_z_height(target)
            bot.client.debug_line_out(
                Point3((pos.x, pos.y, z)),
                Point3((target.x, target.y, target_z)),
                color=Point3((255, 165, 0)),  # Orange line to target
            )


def render_formation_debug(bot, squad_units: list, target) -> None:
    """
    Render formation/cohesion debug visualization.
    
    Shows:
    - Army center of mass (blue sphere)
    - Cohesion radius (blue circle)
    - Units that are ahead of formation (yellow labels + lines)
    - Units with formation adjustments (cyan labels + lines to adjusted pos)
    """
    if not bot.debug or not squad_units or len(squad_units) < 2:
        return
    
    from bot.combat.combat import (
        get_formation_move_target, 
        FORMATION_AHEAD_BASE, FORMATION_AHEAD_SCALE,
        FORMATION_SPREAD_BASE, FORMATION_SPREAD_SCALE,
        MELEE_RANGE_THRESHOLD
    )
    from sc2.position import Point2
    
    # Compute dynamic thresholds (must match get_formation_move_target)
    n_sqrt = math.sqrt(len(squad_units))
    ahead_threshold = FORMATION_AHEAD_BASE + n_sqrt * FORMATION_AHEAD_SCALE
    spread_threshold = FORMATION_SPREAD_BASE + n_sqrt * FORMATION_SPREAD_SCALE
    
    # Get army center of mass
    army_center, _ = cy_find_units_center_mass(squad_units, 10.0)
    army_center = Point2(army_center)
    center_z = bot.get_terrain_z_height(army_center)
    
    # Draw army center (blue sphere)
    bot.client.debug_sphere_out(
        Point3((army_center.x, army_center.y, center_z + 0.5)),
        1.5,
        Point3((0, 100, 255))  # Blue
    )
    
    # Draw cohesion spread radius circle (outer - orange)
    num_points = 24
    for i in range(num_points):
        angle1 = 2 * math.pi * i / num_points
        angle2 = 2 * math.pi * (i + 1) / num_points
        p1 = Point2((army_center.x + spread_threshold * math.cos(angle1),
                     army_center.y + spread_threshold * math.sin(angle1)))
        p2 = Point2((army_center.x + spread_threshold * math.cos(angle2),
                     army_center.y + spread_threshold * math.sin(angle2)))
        z1 = bot.get_terrain_z_height(p1)
        z2 = bot.get_terrain_z_height(p2)
        bot.client.debug_line_out(
            Point3((p1.x, p1.y, z1 + 0.2)),
            Point3((p2.x, p2.y, z2 + 0.2)),
            color=Point3((255, 165, 0))  # Orange circle for spread threshold
        )
    
    # Calculate distances
    center_dist_to_target = cy_distance_to(army_center, target)
    
    # Check each unit's formation status
    ahead_count = 0
    spread_count = 0
    adjusted_count = 0
    
    for unit in squad_units:
        unit_dist_to_target = cy_distance_to(unit.position, target)
        ahead_distance = center_dist_to_target - unit_dist_to_target
        dist_from_center = cy_distance_to(unit.position, army_center)
        
        pos = unit.position
        z = bot.get_terrain_z_height(pos)
        
        # Get this unit's formation target
        formation_target = get_formation_move_target(unit, squad_units, target)
        
        is_melee = unit.ground_range <= MELEE_RANGE_THRESHOLD
        
        # Unit is streaming ahead (cohesion triggered) - will HOLD POSITION
        if ahead_distance > ahead_threshold:
            ahead_count += 1
            # Yellow label for units holding/waiting
            label = f"HOLD +{ahead_distance:.1f}"
            bot.client.debug_text_world(
                label,
                Point3((pos.x, pos.y, z + 1.0)),
                color=(255, 255, 0),  # Yellow
                size=10,
            )
            # Draw line to where they're going (army center area)
            ft_z = bot.get_terrain_z_height(formation_target)
            bot.client.debug_line_out(
                Point3((pos.x, pos.y, z + 0.3)),
                Point3((formation_target.x, formation_target.y, ft_z + 0.3)),
                color=Point3((255, 255, 0))  # Yellow
            )
        # Unit is too spread out from center
        elif dist_from_center > spread_threshold:
            spread_count += 1
            # Orange label for spread units
            label = f"SPREAD {dist_from_center:.1f}"
            bot.client.debug_text_world(
                label,
                Point3((pos.x, pos.y, z + 1.0)),
                color=(255, 165, 0),  # Orange
                size=10,
            )
            # Draw line to formation target
            ft_z = bot.get_terrain_z_height(formation_target)
            bot.client.debug_line_out(
                Point3((pos.x, pos.y, z + 0.3)),
                Point3((formation_target.x, formation_target.y, ft_z + 0.3)),
                color=Point3((255, 165, 0))  # Orange
            )
        # Unit got formation adjustment (ranged behind melee)
        elif formation_target != target:
            adjusted_count += 1
            # Cyan label for formation-adjusted units
            label = "FORM" if not is_melee else "FRONT"
            bot.client.debug_text_world(
                label,
                Point3((pos.x, pos.y, z + 1.0)),
                color=(0, 255, 255),  # Cyan
                size=10,
            )
            # Draw line to adjusted position
            ft_z = bot.get_terrain_z_height(formation_target)
            bot.client.debug_line_out(
                Point3((pos.x, pos.y, z + 0.3)),
                Point3((formation_target.x, formation_target.y, ft_z + 0.3)),
                color=Point3((0, 255, 255))  # Cyan
            )
    
    # Summary text
    _y = min(getattr(bot, '_debug_y', 0.44), 0.95)
    bot.client.debug_text_2d(
        f"Formation: {ahead_count} ahead, {spread_count} spread, {adjusted_count} form",
        Point2((0.1, _y)), None, 14
    )
    bot._debug_y = _y + 0.018


def render_base_defender_debug(bot) -> None:
    """
    Render debug visualization for BASE_DEFENDER units.
    
    Shows:
    - Per-unit detection range circles (yellow)
    - Detected enemies count
    - Threat position marker (red sphere)
    """
    if not bot.debug:
        return
    
    from bot.constants import UNIT_ENEMY_DETECTION_RANGE, COMMON_UNIT_IGNORE_TYPES
    from ares.consts import UnitTreeQueryType
    
    # Get BASE_DEFENDER units
    defender_units = bot.mediator.get_units_from_role(role=UnitRole.BASE_DEFENDER)
    
    # Draw threat position (red sphere)
    threat_pos = getattr(bot, '_defender_threat_position', None)
    if threat_pos:
        z = bot.get_terrain_z_height(threat_pos)
        bot.client.debug_sphere_out(
            Point3((threat_pos.x, threat_pos.y, z + 0.5)),
            1.5,
            Point3((255, 0, 0))  # Red
        )
        bot.client.debug_text_world(
            "THREAT",
            Point3((threat_pos.x, threat_pos.y, z + 2.0)),
            color=(255, 0, 0),
            size=12,
        )
    
    if not defender_units:
        return
    
    # Per-unit detection (matches control_base_defenders logic)
    unit_positions = [u.position for u in defender_units]
    all_close_results = bot.mediator.get_units_in_range(
        start_points=unit_positions,
        distances=UNIT_ENEMY_DETECTION_RANGE,
        query_tree=UnitTreeQueryType.AllEnemy,
        return_as_dict=False,
    )
    
    # Combine and deduplicate enemies
    seen_tags = set()
    all_close_list = []
    for result in all_close_results:
        for u in result:
            if u.tag not in seen_tags and not u.is_memory and not u.is_structure and u.type_id not in COMMON_UNIT_IGNORE_TYPES:
                seen_tags.add(u.tag)
                all_close_list.append(u)
    
    # Draw detection circle around each defender unit
    num_points = 16
    for unit in defender_units:
        unit_pos = unit.position
        z = bot.get_terrain_z_height(unit_pos)
        
        # Detection range circle (yellow)
        for i in range(num_points):
            angle1 = 2 * math.pi * i / num_points
            angle2 = 2 * math.pi * (i + 1) / num_points
            p1 = Point2((
                unit_pos.x + UNIT_ENEMY_DETECTION_RANGE * math.cos(angle1),
                unit_pos.y + UNIT_ENEMY_DETECTION_RANGE * math.sin(angle1)
            ))
            p2 = Point2((
                unit_pos.x + UNIT_ENEMY_DETECTION_RANGE * math.cos(angle2),
                unit_pos.y + UNIT_ENEMY_DETECTION_RANGE * math.sin(angle2)
            ))
            z1 = bot.get_terrain_z_height(p1)
            z2 = bot.get_terrain_z_height(p2)
            bot.client.debug_line_out(
                Point3((p1.x, p1.y, z1 + 0.2)),
                Point3((p2.x, p2.y, z2 + 0.2)),
                color=Point3((255, 255, 0))  # Yellow
            )
    
    # Summary label at center of defenders
    center = defender_units.center
    z = bot.get_terrain_z_height(center)
    enemy_count = len(all_close_list)
    color = (0, 255, 0) if enemy_count > 0 else (255, 255, 0)
    label = f"DEF:{len(defender_units)}u E:{enemy_count}"
    bot.client.debug_text_world(
        label,
        Point3((center.x, center.y, z + 2.0)),
        color=color,
        size=12,
    )
    
    # Draw lines from defender center to detected enemies (green)
    for enemy in all_close_list:
        enemy_z = bot.get_terrain_z_height(enemy.position)
        bot.client.debug_line_out(
            Point3((center.x, center.y, z + 0.5)),
            Point3((enemy.position.x, enemy.position.y, enemy_z + 0.5)),
            color=Point3((0, 255, 0))  # Green
        )


def render_target_scoring_debug(bot, main_army: Units) -> None:
    """Draw lines from each combat unit to its scored target, with score labels.
    
    Re-runs select_target for display purposes. Only renders when debug is on.
    Perf note: skips units with no nearby enemies. Samples at most 20 units to limit draw calls.
    """
    if not bot.debug or not main_army:
        return
    
    from bot.combat.scoring_table import score_target, select_target
    from cython_extensions import cy_closer_than
    
    # Get all visible enemy units (non-structure, non-memory)
    all_enemies = [u for u in (bot.enemy_units or []) if not u.is_structure and not u.is_memory]
    if not all_enemies:
        return
    
    # Sample units to avoid excessive draw calls in large armies
    units = list(main_army)[:20]
    
    for unit in units:
        # Pre-filter enemies within 15 range (matches combat detection range)
        nearby = cy_closer_than(all_enemies, 15.0, unit.position)
        if not nearby:
            continue
        
        target = select_target(unit, nearby)
        best_score = score_target(unit, target)
        
        # Draw line from unit to its chosen target
        u_pos = unit.position
        t_pos = target.position
        u_z = bot.get_terrain_z_height(u_pos)
        t_z = bot.get_terrain_z_height(t_pos)
        
        # Color by unit type: cyan for ranged, yellow for melee
        is_melee = unit.ground_range <= 2.0
        color = Point3((255, 255, 0)) if is_melee else Point3((0, 200, 255))
        
        bot.client.debug_line_out(
            Point3((u_pos.x, u_pos.y, u_z + 0.5)),
            Point3((t_pos.x, t_pos.y, t_z + 0.5)),
            color=color,
        )
        
        # Score label above the unit
        short_name = target.type_id.name[:6]
        label = f"{short_name} {best_score:.0f}"
        text_color = (255, 255, 0) if is_melee else (0, 200, 255)
        bot.client.debug_text_world(
            label,
            Point3((u_pos.x, u_pos.y, u_z + 1.5)),
            color=text_color,
            size=10,
        )


def render_observer_debug(bot) -> None:
    """Render 3D labels on each observer showing role and current mode.
    
    Uses obs.position3d for label height so labels float near flying observers.
    Also renders a 2D summary line with assignment status and hunt mode.
    """
    if not bot.debug:
        return
    
    observers = bot.units.filter(
        lambda u: u.type_id in {UnitTypeId.OBSERVER, UnitTypeId.OBSERVERSIEGEMODE}
    )
    
    assignments = bot.observer_assignments
    hunt_mode = getattr(bot, '_observer_hunt_mode', False)
    urgency = getattr(bot, '_intel_urgency', 0.0)
    hunter_tag = getattr(bot, '_hunting_observer_tag', None)
    detection_assignments = getattr(bot, '_observer_detection_assignments', {})
    
    for obs in observers:
        tag = obs.tag
        is_siege = obs.type_id == UnitTypeId.OBSERVERSIEGEMODE
        target_pos = None  # For drawing line to target
        
        # Detection override takes priority (above hunting and role labels)
        if tag in detection_assignments:
            enemy_tag = detection_assignments[tag]
            enemy_unit = None
            for enemy in bot.enemy_units:
                if enemy.tag == enemy_tag:
                    enemy_unit = enemy
                    break
            if enemy_unit:
                target_pos = enemy_unit.position
                label = f"DETECT {enemy_unit.type_id.name[:4]}"
            else:
                label = "DETECT ?"
            color = (255, 0, 255)  # Magenta
        elif tag == hunter_tag:
            label, color = "HUNT", (255, 165, 0)
        elif tag == assignments.get("army"):
            label, color = "ARMY", (0, 200, 255)
        elif tag == assignments.get("primary"):
            if is_siege:
                label, color = "SIEGE", (100, 100, 255)
            else:
                label, color = "STATION", (0, 255, 0)
        elif tag in assignments.get("patrol", []):
            target_pos = bot.observer_targets.get(tag)
            dist = f" D:{cy_distance_to(obs.position, target_pos):.0f}" if target_pos else ""
            label, color = f"PATROL{dist}", (200, 200, 0)
        else:
            label, color = "UNASSIGNED", (255, 0, 0)
        
        # Use actual 3D position so label appears near the flying observer
        pos3d = obs.position3d
        bot.client.debug_text_world(
            label,
            Point3((pos3d.x, pos3d.y, pos3d.z + 1.0)),
            color=color,
            size=14,
        )
        
        # Draw line from observer to its target if we have one
        if target_pos:
            target_z = bot.get_terrain_z_height(target_pos)
            bot.client.debug_line_out(
                Point3((pos3d.x, pos3d.y, pos3d.z)),
                Point3((target_pos.x, target_pos.y, target_z + 1.0)),
                color=Point3(color),
            )
    
    # 2D summary line
    pri = "SIEGE" if any(
        o.type_id == UnitTypeId.OBSERVERSIEGEMODE and o.tag == assignments.get("primary")
        for o in observers
    ) else ("OK" if assignments.get("primary") else "NONE")
    army = "OK" if assignments.get("army") else "NONE"
    patrol_count = len(assignments.get("patrol", []))
    hunt_str = f"HUNT(closest)" if hunt_mode else "off"
    
    # Hallucinated Phoenix labels
    hallu_phoenixes = bot.units(UnitTypeId.PHOENIX).filter(lambda u: u.is_hallucination)
    hallu_count = len(hallu_phoenixes)
    for phoenix in hallu_phoenixes:
        halu_role = bot._halu_scout_roles.get(phoenix.tag, '')
        if halu_role == "high_ground":
            label, color = "HALU:HIGH_GND", (255, 100, 255)
        elif halu_role == "base_scout":
            label, color = "HALU:BASE_SCOUT", (255, 200, 100)
        else:
            label, color = "HALU:SCOUT", (200, 255, 100)
        pos3d = phoenix.position3d
        bot.client.debug_text_world(
            label,
            Point3((pos3d.x, pos3d.y, pos3d.z + 1.0)),
            color=color,
            size=14,
        )
    
    _y = min(getattr(bot, '_debug_y', 0.46), 0.95)
    detect_count = len(detection_assignments)
    bot.client.debug_text_2d(
        f"Obs: ARMY:{army} PRI:{pri} PAT:{patrol_count} DETECT:{detect_count} {hunt_str} Halu:{hallu_count} Urg:{urgency:.2f}",
        Point2((0.1, _y)), None, 12
    )
    bot._debug_y = _y + 0.018


def render_blind_ramp_debug(bot, unit, ramp) -> None:
    """Render debug for units blocked by blind ramp attack check.

    Shows:
    - Orange 'BLIND RAMP' label on the affected unit
    - Orange line from unit to the ramp bottom it's reacting to
    - Red sphere at the ramp top (unseen enemy location)
    """
    if not bot.debug:
        return

    pos = unit.position
    z = bot.get_terrain_z_height(pos)

    # Find which enemy triggered this (re-check logic to get the enemy type)
    enemy_label = "?"
    for enemy in bot.all_enemy_units:
        if enemy.ground_range < 2:
            continue
        if cy_distance_to(enemy.position, ramp.top_center) < 6.0:
            enemy_label = enemy.type_id.name[:6]
            break

    # Orange label on unit
    bot.client.debug_text_world(
        f"BLIND RAMP ({enemy_label})",
        Point3((pos.x, pos.y, z + 1.5)),
        color=(255, 100, 0),
        size=12,
    )

    # Orange line from unit to ramp bottom
    ramp_z = bot.get_terrain_z_height(ramp.bottom_center)
    bot.client.debug_line_out(
        Point3((pos.x, pos.y, z + 0.3)),
        Point3((ramp.bottom_center.x, ramp.bottom_center.y, ramp_z + 0.3)),
        color=Point3((255, 100, 0)),
    )

    # Red sphere at ramp top (where we can't see)
    top_z = bot.get_terrain_z_height(ramp.top_center)
    bot.client.debug_sphere_out(
        Point3((ramp.top_center.x, ramp.top_center.y, top_z + 0.5)),
        1.0,
        Point3((255, 0, 0)),
    )


def render_concave_formation_debug(
    bot,
    squad_id: str,
    ranged_units: list,
    squad_center,
    enemy_center,
    formation_active: bool,
) -> None:
    """Render concave fan-out debug: sub-group assignments, fan targets, spread line.

    Shows:
    - 3 fan-out target spheres (magenta left, white center, cyan right)
    - Lines from each ranged unit to its sub-group target
    - Perpendicular spread line across the approach axis
    - Formation state label at squad center
    """
    if not bot.debug or not ranged_units:
        return

    from bot.combat.formation import split_into_fan_groups, compute_fan_targets

    # Get formation state
    state = None
    if hasattr(bot, '_squad_formation_state'):
        state = bot._squad_formation_state.get(squad_id)

    status = "SPREADING" if formation_active else ("DONE" if state and state.get("status") == "done" else "READY")

    # Squad center label
    z = bot.get_terrain_z_height(squad_center)
    color = (255, 0, 255) if formation_active else (128, 128, 128)
    frames_str = ""
    if state and "frame_start" in state:
        elapsed = bot.state.game_loop - state["frame_start"]
        frames_str = f" F:{elapsed}"
    bot.client.debug_text_world(
        f"CONCAVE:{status}{frames_str}",
        Point3((squad_center.x, squad_center.y, z + 2.0)),
        color=color,
        size=14,
    )

    # Compute groups and targets
    left_group, center_group, right_group = split_into_fan_groups(
        ranged_units, squad_center, enemy_center
    )
    left_target, center_target, right_target = compute_fan_targets(
        squad_center, enemy_center, len(ranged_units)
    )

    # Draw fan-out target spheres
    # Left = magenta, Center = white, Right = cyan
    targets_and_colors = [
        (left_target, Point3((255, 0, 255)), "L"),
        (center_target, Point3((255, 255, 255)), "C"),
        (right_target, Point3((0, 255, 255)), "R"),
    ]
    for target, sphere_color, label in targets_and_colors:
        tz = bot.get_terrain_z_height(target)
        bot.client.debug_sphere_out(
            Point3((target.x, target.y, tz + 0.5)),
            0.8,
            sphere_color,
        )
        bot.client.debug_text_world(
            label,
            Point3((target.x, target.y, tz + 1.5)),
            color=(255, 255, 255),
            size=12,
        )

    # Draw lines from each unit to its sub-group target
    groups_and_targets = [
        (left_group, left_target, Point3((255, 0, 255))),      # Magenta
        (center_group, center_target, Point3((255, 255, 255))), # White
        (right_group, right_target, Point3((0, 255, 255))),     # Cyan
    ]
    for group, target, line_color in groups_and_targets:
        tz = bot.get_terrain_z_height(target)
        for unit in group:
            uz = bot.get_terrain_z_height(unit.position)
            bot.client.debug_line_out(
                Point3((unit.position.x, unit.position.y, uz + 0.3)),
                Point3((target.x, target.y, tz + 0.3)),
                color=line_color,
            )

    # Draw perpendicular spread line (yellow dashed — left target to right target)
    lz = bot.get_terrain_z_height(left_target)
    rz = bot.get_terrain_z_height(right_target)
    bot.client.debug_line_out(
        Point3((left_target.x, left_target.y, lz + 0.2)),
        Point3((right_target.x, right_target.y, rz + 0.2)),
        color=Point3((255, 255, 0)),  # Yellow spread line
    )

    # 2D summary
    dist = cy_distance_to(squad_center, enemy_center)
    _y = min(getattr(bot, '_debug_y', 0.48), 0.95)
    bot.client.debug_text_2d(
        f"Concave: {status} | {len(ranged_units)}rng | L:{len(left_group)} C:{len(center_group)} R:{len(right_group)} | D:{dist:.0f}",
        Point2((0.1, _y)), None, 12
    )
    bot._debug_y = _y + 0.018


def render_choke_policy_debug(
    bot,
    squad_position,
    enemy_center,
    melee_ratio: float,
    choke_width: float,
    our_width: float,
    enemy_width: float,
) -> None:
    """Render debug for choke policy engagement suppression.

    Shows:
    - Yellow 'CHOKE HOLD' label at squad position with melee DPS ratio
    - Choke width vs army widths for both sides
    - Yellow line from squad to enemy center (crossing the choke)
    """
    if not bot.debug:
        return

    z = bot.get_terrain_z_height(squad_position)

    # Yellow label on squad with width comparison
    bot.client.debug_text_world(
        f"CHOKE HOLD ({melee_ratio:.0%} melee) | choke:{choke_width:.1f} us:{our_width:.1f} them:{enemy_width:.1f}",
        Point3((squad_position.x, squad_position.y, z + 2.0)),
        color=(255, 255, 0),
        size=14,
    )

    # Yellow line from squad to enemy center
    ez = bot.get_terrain_z_height(enemy_center)
    bot.client.debug_line_out(
        Point3((squad_position.x, squad_position.y, z + 0.3)),
        Point3((enemy_center.x, enemy_center.y, ez + 0.3)),
        color=Point3((255, 255, 0)),
    )


def render_choke_decision_debug(
    bot,
    squad_position,
    enemy_center,
    choke_width: float,
    our_width: float,
    enemy_width: float,
    decision: str,
) -> None:
    """Render the choke policy decision for a squad near a choke.

    Shows the decision outcome even when the policy does NOT suppress engagement,
    so you can see why the bot chose to engage or hold.

    Args:
        decision: One of 'HOLD:we_funnel', 'HOLD:lure', 'HOLD:melee_ratio',
                  'PASS:irrelevant', 'PASS:both_fit'
    """
    if not bot.debug:
        return

    z = bot.get_terrain_z_height(squad_position)

    # Color: yellow for HOLD, green for PASS (engaging normally)
    if decision.startswith("HOLD"):
        color = (255, 255, 0)  # Yellow
    else:
        color = (0, 200, 0)    # Green

    bot.client.debug_text_world(
        f"CHOKE {decision} | w:{choke_width:.1f} us:{our_width:.1f} them:{enemy_width:.1f}",
        Point3((squad_position.x, squad_position.y, z + 1.5)),
        color=color,
        size=12,
    )

    # Line from squad to enemy center (yellow for hold, green for pass)
    ez = bot.get_terrain_z_height(enemy_center)
    line_color = Point3((255, 255, 0)) if decision.startswith("HOLD") else Point3((0, 200, 0))
    bot.client.debug_line_out(
        Point3((squad_position.x, squad_position.y, z + 0.3)),
        Point3((enemy_center.x, enemy_center.y, ez + 0.3)),
        color=line_color,
    )


def render_narrow_choke_points(bot) -> None:
    """Render narrow choke point tiles and width labels on the map.
    
    Draws:
    - Small yellow spheres at each narrow choke tile
    - Width label at each choke's center point
    - Side_a → side_b line showing the choke's narrowest span
    
    Called every frame from on_step (SC2 clears debug drawings each frame).
    Lightweight: only iterates the pre-computed dict, no map_analyzer calls.
    """
    if not bot.debug:
        return
    
    choke_width_map = getattr(bot, 'narrow_choke_points', None)
    if not choke_width_map:
        return
    
    # Group tiles by width (each unique width = one choke)
    width_to_tiles: dict[float, list[Point2]] = {}
    for tile, w in choke_width_map.items():
        width_to_tiles.setdefault(round(w, 1), []).append(tile)
    
    # Draw each choke group as small spheres, capped to avoid crashing the SC2 debug renderer.
    for tiles in width_to_tiles.values():
        # Large chokes can have 100+ tiles; rendering all of them every frame
        # exceeds the client's draw call budget and crashes the game.
        max_spheres = 40
        step = max(1, len(tiles) // max_spheres)
        for i in range(0, len(tiles), step):
            tile = tiles[i]
            tz = bot.get_terrain_z_height(tile)
            bot.client.debug_sphere_out(
                Point3((tile.x, tile.y, tz + 0.3)),
                0.3,
                Point3((255, 255, 0)),  # Yellow
            )


def render_ff_split_debug(
    bot,
    ff_assignments: dict[int, list[Point2]] | None,
    sentries: list,
    enemy_center: Point2 | None = None,
    ff_mode: str = "",
) -> None:
    """Render Force Field debug visualization.

    Shows:
    - Cyan spheres at each planned FF position
    - Cyan line connecting FF positions (the placement line)
    - Sentry energy labels showing who's casting
    - Red 'NO SPLIT' label if split was attempted but failed
    - Mode label (RAMP BLOCK / CHOKE BLOCK / FF SPLIT) at enemy center

    Args:
        bot: Bot instance
        ff_assignments: Dict mapping sentry tag → list of FF positions, or None
        sentries: List of sentry units in the squad
        enemy_center: Enemy army center (for label placement)
        ff_mode: Which FF mode fired — "RAMP", "CHOKE", "SPLIT", or "" (unknown/none)
    """
    if not bot.debug:
        return

    # Always show sentry energy labels
    for s in sentries:
        pos = s.position
        z = bot.get_terrain_z_height(pos)
        label = f"E:{s.energy:.0f}"
        color = (0, 255, 255) if s.energy >= 30 else (128, 128, 128)
        bot.client.debug_text_world(
            label,
            Point3((pos.x, pos.y, z + 1.5)),
            color=color,
            size=12,
        )

    if ff_assignments is None:
        # No split this frame — show indicator if we have an enemy center
        if enemy_center is not None:
            z = bot.get_terrain_z_height(enemy_center)
            bot.client.debug_text_world(
                "NO SPLIT",
                Point3((enemy_center.x, enemy_center.y, z + 2.5)),
                color=(255, 80, 80),
                size=12,
            )
        return

    # Draw each FF position as a cyan sphere
    all_positions: list[Point2] = []
    for tag, positions in ff_assignments.items():
        for pos in positions:
            all_positions.append(pos)
            z = bot.get_terrain_z_height(pos)
            # Sphere at FF center
            bot.client.debug_sphere_out(
                Point3((pos.x, pos.y, z + 0.3)),
                1.5,  # FF_RADIUS
                Point3((0, 255, 255)),  # Cyan
            )
            # Label showing which sentry is casting
            bot.client.debug_text_world(
                f"FF→{tag}",
                Point3((pos.x, pos.y, z + 2.0)),
                color=(0, 255, 255),
                size=10,
            )

    # Draw line connecting FF positions (the split line)
    if len(all_positions) >= 2:
        for i in range(len(all_positions) - 1):
            p1 = all_positions[i]
            p2 = all_positions[i + 1]
            z1 = bot.get_terrain_z_height(p1)
            z2 = bot.get_terrain_z_height(p2)
            bot.client.debug_line_out(
                Point3((p1.x, p1.y, z1 + 0.3)),
                Point3((p2.x, p2.y, z2 + 0.3)),
                color=Point3((0, 255, 255)),
            )

    # Label at enemy center showing placement info
    if enemy_center is not None:
        z = bot.get_terrain_z_height(enemy_center)
        total_ffs = sum(len(pos_list) for pos_list in ff_assignments.values())
        total_energy = sum(s.energy for s in sentries)
        if ff_mode == "RAMP":
            label = "RAMP BLOCK"
        elif ff_mode == "CHOKE":
            label = f"CHOKE BLOCK ffs:{total_ffs}"
        elif ff_mode == "DCHOKE":
            label = f"DYN CHOKE ffs:{total_ffs}"
        else:
            label = "RAMP BLOCK" if total_ffs == 1 else f"FF SPLIT ffs:{total_ffs}"
        bot.client.debug_text_world(
            f"{label} pool:{total_energy:.0f}",
            Point3((enemy_center.x, enemy_center.y, z + 3.0)),
            color=(0, 255, 255),
            size=12,
        )


def render_gs_debug(bot, sentries: list, squad_position: Point2) -> None:
    """Render Guardian Shield influence-map debug visualization.

    Shows:
    - Friendly-density grid as on-map text (only non-zero cells, green)
    - Green sphere at each assigned peak (where a Sentry should go)
    - Green ring at GUARDIAN_SHIELD_RADIUS around each peak (coverage area)
    - Sentry energy + "GS→peak" labels for approved Sentries
    - "NO GS" label at squad center if ranged enemies exist but no Sentry
      was approved (full coverage already achieved)

    Reads debug state stashed by _compute_guardian_shield_assignments() on
    bot._gs_debug_density / bot._gs_debug_peaks when bot.debug is True.

    Args:
        bot: Bot instance
        sentries: List of sentry units in the squad
        squad_position: Squad center (for the "NO GS" fallback label)
    """
    if not bot.debug:
        return

    from bot.constants import GUARDIAN_SHIELD_RADIUS

    density = getattr(bot, "_gs_debug_density", None)
    peaks: dict = getattr(bot, "_gs_debug_peaks", {})

    # 1) Density grid overlay — paint non-zero cells as small green numbers
    if density is not None:
        height = bot.get_terrain_z_height(squad_position)
        # Sample every 2nd tile to avoid flooding the renderer on large armies
        ys, xs = np.where(density > 0)
        step = 2
        for i in range(0, len(ys), step):
            x, y = int(xs[i]), int(ys[i])
            val = int(density[x, y])
            bot.client.debug_text_world(
                str(val),
                Point3((x, y, height)),
                color=(0, 200, 0),
                size=8,
            )

    # 2) Peak markers + coverage rings
    for tag, peak in peaks.items():
        z = bot.get_terrain_z_height(peak)
        bot.client.debug_sphere_out(
            Point3((peak.x, peak.y, z + 0.3)),
            0.8,
            Point3((0, 255, 0)),
        )
        bot.client.debug_text_world(
            f"GS→{tag}",
            Point3((peak.x, peak.y, z + 2.0)),
            color=(0, 255, 0),
            size=10,
        )
        # Coverage ring: 12 segments around the peak at GUARDIAN_SHIELD_RADIUS
        for j in range(12):
            a1 = (2 * math.pi * j) / 12
            a2 = (2 * math.pi * (j + 1)) / 12
            p1 = Point2((peak.x + math.cos(a1) * GUARDIAN_SHIELD_RADIUS,
                         peak.y + math.sin(a1) * GUARDIAN_SHIELD_RADIUS))
            p2 = Point2((peak.x + math.cos(a2) * GUARDIAN_SHIELD_RADIUS,
                         peak.y + math.sin(a2) * GUARDIAN_SHIELD_RADIUS))
            z1 = bot.get_terrain_z_height(p1)
            z2 = bot.get_terrain_z_height(p2)
            bot.client.debug_line_out(
                Point3((p1.x, p1.y, z1 + 0.3)),
                Point3((p2.x, p2.y, z2 + 0.3)),
                color=Point3((0, 200, 0)),
            )

    # 3) Sentry labels — energy + whether approved
    for s in sentries:
        pos = s.position
        z = bot.get_terrain_z_height(pos)
        approved = s.tag in peaks
        label = f"E:{s.energy:.0f}{' GS' if approved else ''}"
        color = (0, 255, 0) if approved else (128, 128, 128)
        bot.client.debug_text_world(
            label,
            Point3((pos.x, pos.y, z + 1.5)),
            color=color,
            size=12,
        )

    # 4) "NO GS" if there are sentries with energy but no peaks (full coverage)
    if not peaks and sentries:
        capable = [s for s in sentries if s.energy >= 75]
        if capable:
            z = bot.get_terrain_z_height(squad_position)
            bot.client.debug_text_world(
                "NO GS (covered)",
                Point3((squad_position.x, squad_position.y, z + 2.5)),
                color=(160, 160, 160),
                size=12,
            )


def render_refined_choke_debug(
    bot,
    refined,
    enemy_center: Point2 | None = None,
) -> None:
    """Render debug for a raycast-refined choke when a choke block is active.

    Shows:
    - Green sphere at the refined choke center
    - Green line across the passage (perpendicular direction, spanning the measured width)
    - Green arrow along the passage axis showing orientation

    Args:
        bot: Bot instance
        refined: RefinedChoke dataclass (center, width, axis, perp)
        enemy_center: Enemy center if available, for a line from choke to enemy
    """
    if not bot.debug or refined is None:
        return

    center = refined.center
    z = bot.get_terrain_z_height(center)

    # Green sphere at the refined center
    bot.client.debug_sphere_out(
        Point3((center.x, center.y, z + 0.3)),
        0.8,
        Point3((0, 255, 0)),  # Green
    )

    # Green line across the passage (perpendicular, spanning width)
    half_w = refined.width / 2.0
    p1 = Point2((
        center.x + refined.perp.x * half_w,
        center.y + refined.perp.y * half_w,
    ))
    p2 = Point2((
        center.x - refined.perp.x * half_w,
        center.y - refined.perp.y * half_w,
    ))
    z1 = bot.get_terrain_z_height(p1)
    z2 = bot.get_terrain_z_height(p2)
    bot.client.debug_line_out(
        Point3((p1.x, p1.y, z1 + 0.3)),
        Point3((p2.x, p2.y, z2 + 0.3)),
        color=Point3((0, 255, 0)),
    )

    # Green arrow along the passage axis (short, for orientation)
    arrow_end = Point2((
        center.x + refined.axis.x * 3.0,
        center.y + refined.axis.y * 3.0,
    ))
    za = bot.get_terrain_z_height(arrow_end)
    bot.client.debug_line_out(
        Point3((center.x, center.y, z + 0.3)),
        Point3((arrow_end.x, arrow_end.y, za + 0.3)),
        color=Point3((0, 180, 0)),
    )

    # Dashed-style line from choke center to enemy center if provided
    if enemy_center is not None:
        ez = bot.get_terrain_z_height(enemy_center)
        bot.client.debug_line_out(
            Point3((center.x, center.y, z + 0.3)),
            Point3((enemy_center.x, enemy_center.y, ez + 0.3)),
            color=Point3((0, 100, 0)),
        )


def render_refined_choke_points(bot) -> None:
    """Render raycast-refined choke points near any squad.

    Only draws chokes within REFINED_CHOKE_RENDER_RADIUS of a squad position,
    to avoid exceeding the SC2 debug draw call budget (rendering all 20-30
    refined chokes every frame crashes the client).

    Draws for each nearby refined choke:
    - Green sphere at the refined center
    - Green line spanning the measured width (perpendicular to passage)
    - Dark green arrow along the passage axis
    """
    if not bot.debug:
        return

    cache = getattr(bot, 'refined_choke_points', None)
    if not cache:
        return

    # Collect squad positions to filter chokes by proximity.
    # Rendering all refined chokes every frame crashes the SC2 debug renderer.
    from ares.consts import UnitRole
    from bot.constants import ATTACKING_SQUAD_RADIUS, DEFENDER_SQUAD_RADIUS
    squad_positions: list[Point2] = []
    for role, radius in (
        (UnitRole.ATTACKING, ATTACKING_SQUAD_RADIUS),
        (UnitRole.DEFENDING, ATTACKING_SQUAD_RADIUS),
        (UnitRole.BASE_DEFENDER, DEFENDER_SQUAD_RADIUS),
    ):
        for squad in bot.mediator.get_squads(role=role, squad_radius=radius):
            squad_positions.append(Point2(squad.squad_position))

    if not squad_positions:
        return

    render_radius = 20.0  # Only show chokes within 20 tiles of a squad

    for choke_tile, refined in cache.items():
        if refined is None:
            continue

        center = refined.center
        # Skip chokes far from any squad
        if not any(cy_distance_to(center, sp) <= render_radius for sp in squad_positions):
            continue

        z = bot.get_terrain_z_height(center)

        # Green sphere at the refined center
        bot.client.debug_sphere_out(
            Point3((center.x, center.y, z + 0.3)),
            0.8,
            Point3((0, 255, 0)),  # Green
        )

        # Green line across the passage (perpendicular, spanning width)
        half_w = refined.width / 2.0
        p1 = Point2((
            center.x + refined.perp.x * half_w,
            center.y + refined.perp.y * half_w,
        ))
        p2 = Point2((
            center.x - refined.perp.x * half_w,
            center.y - refined.perp.y * half_w,
        ))
        z1 = bot.get_terrain_z_height(p1)
        z2 = bot.get_terrain_z_height(p2)
        bot.client.debug_line_out(
            Point3((p1.x, p1.y, z1 + 0.3)),
            Point3((p2.x, p2.y, z2 + 0.3)),
            color=Point3((0, 255, 0)),
        )

        # Dark green arrow along the passage axis (short, for orientation)
        arrow_end = Point2((
            center.x + refined.axis.x * 3.0,
            center.y + refined.axis.y * 3.0,
        ))
        za = bot.get_terrain_z_height(arrow_end)
        bot.client.debug_line_out(
            Point3((center.x, center.y, z + 0.3)),
            Point3((arrow_end.x, arrow_end.y, za + 0.3)),
            color=Point3((0, 180, 0)),
        )


def render_snipe_debug(bot, squad_id: str) -> None:
    """Render debug overlay for active blink snipe.

    Shows: phase (approach / fire), target unit, committed stalkers with colored markers,
    and retreat lines. Only renders if bot.debug and snipe is active.

    Args:
        bot: Bot instance
        squad_id: Squad ID to check for active snipe
    """
    if not bot.debug:
        return

    if squad_id not in bot._snipe_state:
        return

    info = bot._snipe_state[squad_id]
    fired = info.get("fired", False)
    target_tag = info["target_tag"]
    stalker_tags = info["stalker_tags"]

    # Find target unit for position
    target = None
    for u in bot.all_enemy_units:
        if u.tag == target_tag:
            target = u
            break

    phase = "FIRE" if fired else "APPROACH"
    color = (255, 60, 60) if fired else (255, 255, 0)

    # Draw phase label at target position
    if target is not None:
        pos = target.position
        z = bot.get_terrain_z_height(pos)
        bot.client.debug_text_world(
            f"SNIPE:{phase} ({len(stalker_tags)}stk)",
            Point3((pos.x, pos.y, z + 2.5)),
            color=color,
            size=14,
        )
        # Ring around target
        bot.client.debug_sphere_out(
            Point3((pos.x, pos.y, z + 0.5)),
            1.0,
            Point3(color),
        )

    # Mark committed stalkers with orange dots
    for u in bot.units:
        if u.tag in stalker_tags:
            z = bot.get_terrain_z_height(u.position)
            bot.client.debug_sphere_out(
                Point3((u.position.x, u.position.y, z + 1.5)),
                0.3,
                Point3((255, 165, 0)),
            )

    # Draw retreat lines when fired (about to blink back)
    if fired and "retreat_point" in info:
        rp = info["retreat_point"]
        rz = bot.get_terrain_z_height(rp)
        for u in bot.units:
            if u.tag in stalker_tags:
                uz = bot.get_terrain_z_height(u.position)
                bot.client.debug_line_out(
                    Point3((u.position.x, u.position.y, uz + 0.5)),
                    Point3((rp.x, rp.y, rz + 0.5)),
                    color=Point3((0, 255, 100)),
                )


def render_chase_debug(bot, squad_id: str) -> None:
    """Render debug overlay for active blink chase.

    Shows: CHASE label at target, lines from stalkers to target,
    cyan markers on committed chasers.
    """
    if not bot.debug:
        return

    if squad_id not in bot._chase_state:
        return

    info = bot._chase_state[squad_id]
    target_tag = info["target_tag"]
    stalker_tags = info["stalker_tags"]

    # Find target unit
    target = None
    for u in bot.all_enemy_units:
        if u.tag == target_tag:
            target = u
            break

    color = (0, 200, 255)  # cyan

    # Draw CHASE label at target
    if target is not None:
        pos = target.position
        z = bot.get_terrain_z_height(pos)
        bot.client.debug_text_world(
            f"CHASE ({len(stalker_tags)}stk)",
            Point3((pos.x, pos.y, z + 2.5)),
            color=color,
            size=14,
        )
        # Ring around target
        bot.client.debug_sphere_out(
            Point3((pos.x, pos.y, z + 0.5)),
            1.0,
            Point3(color),
        )

    # Mark committed stalkers with cyan dots + lines to target
    for u in bot.units:
        if u.tag in stalker_tags:
            uz = bot.get_terrain_z_height(u.position)
            bot.client.debug_sphere_out(
                Point3((u.position.x, u.position.y, uz + 1.5)),
                0.3,
                Point3(color),
            )
            if target is not None:
                tz = bot.get_terrain_z_height(target.position)
                bot.client.debug_line_out(
                    Point3((u.position.x, u.position.y, uz + 0.5)),
                    Point3((target.position.x, target.position.y, tz + 0.5)),
                    color=Point3(color),
                )


def render_focus_debug(bot, squad_id: str) -> None:
    """Render debug overlay for active blink focus-fire.

    Shows: FOCUS label at target, approach method (BLINK/WALK), committed stalkers
    with purple markers, and lines to target. Only renders if bot.debug and focus is active.

    Args:
        bot: Bot instance
        squad_id: Squad ID to check for active focus-fire
    """
    if not bot.debug:
        return

    if squad_id not in bot._focus_state:
        return

    info = bot._focus_state[squad_id]
    target_tag = info["target_tag"]
    stalker_tags = info["stalker_tags"]
    blink_in = info.get("blink_in", False)

    # Find target unit for position
    target = None
    for u in bot.all_enemy_units:
        if u.tag == target_tag:
            target = u
            break

    approach = "BLINK" if blink_in else "WALK"
    color = (180, 0, 255)  # purple

    # Draw FOCUS label at target position
    if target is not None:
        pos = target.position
        z = bot.get_terrain_z_height(pos)
        bot.client.debug_text_world(
            f"FOCUS:{approach} ({len(stalker_tags)}stk)",
            Point3((pos.x, pos.y, z + 2.5)),
            color=color,
            size=14,
        )
        # Ring around target
        bot.client.debug_sphere_out(
            Point3((pos.x, pos.y, z + 0.5)),
            1.0,
            Point3(color),
        )

    # Mark committed stalkers with purple dots + lines to target
    for u in bot.units:
        if u.tag in stalker_tags:
            uz = bot.get_terrain_z_height(u.position)
            bot.client.debug_sphere_out(
                Point3((u.position.x, u.position.y, uz + 1.5)),
                0.3,
                Point3(color),
            )
            if target is not None:
                tz = bot.get_terrain_z_height(target.position)
                bot.client.debug_line_out(
                    Point3((u.position.x, u.position.y, uz + 0.5)),
                    Point3((target.position.x, target.position.y, tz + 0.5)),
                    color=Point3(color),
                )


def log_nova_error(error: Exception) -> None:
    """
    Log NovaManager errors (always shown, not gated by debug flag).
    
    Args:
        error: Exception that occurred
    """
    print(f"DEBUG ERROR updating NovaManager: {error}")


def render_micro_state_debug(
    bot,
    unit,
    is_threatened: bool,
    aggressive: bool,
    weapon_ready: bool,
    in_range: bool,
) -> None:
    """Render per-unit micro state labels above ranged units.

    Shows: threat status (M=melee, R=ranged, -=safe), aggressive mode (A/D),
    weapon state (W=ready, C=cooldown), range status (IR=in range, OR=out of range).

    Color coding:
        Green: safe + aggressive + weapon ready
        Yellow: threatened + aggressive (kiting)
        Red: defensive mode
        Orange: safe + aggressive + cooldown (advancing)
    """
    if not bot.debug:
        return

    # Build short state string
    threat_str = "T" if is_threatened else "-"
    aggr_str = "A" if aggressive else "D"
    weap_str = "W" if weapon_ready else "C"
    range_str = "IR" if in_range else "OR"
    label = f"{threat_str}{aggr_str}{weap_str}{range_str}"

    # Color based on state
    if not aggressive:
        color = (255, 80, 80)       # Red: defensive
    elif is_threatened:
        color = (255, 255, 0)        # Yellow: threatened (kiting)
    elif not weapon_ready:
        color = (255, 165, 0)        # Orange: safe + cooldown (advancing)
    else:
        color = (0, 255, 0)          # Green: safe + weapon ready

    pos = unit.position
    z = bot.get_terrain_z_height(pos)
    bot.client.debug_text_world(
        label,
        Point3((pos.x, pos.y, z + 2.0)),
        color=color,
        size=10,
    )


def render_detection_cannon_debug(bot) -> None:
    """Render detection cannon state overlay: per-base state labels and 2D summary.

    Shows:
    - 3D label above each nexus showing detection cannon state
    - 2D summary line with state counts

    Only renders when bot.debug is True and detection cannon system is active.
    """
    if not bot.debug:
        return

    state_dict = getattr(bot, '_detection_cannon_state', {})
    triggered = getattr(bot, '_detection_cannon_triggered', False)
    if not triggered or not state_dict:
        return

    from sc2.ids.unit_typeid import UnitTypeId

    # State colors for 3D labels
    state_colors = {
        "needs_pylon": (255, 165, 0),    # Orange — waiting to build pylon
        "pylon_pending": (255, 255, 0),  # Yellow — pylon under construction
        "needs_cannon": (0, 200, 255),    # Cyan — waiting to build cannon
        "cannon_pending": (100, 100, 255), # Blue — cannon under construction
        "complete": (0, 255, 0),           # Green — done
    }

    # 3D labels at each nexus
    for nexus in bot.townhalls.ready:
        tag = nexus.tag
        if tag not in state_dict:
            continue
        state = state_dict[tag]
        color = state_colors.get(state, (255, 255, 255))
        pos = nexus.position3d
        label = f"DET:{state}"
        bot.client.debug_text_world(
            label,
            Point3((pos.x, pos.y, pos.z + 3.0)),
            color=color,
            size=12,
        )

    # 2D summary line
    counts = {}
    for state in state_dict.values():
        counts[state] = counts.get(state, 0) + 1
    parts = [f"{state}:{count}" for state, count in sorted(counts.items())]
    summary = "DetCannon " + " ".join(parts)

    _y = min(getattr(bot, '_debug_y', 0.50), 0.95)
    bot.client.debug_text_2d(summary, Point2((0.1, _y)), None, 12)
    bot._debug_y = _y + 0.018


def render_nexus_ability_debug(bot) -> None:
    """Render 3D labels above each Nexus showing energy and available abilities.

    Shows: energy/energy_max, plus a letter per ability if currently castable:
      R = Energy Recharge, C = Chrono Boost, M = Mass Recall

    Color coding:
        Green: 2+ abilities available
        Yellow: 1 ability available
        Gray: no abilities available (low energy or on cooldown)

    Only renders when bot.debug is True.
    """
    if not bot.debug:
        return

    from sc2.ids.ability_id import AbilityId

    for nexus in bot.townhalls.ready:
        if nexus.type_id != UnitTypeId.NEXUS:
            continue

        # Check which abilities are currently castable (accounts for energy + cooldowns)
        abilities = nexus.abilities
        can_recharge = AbilityId.ENERGYRECHARGE_ENERGYRECHARGE in abilities
        can_chrono = AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities
        can_recall = AbilityId.EFFECT_MASSRECALL_NEXUS in abilities

        avail = []
        if can_recharge:
            avail.append("R")
        if can_chrono:
            avail.append("C")
        if can_recall:
            avail.append("M")
        avail_str = "".join(avail) if avail else "CD"

        label = f"E:{int(nexus.energy)}/{int(nexus.energy_max)} [{avail_str}]"

        # Color by how many abilities are available
        n_avail = len(avail)
        if n_avail >= 2:
            color = (0, 255, 0)    # Green — multiple options
        elif n_avail == 1:
            color = (255, 255, 0)  # Yellow — one option
        else:
            color = (128, 128, 128)  # Gray — nothing available

        pos = nexus.position
        z = bot.get_terrain_z_height(pos)
        bot.client.debug_text_world(
            label,
            Point3((pos.x, pos.y, z + 3.0)),
            color=color,
            size=12,
        )


def render_repair_debug(bot) -> None:
    """Render labels over enemy Bunkers/PFs confirmed as being repaired.

    Shows "REPAIRING +N" over each flagged structure (N = tagged SCV/MULE
    repairers nearby). Only renders when bot.debug is True.
    """
    if not bot.debug:
        return

    repairing = getattr(bot, "_repairing_structures", {})
    repairer_tags = getattr(bot, "_repairer_tags", set())
    if not repairing:
        return

    for info in repairing.values():
        pos = info["position"]
        age = bot.time - info["time"]
        # Count live repairers (workers die/leave between detection frames)
        n_repairers = sum(
            1 for u in bot.enemy_units
            if u.tag in repairer_tags and cy_distance_to(u.position, pos) <= 4.0
        )
        z = bot.get_terrain_z_height(pos)
        bot.client.debug_text_world(
            f"REPAIRING +{n_repairers} ({age:.0f}s)",
            Point3((pos.x, pos.y, z + 3.0)),
            color=(255, 140, 0),  # Orange — repair threat
            size=12,
        )
