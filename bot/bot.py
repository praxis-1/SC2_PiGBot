# bot.py
"""Main AresBot subclass wiring all bot modules together.

Purpose: per-frame orchestration — combat, scouting, reactions, and the
    build-runner exit valves that hand off to macro when the opening stalls.
Key Decisions: macro is locked out until the opening completes, so the exit
    valves (stall detector + bank backstops) run in all game states and are
    the only guaranteed escape from a supply-gated build deadlock.
Limitations: valve thresholds are global constants; per-build tuning would
    need profile support (see constants.py BUILD_* group).
"""

from typing import Optional
from itertools import cycle, chain
from pathlib import Path
import numpy as np

# Ares imports (framework-specific)
from ares import AresBot
from ares.consts import ALL_STRUCTURES, WORKER_TYPES, UnitRole
# from ares.managers.unit_manager import UnitManager
from ares.managers.squad_manager import UnitSquad
from ares.managers.manager_mediator import ManagerMediator
from map_analyzer import MapData
from ares.behaviors.macro.restore_power import RestorePower


# SC2-related imports
from sc2.data import Result
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units
from sc2.data import Race



# Modular imports for separated concerns
from bot.managers.macro import handle_macro, get_optimal_gas_workers, get_freeflow_mode
from bot.managers.composition_scoring import CompositionScorer
from bot.managers.structure_manager import use_chronoboost, use_recharge, use_mass_recall
from bot.managers.reactions import ReactionManager, threat_detection
from bot.combat import (
    control_main_army,
    control_defenders,
    warp_prism_follower,
    handle_attack_toggles,
    attack_target,
    gatekeeper_control,
    manage_defensive_unit_roles
)
from bot.intel import update_enemy_intel_tracking, update_repair_detection
from bot.utilities.choke_grid import create_choke_grid, create_narrow_choke_points, refine_all_chokes
from cython_extensions import cy_distance_to
from bot.utilities.debug import render_narrow_choke_points, render_refined_choke_points, render_nexus_ability_debug, render_expansion_debug
from ares.behaviors.macro import Mining
#debugs
from bot.utilities.use_disruptor_nova import UseDisruptorNova
from bot.utilities.nova_manager import NovaManager
from bot.utilities.performance_monitor import PerformanceMonitor
from bot.utilities.game_report import print_end_game_report, print_startup_report, print_periodic_intel_report, get_replay_tags_to_send, emit_match_record
from bot.utilities.telemetry import init_context, reset as telemetry_reset, log_transition
from bot.belief import BeliefUpdater, BeliefState, create_empty_belief_state




class PiG_Bot(AresBot):
    """
    Main class for our Protoss bot, leveraging the Ares framework.
    Logic is organized into separate modules for macro, combat, scouting, and reactions.
    """

    current_base_target: Point2
    expansions_generator: cycle

    def __init__(self, game_step_override: Optional[int] = None):
        """
        Initializes the bot with various flags and references needed by Ares.
        """
        super().__init__(game_step_override)

        # State tracking
        self.unit_roles = {}
        self.scout_targets = {}
        self.bases = {}
        self.total_health_shield_percentage = 1.0
        self.enemy_army = None
        self.own_army: Units = Units([], self)
        # Game state: 0 = early game, 1 = mid game, 2 = late game
        self.game_state = 0
        self.early_game_threshold = 360  # 6 minutes in seconds
        self.mid_game_threshold = 720    # 12 minutes in seconds
        
        # Observer assignments
        self.observer_assignments = {
            "primary": None,      # The first observer (enemy nat)
            "army": None,         # The second observer (follows army)
            "patrol": [],         # Additional observers (map locations)
        }
        self.observer_targets = {}  # Maps observer.tag -> current target
        self._observer_detection_assignments: dict[int, int] = {}  # observer tag → enemy unit tag (detection duty)
        self._location_last_seen: dict[str, float] = {}  # Scout VOI: location key → game time last scouted

        # Flags for in-game logic
        self._commenced_attack = False
        # Attack telemetry accumulators — persist across the whole game,
        # unlike the boolean above which only reflects the current state.
        self._attack_initiation_count = 0  # Total times attack was initiated
        self._total_attack_time = 0.0     # Cumulative seconds spent in attack state
        self._current_attack_start = 0.0  # Timestamp current attack began (0 = not attacking)
        self._rush_time_seconds = 0.0  # Rush time calculated in on_start
        self._under_attack = False
        self._is_building = False

        # Build runner stall tracking (exit valves read these every step)
        self._stall_start_time = 0.0
        self._last_build_progress_key: tuple | None = None

        # Reaction manager: owns all reaction state and routing
        self.reaction_manager = ReactionManager()

        # Debug flags (loaded from config.yml in on_start)
        self.debug = False  # Will be set from config.yml BotDebug
        
        # Target persistence for stable attack behavior
        self.current_attack_target = None
        self.target_lock_distance = 25.0  # Don't switch targets unless new one is 25+ units closer
        
        # Combat engagement tracking
        self._attack_commenced_time = 0.0  # Time when attack was initiated
        self._squad_engagement_tracker = {}  # Per-squad engagement decisions for tactical control
        
        # Performance monitoring (tracks SQ and other efficiency metrics)
        self.performance_monitor = PerformanceMonitor(sample_interval=22, alpha=0.1)
        
        # Gas worker management
        self._gas_worker_toggle = True  # Toggle for gas mining on/off
        
        # Strategic anchor positioning (for smart army placement)
        self._current_defensive_anchor = None  # Current anchor position
        self._anchor_change_time = 0.0  # Last time anchor was changed (for cooldown)
        
        # Enemy intel tracking (for combat sim trust)
        self._enemy_army_ever_seen = False  # Sticky flag: have we EVER seen enemy combat units?
        self._last_enemy_army_visible_time = 0.0  # Last time we had direct vision of enemy army
        self._enemy_unit_last_seen: dict[int, float] = {}  # tag -> last time seen
        
        # Composition caches (set by select_army_composition for debug overlay)
        self._last_base_comp: dict = {}
        self._last_selected_comp: dict = {}
        self._resource_pressure: str = "BALANCED"  # GAS_STARVED, MINERAL_STARVED, or BALANCED

        # Composition scorer — the macro manager's utility-AI decision layer.
        # It owns the candidate ballot, the IntelContext and the incumbent that
        # the commitment bonus protects; the macro manager asks it for a winner.
        self.composition_scorer = CompositionScorer(self.config)
        
        # Intel urgency system (gradual build/decay to avoid oscillation)
        # 0.0 = no urgency, 1.0 = max urgency
        # Responses trigger at thresholds: 0.3 (observer priority), 0.5 (extra observers), 0.7 (speed upgrade)
        self._intel_urgency = 0.0
        self._worker_scout_sent_this_stale_period = False  # Prevents spamming worker scouts
        self._br_scout_waypoints: dict = {}  # {tag: {'waypoints': list[Point2], 'idx': int}}
        self._observer_hunt_mode = False  # Sticky flag: stays True until freshness >= FRESH_INTEL_THRESHOLD
        self._hunting_observer_tag = None  # Tag of observer currently hunting (set per-frame)
        self._hallucination_scout_last_cast = 0.0  # Game time of last hallucination cast (cooldown tracking)
        self._halu_scout_roles: dict = {}  # {tag: role_str} per-Phoenix role tracking
        self._halu_scout_waypoints: dict = {}  # {tag: {'waypoints': list[Point2], 'idx': int}} per-Phoenix waypoints
        self._halu_scout_pending_role: str = ""  # Role for next hallucinated Phoenix (set before cast, consumed on spawn)
        self._blind_ramp_target: Optional[Point2] = None  # Set per-frame by combat when army blocked by blind ramp
        self._cloaked_threat_positions: list[Point2] = []  # Positions of cloaked/burrowed enemies near bases (set per-frame by threat_detection)

        # Blink snipe state (group snipe system)
        self._snipe_committed: dict[int, int] = {}  # tag → expiry_frame (skip-set for per-unit micro)
        self._snipe_state: dict[str, dict] = {}  # squad_id → {state, target_tag, stalker_tags, ...}
        self._snipe_squad_cooldown: dict[str, int] = {}  # squad_id → last_commit_frame

        # Blink chase state (group chase system)
        self._chase_committed: dict[int, int] = {}  # tag → expiry_frame (skip-set for per-unit micro)
        self._chase_state: dict[str, dict] = {}  # squad_id → {target_tag, stalker_tags, ...}

        # Blink focus-fire state (group focus-fire system)
        self._focus_committed: dict[int, int] = {}  # tag → expiry_frame (skip-set for per-unit micro)
        self._focus_state: dict[str, dict] = {}  # squad_id → {target_tag, stalker_tags, ...}

        # Mass Recall state (Nexus emergency evacuation)
        self._mass_recall_last_cast_time: float = -999.0  # Game time of last mass recall cast (global 130s cooldown)
        self._mass_recall_pending: bool = False  # True when units should cluster before recall fires

        # Detection cannon state (per-base Pylon+Cannon behind mineral lines)
        self._detection_cannon_state: dict[int, str] = {}  # nexus tag → state
        self._detection_cannon_triggered: bool = False  # Sticky: True once a harass threat (Banshee/DT/Oracle/Widow Mine/Dark Shrine) is ever seen

        # Belief layer (Phase 1: Composition Belief, Phase 2: Strategy Belief, Phase 4: Opponent Belief)
        enable_strategy = self.config.get("Belief", {}).get("enable_strategy", True)
        enable_opponent = self.config.get("Belief", {}).get("enable_opponent", True)
        self._belief_updater: BeliefUpdater = BeliefUpdater(
            enable_strategy=enable_strategy, enable_opponent=enable_opponent
        )
        self.belief_state: BeliefState = create_empty_belief_state()



    async def on_start(self) -> None:
        """
        Runs at the start of the game. Sets up expansions, initial flags,
        and calculates where to take expansions first.
        """
        await super().on_start()

        # Load debug flag from config
        self.debug = self.config.get("BotDebug", False)

        self._patch_bunker_weapon_data()

        # Debug on start
        self.map_data: MapData  = self.mediator.get_map_data_object
        
        # Debug control - set to True to enable debug output for disruptor nova system
        debug_disruptor_nova = False
        
        self.use_disruptor_nova = UseDisruptorNova(mediator=self.mediator, bot=self, debug_output=debug_disruptor_nova)
        self.nova_manager = NovaManager(bot=self, mediator=self.mediator, debug_output=debug_disruptor_nova)  # Initialize the NovaManager
        
        # Create choke grid for O(1) formation skip detection
        self.choke_grid = create_choke_grid(self)
        # Pre-filtered set of choke tiles from narrow passages only (width < CHOKE_MAX_WIDTH)
        self.narrow_choke_points: dict[Point2, float] = create_narrow_choke_points(self)
        # Lazily-populated cache of raycast-refined chokes (RefinedChoke per choke tile).
        # Static terrain never changes, so no TTL — each choke refined at most once per game.
        self.refined_choke_points: dict[Point2, object] = {}
        # Pre-refine all narrow chokes at on_start so they're visible immediately
        # when the squad approaches one (not just during active combat)
        refine_all_chokes(self)
        # Dynamic choke cache: squad_id → (frame, RefinedChoke | None).
        # Scanned every DYNAMIC_CHOKE_SCAN_INTERVAL frames per squad.
        self.dynamic_choke_cache: dict[int, tuple[int, object]] = {}

        self.current_base_target = self.enemy_start_locations[0]

        # Sort expansions by proximity to the enemy's start
        self.expansion_locations_list.sort(
            key=lambda loc: cy_distance_to(loc, self.enemy_start_locations[0])
        )
        self.scout_targets = self.expansion_locations_list
        


        # Reserve expansions and set flags
        self.natural_expansion: Point2 = self.mediator.get_own_nat
        self.expansions_generator = cycle(self.expansion_locations_list)

        # Set gatekeeping position based on enemy race
        self.gatekeeping_pos = None  # Initialize
        
        if self.enemy_race in {Race.Zerg, Race.Random}:
            # PvZ: use natural choke gatekeeping position
            if self.mediator.get_pvz_nat_gatekeeping_pos is not None:
                self.gatekeeping_pos = self.mediator.get_pvz_nat_gatekeeping_pos
            else:
                self.gatekeeping_pos = self.natural_expansion.towards(self.game_info.map_center, 6)
        
        elif self.enemy_race == Race.Protoss:
            # PvP: use main ramp wall warp-in position (blocks the wall gap)
            warpin_pos = self.main_base_ramp.protoss_wall_warpin
            if warpin_pos is not None:
                self.gatekeeping_pos = warpin_pos
        
        # Set rally point based on gatekeeping position and opponent race
        if self.gatekeeping_pos is not None:
            if self.enemy_race == Race.Protoss:
                # PvP: rally inside base (behind main ramp gatekeeper)
                self.rally_point = self.gatekeeping_pos.towards(self.start_location, 5)
            else:
                # PvZ/Random: rally behind natural choke gatekeeper
                self.rally_point = self.gatekeeping_pos.towards(self.natural_expansion, 5)
        else:
            self.rally_point = self.natural_expansion.towards(self.game_info.map_center, 5)
        
        # Compute rush distance tier for ling rush detection (used in reactions.py)
        from bot.intel import compute_rush_distance_tier
        self.rush_distance_tier = compute_rush_distance_tier(self)
        
        # Load opponent profiles for cross-game priors (Phase 4)
        # Must run before print_startup_report so the prior summary can be included.
        self._belief_updater.load_opponent()

        # Print startup report with all initial game info
        print_startup_report(self)

        # Initialize telemetry context (binds match_id, env, game info)
        init_context(self)

        # Register reaction handlers (maps StrategyCategory + level2 → handler fn)
        self.reaction_manager.register_handlers()

        # Greet opponent with prior info (or just GLHF if unknown)
        await self._send_opponent_greeting()

    def _patch_bunker_weapon_data(self) -> None:
        """Give enemy bunkers a marine-quad weapon so combat sims see their DPS.

        The API reports no DPS/range/passengers for enemy bunkers (passengers are
        only visible for own units — python-sc2 unit.py TODO), so the combat
        simulator would treat them as weaponless 400 HP shells. We copy the
        Marine weapon proto into the Bunker's type data, scaled to a full
        garrison, and treat every enemy bunker as occupied.
        """
        from bot.constants import BUNKER_SIM_GARRISON_COUNT, BUNKER_SIM_RANGE

        bunker_data = self.game_data.units[UnitTypeId.BUNKER.value]
        if bunker_data._proto.weapons:
            return  # already patched (e.g. on_start re-entry) — don't stack garrisons

        marine_weapon = self.game_data.units[UnitTypeId.MARINE.value]._proto.weapons[0]
        weapon = bunker_data._proto.weapons.add()
        weapon.CopyFrom(marine_weapon)
        weapon.attacks = marine_weapon.attacks * BUNKER_SIM_GARRISON_COUNT
        weapon.range = BUNKER_SIM_RANGE

    async def _send_opponent_greeting(self) -> None:
        """Send a greeting at game start with opponent prior info if available."""
        opponent_id = getattr(self, 'opponent_id', None)
        enemy_race = self.enemy_race.name if hasattr(self, 'enemy_race') else "Unknown"

        if not opponent_id:
            # Local game — no opponent ID
            return

        # Check if we have a prior for this opponent
        if self._belief_updater._opponent is None:
            await self.chat_send(f"Hey {opponent_id[:8]}, GLHF!")
            return

        prior = self._belief_updater._opponent.get_prior(opponent_id, enemy_race)
        from bot.constants import StrategyCategory
        total_games = sum(prior.values()) - 4.0  # Subtract baseline [1,1,1,1]

        if total_games < 1:
            # First time meeting — just say GLHF
            await self.chat_send(f"Hey {opponent_id[:8]}, first time! GLHF!")
        else:
            # Known opponent — show tendency in natural language
            cat_names = {
                StrategyCategory.CHEESE: "cheese",
                StrategyCategory.ALL_IN: "all-in",
                StrategyCategory.TIMING_ATTACK: "timing attack",
                StrategyCategory.MACRO: "macro",
            }
            best_cat = max(prior.keys(), key=lambda k: prior[k])
            top_pct = int(prior[best_cat] / sum(prior.values()) * 100)
            games = int(total_games)
            if games >= 3:
                msg = (f"We've played {games} games, so I think you're "
                       f"{top_pct}% likely to go {cat_names[best_cat]}. GLHF!")
            else:
                msg = (f"We've only played {games} game{'s' if games > 1 else ''}, "
                       f"but I think you're {top_pct}% likely to go "
                       f"{cat_names[best_cat]}. GLHF!")
            await self.chat_send(msg)

    def _force_complete_build(self, reason: str) -> None:
        """Mark the opening build order complete and hand off to macro.

        All three exit valves (stall detector, mineral bank, gas bank) funnel
        here so the recovery path is identical to the pre-existing fail-safe
        and each firing is greppable from logs/telemetry.
        """
        if self.build_order_runner.build_completed:
            return
        self.build_order_runner.set_build_completed()
        print(f"Build order force-completed at {self.time:.1f}s: {reason}")
        log_transition(
            subsystem="build_runner",
            action="force_complete",
            reason=reason,
            key="build_runner_state",
            value="force_completed",
            _ts=self.time,
        )

    def _current_gate_supply(self, runner) -> int:
        """Supply gate of the step the runner is actually working on.

        The look-ahead commits to a temporary step (`_temporary_build_step`)
        and freezes there; if that step is gate-blocked it's the same
        deadlock. Falls back to the current step, then 0 (never blocks).
        """
        idx = runner._temporary_build_step if runner._temporary_build_step != -1 else runner.build_step
        if 0 <= idx < len(runner.build_order):
            return runner.build_order[idx].start_at_supply
        return 0

    async def on_step(self, iteration: int) -> None:
        """
        Main loop executed each game step. Calls out to macro, combat, scouting,
        and reaction modules to handle behavior.
        """
        await super(PiG_Bot, self).on_step(iteration)
        
        # Render choke points on map (debug only, no-op when debug=False)
        render_narrow_choke_points(self)
        render_refined_choke_points(self)
        render_nexus_ability_debug(self)
        
        # Print periodic intel report (every 30 game seconds)
        print_periodic_intel_report(self, iteration)
        
        # Send replay tags (for filtering replays later)
        for tag in get_replay_tags_to_send(self):
            await self.chat_send(f"Tag: {tag}")
        
        # Send pending strategy detection chat (set by strategy_belief.py / strategy_detect.py)
        if hasattr(self, '_strategy_chat_pending') and self._strategy_chat_pending:
            await self.chat_send(self._strategy_chat_pending)
            self._strategy_chat_pending = None
        
        # Update performance metrics (SQ tracking)
        self.performance_monitor.update(iteration, self)
        
        # Dynamic gas worker management (logic in macro.py)
        self.register_behavior(Mining(
            workers_per_gas=get_optimal_gas_workers(self),
            keep_safe=self.reaction_manager.keep_workers_safe
        )) 

        # Filter expired ghosts (age >= 30s) from cached enemy army;
        # UnitCacheManager retains units indefinitely but they're stale after 30s
        from bot.constants import MEMORY_EXPIRY_TIME
        self.enemy_army = self.mediator.get_cached_enemy_army.filter(
            lambda u: u.age < MEMORY_EXPIRY_TIME
        )
        
        # Update enemy intel tracking (for combat sim trust)
        update_enemy_intel_tracking(self)

        # Update belief layer (Phase 1: Composition Belief)
        # Feature-gated: when disabled, belief_state stays at defaults
        if self.config.get("Belief", {}).get("enable_composition", False):
            self.belief_state = self._belief_updater.update(
                bot=self,
                cached_army=self.mediator.get_cached_enemy_army,
                visible_structures=self.enemy_structures,
                game_time=self.time,
            )

        # Scout VOI staleness tracking (Phase 3, Part 2)
        if self.config.get("Belief", {}).get("enable_scout_voi", False):
            from bot.belief.scout_voi import update_location_sightings
            update_location_sightings(self)

        # Reset per-frame flags (set by combat during this step)
        self._blind_ramp_target = None
        self._cloaked_threat_positions = []

        # Retrieve roles (initial fetch)
        main_army = self.mediator.get_units_from_role(role=UnitRole.ATTACKING)
        warp_prism = self.mediator.get_units_from_role(role=UnitRole.DROP_SHIP)
        gatekeeper = self.mediator.get_units_from_role(role=UnitRole.GATE_KEEPER)

        # Always run combat-oriented threat detection first
        # This ensures we're always responding to immediate threats regardless of build order status
        if main_army:  # Only run detection if we have an army to use for defense
            threat_detection(self, main_army)
            self.own_army = self.mediator.get_own_army


        # Early game logic
        if not self.build_order_runner.build_completed:
            from bot.intel import track_enemy_timings
            track_enemy_timings(self)  # Always track timings/speed during build order phase
            self.register_behavior(RestorePower()) # Restore power to depowered buildings

            # Route prediction → handler via ReactionManager
            self.reaction_manager.update(self)
            self.reaction_manager.execute(self)
        else:
            self.reaction_manager.execute(self)

            # Detect enemy repair of Bunkers/PFs every frame (cheap, any race)
            update_repair_detection(self)

            # Macro calls (only run if build order is complete)
            await handle_macro(
                bot=self,
                main_army=main_army,
                warp_prism=warp_prism,
                freeflow=get_freeflow_mode(self),  # Dynamic calculation
            )
            
            # Use energy recharge on priority targets
            use_recharge(self, main_army)
            
            # Use chronoboost on production/research structures
            use_chronoboost(self)
            
            

        # Manage defensive unit roles (return them to attacking when threats are cleared)
        manage_defensive_unit_roles(self)
        
        # CRITICAL: Refresh main_army after role management to get current state
        main_army = self.mediator.get_units_from_role(role=UnitRole.ATTACKING)
        
        # Create squads ONCE after all role management is complete (ARES pattern)
        from bot.constants import ATTACKING_SQUAD_RADIUS
        squads: list[UnitSquad] = self.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=ATTACKING_SQUAD_RADIUS)
        
        # Determine attack target and handle attack decision logic
        if main_army and squads:
            self.main_army_position = self.mediator.get_position_of_main_squad(role=UnitRole.ATTACKING)
            # handle_attack_toggles decides target (attack/retreat/rally) and sets _commenced_attack flag
            final_target = handle_attack_toggles(self, main_army, attack_target(self, main_army_position=self.main_army_position), squads=squads)
            control_main_army(self, main_army, final_target, squads)
        elif main_army:
            # Fallback: use main army center if no squads can be formed
            self.main_army_position = main_army.center
            # Still run decision logic to set flags
            handle_attack_toggles(self, main_army, attack_target(self, main_army_position=self.main_army_position), squads=None)

        # Control BASE_DEFENDER units every frame (separated from allocation in threat_detection)
        control_defenders(self)

        # Expansion debug overlay — after combat renders so it chains off _debug_y
        render_expansion_debug(self)

        # Update active novas every frame (critical for trajectory correction)
        if hasattr(self, 'nova_manager') and self.nova_manager:
            try:
                # Get all enemy units for nova targeting updates
                enemy_units = self.enemy_units.filter(lambda u: not u.is_memory and not u.is_structure)
                self.nova_manager.update(enemy_units, main_army if main_army else self.units)
            except Exception as e:
                print(f"ERROR updating nova_manager: {e}")
        
        # Warp Prism following main army
        warp_prism_follower(self, warp_prism, main_army)

        # Army cohesion is now handled proactively by the main squad coordination system

        # Scouting actions
        from bot.managers.scouting import (
            control_build_runner_scout, control_hallucination_scout,
            control_hallucination_scouts, control_observers, control_worker_scout,
        )
        observers = self.units.filter(
            lambda u: u.type_id in {UnitTypeId.OBSERVER, UnitTypeId.OBSERVERSIEGEMODE}
        )
        control_observers(self, observers, main_army)
        control_worker_scout(self)  # Early game worker scout when intel is stale
        control_build_runner_scout(self)  # Safe pathing for build-order scout probe
        control_hallucination_scout(self)  # Cast hallucinated Phoenix when blind and low on observers
        control_hallucination_scouts(self)  # Path hallucinated Phoenixes to hunt targets

       
        
        # Update game state based on game time
        current_time = self.time
        
        # ===== Build runner exit valves =====
        # A stuck build runner blocks handle_macro() entirely, so these run in
        # ALL game states — not just early game.
        if not self.build_order_runner.build_completed:
            # Fast valve: no step progress for N seconds while the next step's
            # supply gate is above us. If the gate were passed, the wait would
            # be affordability (e.g. banking for an expansion) — legitimate.
            from bot.constants import BUILD_STALL_FORCE_COMPLETE_SECONDS
            runner = self.build_order_runner
            progress_key = (runner.build_step, runner._temporary_build_step, runner.current_step_started)
            if progress_key != self._last_build_progress_key:
                self._stall_start_time = self.time
                self._last_build_progress_key = progress_key
            elif (not runner.current_step_started
                  and runner.build_order
                  and self.supply_used < self._current_gate_supply(runner)
                  and self.time - self._stall_start_time > BUILD_STALL_FORCE_COMPLETE_SECONDS):
                self._force_complete_build(reason=f"stalled {self.time - self._stall_start_time:.0f}s below supply gate {self._current_gate_supply(runner)} at {self.supply_used} supply")

            # Backstop valves: a *started* step that got stuck (placement
            # failure, builder died) banks money because nothing spends it.
            from bot.constants import BUILD_FORCE_COMPLETE_MINERALS, BUILD_FORCE_COMPLETE_VESPENE
            if self.minerals > BUILD_FORCE_COMPLETE_MINERALS:
                self._force_complete_build(reason=f"banked {self.minerals} minerals")
            elif self.vespene > BUILD_FORCE_COMPLETE_VESPENE:
                self._force_complete_build(reason=f"banked {self.vespene} vespene")
        
        if current_time >= self.mid_game_threshold:
            self.game_state = 2  # late game
        elif current_time >= self.early_game_threshold:
            self.game_state = 1  # mid game
            if gatekeeper:
                for zealot in gatekeeper:
                    self.mediator.clear_role(tag=zealot.tag)
                    self.mediator.assign_role(tag=zealot.tag, role=UnitRole.ATTACKING)
        else:
            self.game_state = 0  # early game
            # Gatekeeper for Zerg and Protoss (if position exists)
            if self.enemy_race in {Race.Zerg, Race.Random, Race.Protoss} and self.gatekeeping_pos is not None:
                if not gatekeeper:
                    zealots = self.units(UnitTypeId.ZEALOT).ready
                    if zealots:
                        tag = zealots.first.tag
                        self.mediator.clear_role(tag=tag)
                        self.mediator.assign_role(tag=tag, role=UnitRole.GATE_KEEPER)

                else:
                    # Keep gatekeeper active throughout early game regardless of attack status
                    gatekeeper_control(self, gatekeeper)
                    
                    
    
    async def on_building_construction_complete(self, unit: Unit) -> None:
        """
        Called whenever a new building is completed. 
        """
        await super().on_building_construction_complete(unit)
        if unit.type_id == UnitTypeId.GATEWAY:
            if self.rally_point:
                unit(AbilityId.RALLY_BUILDING, self.rally_point)
        elif unit.type_id == UnitTypeId.ROBOTICSFACILITY:
            # Rally robo units to starting townhall so they don't wander
            unit(AbilityId.RALLY_BUILDING, self.start_location)
            

    async def on_unit_created(self, unit: Unit) -> None:
        """
        Called whenever a new unit spawns. Assign roles based on type.
        """
        await super().on_unit_created(unit)
        if unit.type_id in ALL_STRUCTURES or unit.type_id in WORKER_TYPES:
            return

        if unit.type_id == UnitTypeId.OBSERVER:
            # Waterfall: army (highest priority) → primary → patrol
            if self.observer_assignments["army"] is None:
                self.observer_assignments["army"] = unit.tag
                self.mediator.assign_role(tag=unit.tag, role=UnitRole.CONTROL_GROUP_EIGHT)
            elif self.observer_assignments["primary"] is None:
                self.observer_assignments["primary"] = unit.tag
                self.mediator.assign_role(tag=unit.tag, role=UnitRole.SCOUTING)
            else:
                self.observer_assignments["patrol"].append(unit.tag)
                self.mediator.assign_role(tag=unit.tag, role=UnitRole.CONTROL_GROUP_NINE)
            return
            
        if unit.type_id == UnitTypeId.WARPPRISM:
            self.mediator.assign_role(tag=unit.tag, role=UnitRole.DROP_SHIP)
            unit.move(Point2(self.rally_point))
            return
        

        if unit.type_id == UnitTypeId.DISRUPTORPHASED:
            # When a DISRUPTORPHASED unit (the nova) is created, send it to the NovaManager
            self.nova_manager.add_nova(unit)
            return

        # Hallucinated units are scouts — keep them out of army squads
        if unit.is_hallucination:
            halu_role = self._halu_scout_roles.get(unit.tag, self._halu_scout_pending_role)
            if halu_role == "high_ground" or (not halu_role and self._blind_ramp_target):
                self.mediator.assign_role(tag=unit.tag, role=UnitRole.HIGH_GROUND_SPOTTER)
            else:
                self.mediator.assign_role(tag=unit.tag, role=UnitRole.SCOUTING)
            return

        # Default: Attacking role
        self.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)
        unit.attack(Point2(self.rally_point))

    async def on_unit_destroyed(self, unit_tag: int) -> None:
        """Track when units get destroyed."""
        await super(PiG_Bot, self).on_unit_destroyed(unit_tag)
        
        # Record enemy destruction in belief layer (ground truth — P(exists) → 0.0)
        # Only enemy tags matter; friendly tags would bloat _destroyed with no-ops
        if self._belief_updater.is_known_enemy_tag(unit_tag):
            self._belief_updater.record_destruction(unit_tag)
        
        # Handle observer reassignment if destroyed (army is highest priority)
        if unit_tag == self.observer_assignments.get("army"):
            self.observer_assignments["army"] = None
            # Promote primary → army, then backfill primary from patrol
            if self.observer_assignments["primary"]:
                promoted = self.observer_assignments["primary"]
                self.observer_assignments["army"] = promoted
                self.observer_assignments["primary"] = None
                if promoted in {u.tag for u in self.units}:
                    self.mediator.clear_role(tag=promoted)
                    self.mediator.assign_role(tag=promoted, role=UnitRole.CONTROL_GROUP_EIGHT)
                # Backfill primary from patrol
                if self.observer_assignments["patrol"]:
                    backfill = self.observer_assignments["patrol"].pop(0)
                    self.observer_assignments["primary"] = backfill
                    if backfill in {u.tag for u in self.units}:
                        self.mediator.clear_role(tag=backfill)
                        self.mediator.assign_role(tag=backfill, role=UnitRole.SCOUTING)
            elif self.observer_assignments["patrol"]:
                promoted = self.observer_assignments["patrol"].pop(0)
                self.observer_assignments["army"] = promoted
                if promoted in {u.tag for u in self.units}:
                    self.mediator.clear_role(tag=promoted)
                    self.mediator.assign_role(tag=promoted, role=UnitRole.CONTROL_GROUP_EIGHT)
        
        elif unit_tag == self.observer_assignments.get("primary"):
            self.observer_assignments["primary"] = None
            # Backfill primary from patrol
            if self.observer_assignments["patrol"]:
                promoted = self.observer_assignments["patrol"].pop(0)
                self.observer_assignments["primary"] = promoted
                if promoted in {u.tag for u in self.units}:
                    self.mediator.clear_role(tag=promoted)
                    self.mediator.assign_role(tag=promoted, role=UnitRole.SCOUTING)
        
        elif unit_tag in self.observer_assignments.get("patrol", []):
            self.observer_assignments["patrol"].remove(unit_tag)
            
        # Handle worker scout death - check if destroyed unit was in SCOUTING role
        scouting_tags = self.mediator.get_unit_role_dict.get(UnitRole.SCOUTING, set())
        if unit_tag in scouting_tags:
            # Clear the role and reset the flag so a new scout can be sent
            self.mediator.clear_role(tag=unit_tag)
            self._worker_scout_sent_this_stale_period = False
            
        # Clean up blink snipe state if a committed stalker dies
        self._snipe_committed.pop(unit_tag, None)
        # Also remove from any active snipe's stalker_tags set
        for sid, info in list(self._snipe_state.items()):
            if unit_tag in info["stalker_tags"]:
                info["stalker_tags"].discard(unit_tag)
                if not info["stalker_tags"]:
                    del self._snipe_state[sid]

        # Clean up blink chase state if a committed stalker dies
        self._chase_committed.pop(unit_tag, None)
        for sid, info in list(self._chase_state.items()):
            if unit_tag in info["stalker_tags"]:
                info["stalker_tags"].discard(unit_tag)
                if not info["stalker_tags"]:
                    del self._chase_state[sid]

        # Clean up blink focus-fire state if a committed stalker dies
        self._focus_committed.pop(unit_tag, None)
        for sid, info in list(self._focus_state.items()):
            if unit_tag in info["stalker_tags"]:
                info["stalker_tags"].discard(unit_tag)
                if not info["stalker_tags"]:
                    del self._focus_state[sid]

        # Clean up observer targets if needed
        if unit_tag in self.observer_targets:
            del self.observer_targets[unit_tag]

        # Clean up detection assignments if observer destroyed
        if unit_tag in self._observer_detection_assignments:
            del self._observer_detection_assignments[unit_tag]
        # Also remove if the observed enemy target was destroyed
        self._observer_detection_assignments = {
            k: v for k, v in self._observer_detection_assignments.items() if v != unit_tag
        }

    async def on_unit_type_changed(self, unit: Unit, previous_type: UnitTypeId) -> None:
        """Called when a unit changes type, like Disruptor firing a Nova."""
        await super(PiG_Bot, self).on_unit_type_changed(unit, previous_type)
        
        # Detect when a Disruptor fires a Nova (it changes type temporarily)
        if previous_type == UnitTypeId.DISRUPTOR and unit.type_id == UnitTypeId.DISRUPTORPHASED:
            # Add this nova to our manager for tracking
            self.nova_manager.add_nova(unit)
            
    async def on_unit_took_damage(self, unit: Unit, amount_damage_taken: float) -> None:
        """
        Allows us to cancel building structures if they're badly damaged, 
        preventing resource waste.
        """
        await super().on_unit_took_damage(unit, amount_damage_taken)

        if unit.type_id not in ALL_STRUCTURES:
            return
        compare_health = max(50.0, unit.health_max * 0.09)
        if unit.health < compare_health:
            self.mediator.cancel_structure(structure=unit)

    async def on_end(self, game_result: Result) -> None:
        """
        Called at the end of the game - prints performance report and emits telemetry.
        """
        print_end_game_report(
            performance_monitor=self.performance_monitor,
            game_result=game_result,
            game_time=self.time,
            idle_worker_time=self.state.score.idle_worker_time,
            idle_production_time=self.state.score.idle_production_time
        )
        
        # Emit single match record (performance + rush detection data)
        emit_match_record(
            bot=self,
            game_result=game_result,
            game_time=self.time,
            idle_worker_time=self.state.score.idle_worker_time,
            idle_production_time=self.state.score.idle_production_time
        )
        
        # Save opponent profiles for cross-game priors (Phase 4).
        # Records the OBSERVED category (evidence-anchored facts from
        # classify_observed_game) — never the model's prediction.
        self._belief_updater.save_opponent(
            bot=self,
            opponent_id=getattr(self, 'opponent_id', None),
            enemy_race=self.enemy_race.name,
            game_result=game_result,
        )
        
        # Reset telemetry state between games
        telemetry_reset()

   
    

   
