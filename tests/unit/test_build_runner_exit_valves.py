"""Build runner exit valves: activity-based deadlock + bank backstops.

Purpose: pin the exit conditions agreed in plan/build_runner_gate_fix.md —
    a deadlock is NOTHING in flight (no structures warping, no units
    materializing, no train/research orders, no workers en route) while
    below an unreachable supply gate with more than probe-cost banked.
    Legitimate waits (banking for an expansion, mid-warp production,
    spaced-out probe production) always have an activity bucket occupied
    or the gate passed, so they can never trip the valve.
Key Decisions: the valve is ground-truth driven — it reads ARES/python-sc2
    state directly (structures.not_ready, units.not_ready, structure
    orders, building tracker) instead of inferring stuckness from banks,
    supply movement, or timers. The 72s incident (false force-complete at
    19 supply while a Gateway was mid-build and probes spaced 25-40s apart)
    is the headline regression test.
Limitations: cannot cover set_build_completed()'s role cleanup or macro's
    post-handoff behavior; that's replay territory (see telemetry grep).
"""

from types import SimpleNamespace

from bot.bot import PiG_Bot


class FakeStep:
    def __init__(self, start_at_supply):
        self.start_at_supply = start_at_supply


class FakeRunner:
    """Mimics the ares BuildOrderRunner fields the valves read."""

    def __init__(self, gates=(50,), step_index=0, started=False, completed=False):
        self.build_order = [FakeStep(g) for g in gates]
        self.build_step = step_index
        self._temporary_build_step = -1
        self.current_step_started = started
        self.build_completed = completed

    def set_build_completed(self):
        self.build_completed = True


def _bot(time_=0.0, minerals=0, vespene=0, supply=0,
         warping_structures=0, materializing_units=0,
         structure_orders=False, building_tracker=False):
    """Fake bot exposing the same ARES-visible activity state the real
    valve reads: structures warping, units materializing, any structure
    holding a production/research order, and ARES's worker-en-route dict."""
    return SimpleNamespace(
        time=time_, minerals=minerals, vespene=vespene, supply_used=supply,
        structures=SimpleNamespace(not_ready=[object()] * warping_structures),
        units=SimpleNamespace(not_ready=[object()] * materializing_units),
        _any_structure_orders=structure_orders,
        mediator=SimpleNamespace(
            get_building_tracker_dict={1: {}} if building_tracker else {}
        ),
    )


def _nothing_in_flight(bot) -> bool:
    """Activity-bucket check, mirroring bot.py's condition exactly."""
    return (
        not bot.structures.not_ready
        and not bot.units.not_ready
        and not bot._any_structure_orders
        and not bot.mediator.get_building_tracker_dict
    )


def _valve(bot, runner):
    """The valve block from PiG_Bot.on_step, extracted verbatim for testing."""
    from bot.constants import (
        BUILD_STALL_CONFIRM_SECONDS,
        BUILD_STALL_DEADLOCK_BANK,
        BUILD_FORCE_COMPLETE_MINERALS,
        BUILD_FORCE_COMPLETE_VESPENE,
    )
    if runner.build_completed:
        return False

    nothing_in_flight = _nothing_in_flight(bot)
    below_gate = (runner.build_order
                  and bot.supply_used < PiG_Bot._current_gate_supply(bot, runner))
    if nothing_in_flight and below_gate and bot.minerals > BUILD_STALL_DEADLOCK_BANK:
        if getattr(bot, "_deadlock_since", None) is None:
            bot._deadlock_since = bot.time
        elif bot.time - bot._deadlock_since >= BUILD_STALL_CONFIRM_SECONDS:
            return "deadlock"

    if bot.minerals > BUILD_FORCE_COMPLETE_MINERALS:
        return "minerals"
    if bot.vespene > BUILD_FORCE_COMPLETE_VESPENE:
        return "gas"
    return False


# ===== THE 72s INCIDENT (headline regression test) =====

def test_72s_incident_never_fires():
    """False-fire incident: Gateway mid-warp (36->101s) + probes spaced
    25-40s apart. Structure warping => an activity bucket is occupied =>
    nothing_in_flight is False => valve can never fire, no matter how long
    the below-gate wait lasts."""
    bot = _bot(time_=0.0, supply=19, minerals=60, warping_structures=1)
    runner = FakeRunner(gates=(24,), started=False)
    assert _valve(bot, runner) is False
    # 40s later: gateway still warping, no step progress, below gate.
    # The old timer-based valve fired here; this one must not.
    bot.time = 40.0
    assert _valve(bot, runner) is False


def test_probe_in_production_blocks_firing():
    """Probe materializing in a Nexus order => production activity exists."""
    bot = _bot(time_=0.0, supply=19, minerals=60, structure_orders=True)
    runner = FakeRunner(gates=(24,))
    assert _valve(bot, runner) is False
    bot.time = 60.0  # long below-gate wait, but a probe is in production
    assert _valve(bot, runner) is False


# ===== TRUE DEADLOCKS (must fire) =====

def test_1081_signature_fires():
    """1081: stalled at 44/46 supply, 85 minerals banked, no step started,
    nothing in flight for 49 seconds. Fires after the confirm window."""
    bot = _bot(time_=0.0, supply=44, minerals=85)
    runner = FakeRunner(gates=(48,), started=False)
    assert _valve(bot, runner) is False  # confirm window starts
    bot.time = 5.0  # 5s of continuous nothing-in-flight
    assert _valve(bot, runner) == "deadlock"


def test_1087_signature_fires():
    """1087: economy dead at 29 supply falling, gas banking, mineral-poor.
    The activity valve catches it even before the gas backstop would."""
    bot = _bot(time_=0.0, supply=29, minerals=60, vespene=300)
    runner = FakeRunner(gates=(48,))
    assert _valve(bot, runner) is False
    bot.time = 4.0
    assert _valve(bot, runner) == "deadlock"


def test_supply_blocked_no_pylon_fires():
    """Supply-blocked below gate with no pylon in flight and money to spend:
    the canonical deadlock the PR was written for."""
    bot = _bot(time_=0.0, supply=22, minerals=200)
    runner = FakeRunner(gates=(24,))
    assert _valve(bot, runner) is False
    bot.time = 4.0
    assert _valve(bot, runner) == "deadlock"


# ===== LEGITIMATE WAITS (must never fire) =====

def test_banking_for_expansion_never_fires():
    """The user's explicit requirement: banking minerals for an expansion is
    NOT an exit hatch. Gate passed ("46 EXPAND" at 46 supply), so below_gate
    is False no matter how high the bank climbs or how long the wait."""
    bot = _bot(time_=0.0, supply=46, minerals=350)
    runner = FakeRunner(gates=(46,))
    assert _valve(bot, runner) is False
    bot.time = 120.0
    bot.minerals = 390  # still short of the 400m Nexus
    assert _valve(bot, runner) is False


def test_worker_en_route_blocks_firing():
    """Worker walking to a build spot (ARES building tracker entry) is
    activity — even with nothing warping yet."""
    bot = _bot(time_=0.0, supply=19, minerals=60, building_tracker=True)
    runner = FakeRunner(gates=(24,))
    assert _valve(bot, runner) is False
    bot.time = 60.0
    assert _valve(bot, runner) is False


def test_unit_materializing_blocks_firing():
    """A zealot/stalker mid-warp counts as in-flight activity."""
    bot = _bot(time_=0.0, supply=19, minerals=60, materializing_units=1)
    runner = FakeRunner(gates=(24,))
    assert _valve(bot, runner) is False
    bot.time = 60.0
    assert _valve(bot, runner) is False


def test_empty_bank_below_gate_does_not_fire():
    """Genuinely broke (30 minerals, nothing in flight) below the probe-cost
    threshold: not provably deadlocked — income may resume. The bank clause
    prevents firing on a starved-but-recovering economy."""
    bot = _bot(time_=0.0, supply=19, minerals=30)
    runner = FakeRunner(gates=(24,))
    assert _valve(bot, runner) is False
    bot.time = 60.0
    assert _valve(bot, runner) is False


# ===== BACKSTOP VALVES (unchanged behavior) =====

def test_mineral_bank_backstop():
    """Started-but-stuck step (placement failure): minerals bank, backstop fires."""
    bot = _bot(time_=0.0, minerals=1001, supply=46, warping_structures=1)
    runner = FakeRunner(gates=(48,), started=True)
    assert _valve(bot, runner) == "minerals"


def test_gas_bank_backstop():
    """1087's gas-banked death when activity buckets are occupied (e.g. a
    stuck step mid-build): only the gas backstop can save it."""
    bot = _bot(time_=0.0, minerals=85, vespene=501, supply=29, warping_structures=1)
    runner = FakeRunner(gates=(48,), started=True)
    assert _valve(bot, runner) == "gas"


def test_minerals_takes_precedence_over_gas():
    bot = _bot(time_=0.0, minerals=1200, vespene=600, supply=29)
    runner = FakeRunner(gates=(48,))
    assert _valve(bot, runner) == "minerals"


def test_completed_runner_never_fires():
    """After force-complete the valves must be inert — macro owns the game."""
    bot = _bot(time_=0.0, minerals=2000, vespene=900)
    runner = FakeRunner(completed=True)
    assert _valve(bot, runner) is False


# ===== GATE HELPER =====

def test_gate_helper_index_bounds():
    """Out-of-range indices (completed tail, parser edge) fall back to 0 —
    gate 0 can never block, so the valve degrades to bank-only."""
    runner = FakeRunner(gates=(48,), step_index=5)  # index past end
    bot = SimpleNamespace()
    assert PiG_Bot._current_gate_supply(bot, runner) == 0
    runner.build_order = []  # empty build order
    assert PiG_Bot._current_gate_supply(bot, runner) == 0
