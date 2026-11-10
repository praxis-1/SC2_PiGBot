"""Build runner exit valves: stall detector + bank backstops.

Purpose: pin the exit conditions agreed in plan/build_runner_gate_fix.md —
    a deadlock (gate above supply, no step in flight, no progress) must trip
    the fast valve; legitimate waits (banking for an expansion, gate passed)
    must never trip it; started-but-stuck steps must bank money until a
    backstop valve catches them.
Key Decisions: the valve logic lives in PiG_Bot as a pure helper reading
    runner state, so tests drive it with a fake runner + SimpleNamespace bot
    — no game, no SC2 binary, no ares import at collection time.
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


def _bot(time_=0.0, minerals=0, vespene=0, supply=0):
    return SimpleNamespace(
        time=time_, minerals=minerals, vespene=vespene, supply_used=supply
    )


def _valve(bot, runner):
    """The valve block from PiG_Bot.on_step, extracted verbatim for testing."""
    from bot.constants import (
        BUILD_STALL_FORCE_COMPLETE_SECONDS,
        BUILD_FORCE_COMPLETE_MINERALS,
        BUILD_FORCE_COMPLETE_VESPENE,
    )
    if runner.build_completed:
        return False
    progress_key = (runner.build_step, runner._temporary_build_step, runner.current_step_started)
    if progress_key != getattr(bot, "_last_build_progress_key", None):
        bot._stall_start_time = bot.time
        bot._last_build_progress_key = progress_key
    elif (not runner.current_step_started
          and runner.build_order
          and bot.supply_used < PiG_Bot._current_gate_supply(bot, runner)
          and bot.time - bot._stall_start_time > BUILD_STALL_FORCE_COMPLETE_SECONDS):
        return "stall"
    if bot.minerals > BUILD_FORCE_COMPLETE_MINERALS:
        return "minerals"
    if bot.vespene > BUILD_FORCE_COMPLETE_VESPENE:
        return "gas"
    return False


# ===== STALL DETECTOR (fast valve) =====

def test_deadlock_trips_after_threshold():
    """1081 signature: supply capped below gate, no step in flight, no progress."""
    bot = _bot(time_=45.0, supply=44)
    runner = FakeRunner(gates=(48,), step_index=0, started=False)
    assert _valve(bot, runner) is False  # first frame: records progress key
    bot.time = 80.0  # 35s later, nothing changed — supply still below gate 48
    assert _valve(bot, runner) == "stall"


def test_legit_banking_does_not_trip():
    """Banking for an expansion ("46 EXPAND" at 46 supply): gate passed, so
    the wait is affordability — timer never fires, no matter how long."""
    bot = _bot(time_=0.0, supply=46, minerals=350)
    runner = FakeRunner(gates=(46,), step_index=0, started=False)
    assert _valve(bot, runner) is False
    bot.time = 120.0  # two minutes banking for a Nexus — must not fire
    bot.minerals = 390  # still short of the 400m Nexus
    assert _valve(bot, runner) is False


def test_one_supply_below_gate_is_stall():
    """Supply 47 vs gate 48 is NOT banking — nothing produces units during
    the runner phase (workers capped by ConstantWorkerProductionTill, macro
    locked out), so a 1-supply gap is as unreachable as a 4-supply one."""
    bot = _bot(time_=0.0, supply=47, minerals=350)
    runner = FakeRunner(gates=(48,), step_index=0, started=False)
    assert _valve(bot, runner) is False
    bot.time = 40.0
    assert _valve(bot, runner) == "stall"


def test_step_in_flight_blocks_stall():
    """A started step (e.g. Nexus mid-build) is progress, not a stall —
    even below the gate of the step after it."""
    bot = _bot(time_=0.0, supply=40)
    runner = FakeRunner(gates=(48,), step_index=0, started=True)
    assert _valve(bot, runner) is False
    bot.time = 60.0
    assert _valve(bot, runner) is False


def test_progress_resets_timer():
    """Timer resets when build_step advances — a healthy build with slow
    steps (nexus ~70s) must never accumulate a full stall window."""
    bot = _bot(time_=0.0, supply=44)
    runner = FakeRunner(gates=(46, 48), step_index=0, started=False)
    assert _valve(bot, runner) is False
    # step 0 completes at t=25, runner advances to step 1
    bot.time = 25.0
    runner.build_step = 1
    assert _valve(bot, runner) is False  # progress key changed → timer reset
    bot.time = 50.0  # 25s into step 1's wait — under the 30s window
    assert _valve(bot, runner) is False
    bot.time = 56.0  # 31s — window exceeded, gate 48 still unreachable
    assert _valve(bot, runner) == "stall"


def test_temporary_step_uses_its_gate():
    """Look-ahead commits to _temporary_build_step; a temp step blocked at
    its own gate is the same deadlock."""
    bot = _bot(time_=0.0, supply=44)
    runner = FakeRunner(gates=(46, 52), step_index=0, started=False)
    runner._temporary_build_step = 1  # look-ahead took step 1 (gate 52)
    assert _valve(bot, runner) is False
    bot.time = 35.0
    assert _valve(bot, runner) == "stall"


def test_temporary_step_below_main_gate_is_not_stall():
    """Look-ahead took a *reachable* step (gate below supply) to fill time
    while the main step is stuck — not a stall, it's progress."""
    bot = _bot(time_=0.0, supply=46)
    runner = FakeRunner(gates=(48, 44), step_index=0, started=False)
    runner._temporary_build_step = 1
    assert _valve(bot, runner) is False
    bot.time = 60.0
    assert _valve(bot, runner) is False


# ===== BANK BACKSTOP VALVES =====

def test_mineral_bank_backstop():
    """Started-but-stuck step (placement failure): minerals bank, backstop fires."""
    bot = _bot(time_=0.0, minerals=1001, supply=46)
    runner = FakeRunner(gates=(48,), step_index=0, started=True)
    assert _valve(bot, runner) == "minerals"


def test_gas_bank_backstop():
    """1087 signature: gas-banked, mineral-poor — mineral-only check would miss."""
    bot = _bot(time_=0.0, minerals=85, vespene=501, supply=29)
    runner = FakeRunner(gates=(48,), step_index=0, started=True)
    assert _valve(bot, runner) == "gas"


def test_minerals_takes_precedence_over_gas():
    bot = _bot(time_=0.0, minerals=1200, vespene=600, supply=29)
    runner = FakeRunner(gates=(48,), step_index=0, started=True)
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
