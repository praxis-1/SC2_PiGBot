"""Gatekeeper resting-spot selection: the designed anchor, with build-plan foresight.

Purpose: pin the gatekeeper behaviour agreed for the resting spot - the zealot rests
    at one designed spot (gate + towards base * GATEKEEPER_MOVE_DISTANCE), close
    enough to be back on the gap before a detected enemy arrives. When a planned or
    built building sits on that spot, it steps to the clear spot closest to the gate
    - never onto a tile the probe will build on, never re-asserting its order every
    frame (the per-frame order is what made it fight the probe for the build spot).
    While enemies are near it walks onto the gap and holds there.
Key Decisions: the helpers in bot/combat/combat.py are pure functions over positions
    and obstacle lists, so tests drive them with fake structures and a
    SimpleNamespace bot - no game, no SC2 binary. Planned buildings come from ARES
    placements_dict (the parsed protoss_building_placements.yml).
Limitations: does not cover the wall layout that puts buildings near the resting
    spot; that is a natural_wall_manager / placement-plan concern.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from bot.combat.combat import (
    _clearance_to_obstacles,
    _find_resting_spot,
    _gatekeeper_obstacles,
    _order_move,
    gatekeeper_control,
)
from bot.constants import (
    GATEKEEPER_ARRIVAL_DISTANCE,
    GATEKEEPER_CLEARANCE_MARGIN,
    GATEKEEPER_PARK_MAX_DISTANCE,
    GATEKEEPER_PARK_MIN_DISTANCE,
    GATEKEEPER_ZEALOT_RADIUS,
)
from ares.consts import BuildingSize
from cython_extensions import cy_distance_to
from sc2.data import Race
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2

GATE = Point2((10, 10))
BASE = Point2((10, 30))
ANCHOR = Point2((10, 13))  # gate + towards base * GATEKEEPER_MOVE_DISTANCE


class FakeStructure:
    """Mimics the python-sc2 structure fields the clearance math reads."""

    def __init__(self, x, y, footprint_radius=1.5):
        self.position = Point2((x, y))
        self.footprint_radius = footprint_radius
        self.radius = footprint_radius


class FakeUnits:
    """Mimics python-sc2 Units.closer_than."""

    def __init__(self, units=()):
        self.units = list(units)

    def closer_than(self, radius, position):
        return [unit for unit in self.units if cy_distance_to(unit.position, position) < radius]


class FakeGatekeeper:
    """Mimics the python-sc2 Unit surface the gatekeeper code touches."""

    def __init__(self, position, order_target=None):
        self.position = Point2(position)
        self.order_target = order_target
        self.orders = []
        self.moves = []
        self.holds = 0

    def move(self, target):
        self.moves.append(target)
        self.order_target = target
        self.orders = [SimpleNamespace(ability=SimpleNamespace(id=AbilityId.MOVE))]

    def __call__(self, ability, **kwargs):
        self.orders = [SimpleNamespace(ability=SimpleNamespace(id=ability))]
        self.holds += 1


class FakeMediator:
    """Mirrors the real ManagerMediator members the gatekeeper reads.

    get_placements_dict is a property on the real mediator - a method-shaped fake
    would hide an accidental `()` call bug in the gatekeeper code.
    """

    def __init__(self, grid, enemies, placements, placement_calls):
        self._grid = grid
        self._enemies = list(enemies)
        self._placements = placements or {}
        self._placement_calls = placement_calls

    @property
    def get_ground_grid(self):
        return self._grid

    def get_units_in_range(self, **kwargs):
        return [self._enemies]

    @property
    def get_placements_dict(self):
        self._placement_calls["count"] += 1
        return self._placements


def _bot(structures=(), enemies=(), gate_pos=GATE, walkable=True, placements=None):
    """Bot double: our base sits straight behind the gate, so 'towards base' is +y."""
    grid = np.ones((60, 60), dtype=np.float32)
    if not walkable:
        grid[:] = np.inf

    placement_calls = {"count": 0}
    mediator = FakeMediator(grid, enemies, placements, placement_calls)

    bot = SimpleNamespace(
        gatekeeping_pos=Point2(gate_pos),
        start_location=BASE,
        natural_expansion=BASE,
        enemy_race=Race.Zerg,
        structures=FakeUnits(structures),
        enemy_structures=FakeUnits(),
        mediator=mediator,
    )
    bot.placement_calls = placement_calls
    return bot


def _pylon_on_the_anchor():
    """A placements_dict with one planned pylon sitting exactly on the resting spot."""
    return {
        BASE: {
            BuildingSize.TWO_BY_TWO: {
                ANCHOR: {"available": True, "taken": False},
            }
        }
    }


# ===== CLEARANCE =====


def test_clearance_accounts_for_the_whole_footprint():
    obstacles = [(Point2((10, 13)), 1.5)]
    # 3 tiles from a 1.5-radius building, minus the zealot's own radius
    assert _clearance_to_obstacles(Point2((10, 16)), obstacles) == pytest.approx(0.5)
    # the building centre is deep inside its footprint
    assert _clearance_to_obstacles(Point2((10, 13)), obstacles) < 0


# ===== RESTING SPOT =====


def test_resting_spot_is_the_designed_anchor_when_nothing_is_planned():
    bot = _bot()
    spot = _find_resting_spot(bot, GATE, BASE, _gatekeeper_obstacles(bot, GATE))
    assert spot == ANCHOR


def test_planned_building_on_the_anchor_moves_the_spot_closer_to_the_gate():
    """Ultralove case: the pylon is planned where the zealot rests. The fallback is
    the clear spot closest to the gate - on the gap side of the pylon, never on it."""
    bot = _bot(placements=_pylon_on_the_anchor())
    spot = _find_resting_spot(bot, GATE, BASE, _gatekeeper_obstacles(bot, GATE))

    assert spot is not None
    assert spot != ANCHOR
    # closest to the gate, so the pylon is never between the zealot and the gap
    assert cy_distance_to(spot, GATE) < cy_distance_to(ANCHOR, GATE)
    assert GATEKEEPER_PARK_MIN_DISTANCE <= cy_distance_to(spot, GATE) <= GATEKEEPER_PARK_MAX_DISTANCE
    # clear of the planned pylon's footprint plus the zealot's own body
    assert cy_distance_to(spot, ANCHOR) >= 1.0 + GATEKEEPER_ZEALOT_RADIUS


def test_planned_buildings_are_read_once_per_game():
    bot = _bot(placements=_pylon_on_the_anchor())

    _gatekeeper_obstacles(bot, GATE)
    _gatekeeper_obstacles(bot, GATE)

    assert bot.placement_calls["count"] == 1


def test_built_building_on_the_anchor_moves_the_spot_closer_to_the_gate():
    bot = _bot(structures=[FakeStructure(10, 13, footprint_radius=1.0)])
    spot = _find_resting_spot(bot, GATE, BASE, _gatekeeper_obstacles(bot, GATE))

    assert spot is not None
    assert cy_distance_to(spot, GATE) < cy_distance_to(ANCHOR, GATE)
    assert cy_distance_to(spot, ANCHOR) >= 1.0 + GATEKEEPER_ZEALOT_RADIUS


def test_no_resting_spot_when_the_area_is_not_walkable():
    bot = _bot(walkable=False)
    assert _find_resting_spot(bot, GATE, BASE, _gatekeeper_obstacles(bot, GATE)) is None


# ===== ORDER CADENCE =====


def test_move_order_is_not_reissued_for_the_same_target():
    gate = FakeGatekeeper((10, 12), order_target=Point2((10, 13)))

    _order_move(gate, Point2((10, 13)))  # already on its way
    assert gate.moves == []

    _order_move(gate, Point2((10, 16)))  # new destination
    assert gate.moves == [Point2((10, 16))]

    gate.position = Point2((10, 16))  # arrived, idle
    gate.order_target = None
    _order_move(gate, Point2((10, 16)))
    assert gate.moves == [Point2((10, 16))]  # no order while standing on the spot


# ===== CONTROL =====


def test_gatekeeper_goes_to_the_gap_when_enemies_are_close():
    """The wall flanks the gap at 1.5 - that must never read as 'the gap is blocked'."""
    wall = [FakeStructure(11.5, 10, footprint_radius=1.5), FakeStructure(8.5, 10, footprint_radius=1.5)]
    bot = _bot(structures=wall, enemies=[FakeStructure(12, 10, footprint_radius=1.0)])
    gate = FakeGatekeeper((10, 13))

    gatekeeper_control(bot, [gate])

    assert gate.moves == [GATE]


def test_gatekeeper_keeps_walking_until_it_is_on_the_gap():
    bot = _bot(enemies=[FakeStructure(12, 10, footprint_radius=1.0)])

    walking = FakeGatekeeper((10, 11.5))  # 1.5 tiles out
    gatekeeper_control(bot, [walking])
    assert walking.moves == [GATE]
    assert walking.holds == 0

    arrived = FakeGatekeeper((10, 10 + GATEKEEPER_ARRIVAL_DISTANCE / 2))
    gatekeeper_control(bot, [arrived])
    assert arrived.moves == []
    assert arrived.holds == 1

    gatekeeper_control(bot, [arrived])  # already holding
    assert arrived.holds == 1


def test_parked_gatekeeper_issues_no_orders():
    bot = _bot()
    gate = FakeGatekeeper(GATE)

    gatekeeper_control(bot, [gate])  # parks at the designed spot
    assert gate.moves == [ANCHOR]

    gatekeeper_control(bot, [gate])  # still on its way - no duplicate order
    assert gate.moves == [ANCHOR]

    gate.position = ANCHOR  # arrived, idle
    gate.order_target = None
    gatekeeper_control(bot, [gate])
    assert gate.moves == [ANCHOR]  # standing on the spot: no order at all
    assert gate.holds == 0


def test_building_landing_on_the_resting_spot_reparks_the_gatekeeper():
    bot = _bot()
    gate = FakeGatekeeper(GATE)

    gatekeeper_control(bot, [gate])  # parks at the designed spot
    assert gate.moves == [ANCHOR]

    # a probe places a building where the zealot was resting
    bot.structures = FakeUnits([FakeStructure(10, 13, footprint_radius=1.0)])
    gate.position = ANCHOR
    gate.order_target = None
    gatekeeper_control(bot, [gate])

    assert len(gate.moves) == 2
    spot = gate.moves[1]
    assert spot != ANCHOR
    assert cy_distance_to(spot, ANCHOR) >= 1.0 + GATEKEEPER_ZEALOT_RADIUS


def test_gatekeeper_holds_when_no_resting_spot_exists():
    # a 5x5 footprint behind the gate covers every candidate spot
    bot = _bot(structures=[FakeStructure(10, 12, footprint_radius=2.5)])
    gate = FakeGatekeeper(GATE)

    gatekeeper_control(bot, [gate])

    assert gate.moves == []
    assert gate.holds == 1
