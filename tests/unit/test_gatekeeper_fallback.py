"""Gatekeeper fallback positioning: footprint-aware parking.

Purpose: pin the parking behaviour agreed for the gatekeeper fallback — the
    zealot must wait off the gap, but off it *with room*: a tile that overlaps a
    building footprint gets the zealot shoved by the engine, and re-ordering it
    every frame turned that shove into a tug-of-war with the probe that was
    building (the ultralove case, where a pylon is planned where the zealot
    rests).
Key Decisions: the helpers in bot/combat/combat.py are pure functions over
    positions and structure lists, so tests drive them with fake structures and a
    SimpleNamespace bot — no game, no SC2 binary.
Limitations: does not cover the wall layout that puts the pylon on the gap; that
    is a natural_wall_manager change, tracked separately.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from bot.combat.combat import (
    _clearance_to_structures,
    _find_parking_spot,
    _gatekeeper_structures,
    _order_move,
    gatekeeper_control,
)
from bot.constants import (
    GATEKEEPER_CLEARANCE_MARGIN,
    GATEKEEPER_DETECTION_RANGE,
    GATEKEEPER_ORDER_TOLERANCE,
    GATEKEEPER_PARK_MAX_DISTANCE,
    GATEKEEPER_PARK_MIN_DISTANCE,
    GATEKEEPER_ZEALOT_RADIUS,
)
from cython_extensions import cy_distance_to
from sc2.data import Race
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2


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


def _bot(structures=(), enemies=(), gate_pos=(10, 10), walkable=True):
    """Bot double: our base sits straight behind the gate, so 'inward' is +y."""
    grid = np.ones((60, 60), dtype=np.float32)
    if not walkable:
        grid[:] = np.inf

    mediator = SimpleNamespace(
        get_ground_grid=grid,
        get_units_in_range=lambda **kwargs: [list(enemies)],
    )
    return SimpleNamespace(
        gatekeeping_pos=Point2(gate_pos),
        start_location=Point2((10, 30)),
        natural_expansion=Point2((10, 30)),
        enemy_race=Race.Zerg,
        structures=FakeUnits(structures),
        enemy_structures=FakeUnits(),
        mediator=mediator,
    )


# ===== CLEARANCE =====


def test_clearance_accounts_for_the_whole_footprint():
    structures = [FakeStructure(10, 13, footprint_radius=1.5)]
    # 3 tiles from a 1.5-radius building, minus the zealot's own radius
    assert _clearance_to_structures(Point2((10, 16)), structures) == pytest.approx(0.5)
    # the building centre is deep inside its footprint
    assert _clearance_to_structures(Point2((10, 13)), structures) < 0


# ===== PARKING SPOT =====


def test_parking_spot_keeps_clear_of_the_building_the_probe_uses():
    """The pylon case: a building four tiles behind the gate is where the old
    fixed-offset fallback used to land, so the parking spot has to move."""
    gate_pos = Point2((10, 10))
    structures = [FakeStructure(10, 15, footprint_radius=1.5)]
    bot = _bot(structures=structures)

    spot = _find_parking_spot(bot, gate_pos, _gatekeeper_structures(bot, gate_pos))

    assert spot is not None
    assert _clearance_to_structures(spot, structures) >= GATEKEEPER_CLEARANCE_MARGIN
    assert cy_distance_to(spot, Point2((10, 15))) >= 1.5 + GATEKEEPER_ZEALOT_RADIUS
    assert cy_distance_to(spot, gate_pos) >= GATEKEEPER_PARK_MIN_DISTANCE


def test_parking_spot_stays_inside_the_reaction_budget():
    """Parked too far back, the zealot is still walking towards the gap when a
    detected enemy arrives - half the detection range is the budget."""
    gate_pos = Point2((10, 10))
    bot = _bot()

    spot = _find_parking_spot(bot, gate_pos, _gatekeeper_structures(bot, gate_pos))

    assert spot is not None
    assert cy_distance_to(spot, gate_pos) <= GATEKEEPER_PARK_MAX_DISTANCE
    assert GATEKEEPER_PARK_MAX_DISTANCE == GATEKEEPER_DETECTION_RANGE / 2


def test_no_parking_spot_outside_the_reaction_budget():
    """A room further back than the budget is no room at all - the gatekeeper
    would not make it back in time."""
    gate_pos = Point2((10, 10))
    structures = [
        FakeStructure(10, 13, footprint_radius=1.5),
        FakeStructure(10, 16, footprint_radius=2.5),
    ]
    bot = _bot(structures=structures)

    assert _find_parking_spot(bot, gate_pos, _gatekeeper_structures(bot, gate_pos)) is None


def test_no_parking_spot_when_structures_cover_every_candidate():
    gate_pos = Point2((10, 10))
    structures = [FakeStructure(10, 13.5, footprint_radius=2.5)]
    bot = _bot(structures=structures)

    assert _find_parking_spot(bot, gate_pos, _gatekeeper_structures(bot, gate_pos)) is None


def test_no_parking_spot_when_the_area_is_not_walkable():
    gate_pos = Point2((10, 10))
    bot = _bot(walkable=False)

    assert _find_parking_spot(bot, gate_pos, []) is None


# ===== ORDER CADENCE =====


def test_move_order_is_not_reissued_for_the_same_target():
    """Re-issuing the move every frame is what fought the engine's push-out."""
    gate = FakeGatekeeper((10, 12), order_target=Point2((10, 13)))

    _order_move(gate, Point2((10, 13 + GATEKEEPER_ORDER_TOLERANCE / 2)))
    assert gate.moves == []

    _order_move(gate, Point2((10, 16)))
    assert gate.moves == [Point2((10, 16))]


# ===== CONTROL =====


def test_gatekeeper_parks_clear_of_the_building_instead_of_standing_on_it():
    structures = [FakeStructure(10, 15, footprint_radius=1.5)]
    bot = _bot(structures=structures)
    gate = FakeGatekeeper((10, 12.5))

    gatekeeper_control(bot, [gate])

    assert len(gate.moves) == 1
    parked = gate.moves[0]
    assert _clearance_to_structures(parked, structures) >= GATEKEEPER_CLEARANCE_MARGIN
    assert cy_distance_to(parked, Point2((10, 15))) >= 1.5 + GATEKEEPER_ZEALOT_RADIUS


def test_parked_gatekeeper_is_not_re_ordered_every_frame():
    bot = _bot(structures=[FakeStructure(10, 15, footprint_radius=1.5)])
    gate = FakeGatekeeper((10, 12.5))

    gatekeeper_control(bot, [gate])
    gatekeeper_control(bot, [gate])

    assert len(gate.moves) == 1


def test_gatekeeper_holds_when_nothing_fits_behind_the_gate():
    """No room inside the reaction budget: hold where we are rather than walk
    back into the building the probe is placing."""
    structures = [
        FakeStructure(10, 13, footprint_radius=1.5),
        FakeStructure(10, 16, footprint_radius=2.5),
    ]
    bot = _bot(structures=structures)
    gate = FakeGatekeeper((10, 12.5))

    gatekeeper_control(bot, [gate])

    assert gate.moves == []
    assert gate.holds == 1


def test_building_landing_on_the_parking_spot_drops_the_stale_move_order():
    bot = _bot()
    gate = FakeGatekeeper((10, 10))

    gatekeeper_control(bot, [gate])
    parked = gate.moves[0]

    # the probe places a building where the zealot was parked
    bot.structures = FakeUnits([FakeStructure(parked.x, parked.y, footprint_radius=1.5)])
    gatekeeper_control(bot, [gate])

    # the zealot must not keep walking towards the tile the building now covers
    assert gate.moves == [parked]
    assert gate.orders[0].ability.id == AbilityId.HOLDPOSITION


def test_gate_is_held_when_enemies_are_close():
    bot = _bot(enemies=[FakeStructure(12, 10, footprint_radius=1.0)])
    gate = FakeGatekeeper((10, 13))

    gatekeeper_control(bot, [gate])

    assert gate.moves == [Point2((10, 10))]


def test_gatekeeper_holds_the_gap_once():
    bot = _bot(enemies=[FakeStructure(12, 10, footprint_radius=1.0)])
    gate = FakeGatekeeper((10, 10))

    gatekeeper_control(bot, [gate])
    assert gate.moves == []
    assert gate.holds == 1

    gatekeeper_control(bot, [gate])
    assert gate.holds == 1


def test_gatekeeper_does_not_fight_for_an_occupied_gap():
    """A building on the gap leaves no tile to hold - park instead of ordering onto it."""
    bot = _bot(
        structures=[FakeStructure(10, 10, footprint_radius=1.5)],
        enemies=[FakeStructure(12, 10, footprint_radius=1.0)],
    )
    gate = FakeGatekeeper((10, 13))

    gatekeeper_control(bot, [gate])

    assert Point2((10, 10)) not in gate.moves
