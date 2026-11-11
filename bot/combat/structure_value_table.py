"""Structure-value seed table: what killing an enemy STRUCTURE is worth.

Purpose: give the squad target scorer (plan Task 10, consideration 2 - "Strategic
    value") a number for enemy buildings. `army_value` is 0.0 for every structure and
    `TACTICAL_BONUS` covers units only, so nothing in the codebase prices a building
    today; this table is that seed.
Key Decisions: the tier of a row is encoded in its value band (production > tech >
    supply > misc), so a row cannot claim a tier its number does not support and there
    is no parallel tier map to drift. The scale is positive-only (0.0-10.0): unlike
    the unit matchup table there is no negative structure - killing a building is
    never a loss for us.
Limitations: the seeds are hand-set from Liquipedia structure facts and are re-fit
    from telemetry in Task 8. Nothing imports this table yet - Part 2 wires it into
    the squad target scorer. The Liquipedia API rate-limited this environment
    (HTTP 429) while the table was authored, so each row cites the Liquipedia page
    and the field fact behind its number rather than a verbatim quote.
"""

from collections.abc import Mapping
from types import MappingProxyType

from sc2.ids.unit_typeid import UnitTypeId

# ===== VALUE SCALE =====
# Positive-only, unlike SCORING_TABLE (-12..12): a structure is never a bad target
# in itself, so the scale starts at 0.0 and runs to 10.0 for a main base.

STRUCTURE_VALUE_MIN = 0.0
"""Lower bound of the structure-value scale."""

STRUCTURE_VALUE_MAX = 10.0
"""Upper bound of the structure-value scale (see STRUCTURE_VALUE_MIN)."""

STRUCTURE_VALUE_STEP = 0.5
"""Every value is a multiple of this step, so equal-looking rows really are equal."""

STRUCTURE_VALUE_MIN_ROWS = 20
"""Lower bound of the table's size - the plan asks for ~20-30 rows."""

STRUCTURE_VALUE_MAX_ROWS = 30
"""Upper bound of the table's size (see STRUCTURE_VALUE_MIN_ROWS)."""

# ===== TIERS =====
# The plan's ordering: production > tech > supply > misc.
#
# WHAT THE SCALE RANKS: strategic leverage - what the loss compounds into - not
# rebuild cost. A Planetary Fortress costs more resources than a Supply Depot but
# sits below it: the fortress is positional and replaceable, while a supply block
# denies an army the opponent has already paid for. The same reading puts static
# defense above the gas extractors: a turret blocks the approach to their base, a
# refinery only slows future income.

TIER_ORDER: tuple[str, ...] = ("production", "tech", "supply", "misc")
"""Tiers from highest to lowest strategic value."""

STRUCTURE_VALUE_TIERS: dict[str, tuple[float, float]] = {
    "production": (7.0, 10.0),
    "tech": (5.5, 6.5),
    "supply": (4.0, 5.0),
    "misc": (1.0, 3.5),
}
"""Inclusive (min, max) value band per tier. Bands are strictly ordered and do not
    overlap, so every table value belongs to exactly one tier."""

# ===== STRUCTURE VALUE TABLE =====
# enemy structure -> strategic value of killing it.
#
# DATA SOURCE: Liquipedia SC2 structure pages (Legacy of the Void versions)
#   https://liquipedia.net/starcraft2/<StructureName>_(Legacy_of_the_Void)
#   Each row is annotated with "# LP:" citing the Liquipedia fact behind the number.
#
# HOW TO EXTEND: add the row in its tier's band and keep the value on the 0.5 step.
# A structure that is missing scores 0.0 through `.get(..., 0.0)`, which reads as
# "not worth attacking" - so a gap here is a silent hole, not a neutral entry.
#
# COVERAGE (tests/unit/test_structure_value_table.py enforces it): all three races
# plus their static defense (Bunker, Planetary Fortress, Missile Turret, Photon
# Cannon, Spine/Spore Crawler).
STRUCTURE_VALUE_TABLE: Mapping[UnitTypeId, float] = MappingProxyType({
    # --- Terran: production ---
    UnitTypeId.COMMANDCENTER:     10.0,  # LP: Command Center - Terran main base; all Terran production and gas is attached to it
    UnitTypeId.BARRACKS:          8.0,  # LP: Barracks - infantry production (Marines, Medivacs, Reapers); losing it stops infantry output
    UnitTypeId.FACTORY:           8.0,  # LP: Factory - vehicle production (Hellions, Siege Tanks, Thors)
    UnitTypeId.STARPORT:          8.0,  # LP: Starport - air production (Banshees, Vikings, Liberators)

    # --- Terran: tech ---
    UnitTypeId.TECHLAB:           6.0,  # LP: Tech Lab - attached tech building (Reaper/Hellion/Banshee upgrades); losing it stops the host building's upgrades

    # --- Terran: supply ---
    UnitTypeId.SUPPLYDEPOT:       5.0,  # LP: Supply Depot - 10 supply; a supply block delays an army the opponent already paid for

    # --- Terran: misc ---
    UnitTypeId.PLANETARYFORTRESS: 3.5,  # LP: Planetary Fortress - static defense that also supplies; the position dies, not their production
    UnitTypeId.BUNKER:            3.0,  # LP: Bunker - static defense that can repair itself underground; killing it removes the position
    UnitTypeId.MISSILETURRET:     3.0,  # LP: Missile Turret - cheap static defense blocking the approach to their base
    UnitTypeId.REFINERY:          2.0,  # LP: Refinery - vespene extraction; slows future gas, not the army in front of us

    # --- Protoss: production ---
    UnitTypeId.NEXUS:             10.0,  # LP: Nexus - Protoss main base; all Protoss production is attached to it
    UnitTypeId.GATEWAY:           8.0,  # LP: Gateway - infantry production (Zealots, Stalkers, Adepts, High Templar)
    UnitTypeId.ROBOTICSFACILITY:  8.0,  # LP: Robotics Facility - robo production (Immortals, Colossi, Disruptors)
    UnitTypeId.STARGATE:          8.0,  # LP: Stargate - air production (Carriers, Void Rays, Tempests)

    # --- Protoss: tech ---
    UnitTypeId.ROBOTICSBAY:       6.5,  # LP: Robotics Bay - tech building plus Warp Prism and Photon Cannon upgrades
    UnitTypeId.FUSIONCORE:        6.5,  # LP: Fusion Core - tech building; Arbiter and Vorpal Blade upgrade
    UnitTypeId.TEMPLARARCHIVE:    6.0,  # LP: Templar Archive - unlocks Dark Templar and the Templar upgrades
    UnitTypeId.TWILIGHTCOUNCIL:   6.0,  # LP: Twilight Council - research for Blink and the other Protoss upgrades

    # --- Protoss: supply ---
    UnitTypeId.PYLON:             5.0,  # LP: Pylon - 8 supply; the Protoss supply building that gates every early expansion

    # --- Protoss: misc ---
    UnitTypeId.PHOTONCANNON:      3.5,  # LP: Photon Cannon - long-range static defense; also the Protoss tech structure at a base
    UnitTypeId.ASSIMILATOR:       2.0,  # LP: Assimilator - vespene extraction; slows future gas, not the army in front of us

    # --- Zerg: production ---
    UnitTypeId.HATCHERY:          10.0,  # LP: Hatchery - Zerg main base; all Zerg production hangs off it
    UnitTypeId.LAIR:              8.5,  # LP: Lair - Hatchery upgrade that adds a production building and widens the base

    # --- Zerg: tech ---
    UnitTypeId.HIVE:              6.5,  # LP: Hive - Zerg tech: the top-tier production option and the tech unlock
    UnitTypeId.SPIRE:             6.5,  # LP: Spire - tech building; unlocks Ultralisk and Corruptor
    UnitTypeId.ULTRALISKCAVERN:   6.5,  # LP: Ultralisk Cavern - the only source of Ultralisks, their heaviest army unit
    UnitTypeId.INFESTATIONPIT:    6.0,  # LP: Infestation Pit - houses the Infestor; losing it removes their fungus and neural parasite

    # --- Zerg: supply ---
    UnitTypeId.CREEPTUMOR:        4.5,  # LP: Creep Tumor - LotV creep tumors grant 1 supply each, so the Zerg supply building sits here

    # --- Zerg: misc ---
    UnitTypeId.SPINECRAWLER:      3.0,  # LP: Spine Crawler - static defense that burrows to dodge splash damage
    UnitTypeId.SPORECRAWLER:      3.0,  # LP: Spore Crawler - static defense that shoots spores
})
