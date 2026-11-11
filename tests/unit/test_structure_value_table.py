"""Format + tier validation for STRUCTURE_VALUE_TABLE (bot/combat/structure_value_table.py).

Purpose: pin the structure-value seed table's invariants so a bad row cannot slip in
    unnoticed - duplicate keys, keys that do not resolve, off-scale values, tier
    ordering violations, and unit ids leaking into what must be a structure table.
Key Decisions: the table is inspected directly - no game, no SC2 binary. The duplicate
    check parses the source with `ast`: a Python dict keeps only the last value for a
    repeated key, so the runtime table cannot show duplicates. Structure membership is
    checked against ARES' ALL_STRUCTURES oracle instead of a hand-written list, so a
    typo'd id cannot masquerade as a structure.
Limitations: the checks are structural - they cannot tell whether a row is
    strategically right; the Liquipedia provenance behind each row lives in the
    table's own comments. Tier membership is derived from the value band (the value
    encodes the tier), so a row with a value between bands fails instead of being
    silently unclassified.
"""

import ast
from pathlib import Path
from types import MappingProxyType

from sc2.ids.unit_typeid import UnitTypeId

from bot.combat.structure_value_table import (
    STRUCTURE_VALUE_MAX,
    STRUCTURE_VALUE_MAX_ROWS,
    STRUCTURE_VALUE_MIN,
    STRUCTURE_VALUE_MIN_ROWS,
    STRUCTURE_VALUE_STEP,
    STRUCTURE_VALUE_TABLE,
    STRUCTURE_VALUE_TIERS,
    TIER_ORDER,
)
from ares.consts import ALL_STRUCTURES

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE_SOURCE = REPO_ROOT / "bot" / "combat" / "structure_value_table.py"

MIN_ROWS = STRUCTURE_VALUE_MIN_ROWS
"""Lower bound of the table's documented size. The plan asks for ~20-30 rows; a table
that shrinks below this means coverage was lost, not that the plan was trimmed."""

MAX_ROWS = STRUCTURE_VALUE_MAX_ROWS
"""Upper bound of the table's documented size (see MIN_ROWS)."""

VALUE_MIN = STRUCTURE_VALUE_MIN
"""Lower bound of the value scale. Structures are only ever worth a positive amount -
    there is no "negative structure" to target, unlike the unit matchup table."""

VALUE_MAX = STRUCTURE_VALUE_MAX
"""Upper bound of the value scale (see VALUE_MIN)."""

VALUE_STEP = STRUCTURE_VALUE_STEP
"""Every value is a multiple of this step, so the scale stays readable and two rows
    that look equal in review really are equal."""


# ===== SOURCE PARSING HELPERS =====
# The runtime table hides duplicates (dict literal semantics), so the source is
# the ground truth for "which keys are listed".

def _assigned_name(node: ast.stmt) -> str | None:
    """Name of the single module-level variable a statement assigns, if any."""
    if isinstance(node, ast.AnnAssign):
        targets = [node.target]
    elif isinstance(node, ast.Assign):
        targets = node.targets
    else:
        return None
    if len(targets) == 1 and isinstance(targets[0], ast.Name):
        return targets[0].id
    return None


def _dict_literal(source: Path, literal_name: str) -> ast.Dict:
    """Return the ast.Dict node behind a module-level mapping literal.

    Args:
        source: Python file holding the literal.
        literal_name: Name of the module-level variable to read.

    Returns:
        The ast.Dict node. Raises AssertionError if the literal is missing or is
        not a mapping.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if _assigned_name(node) != literal_name:
            continue
        value = node.value
        # The table is wrapped in MappingProxyType for immutability - unwrap it.
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "MappingProxyType"
            and len(value.args) == 1
        ):
            value = value.args[0]
        assert isinstance(value, ast.Dict), f"{literal_name} is not a mapping literal"
        return value
    raise AssertionError(f"{literal_name} literal not found in {source}")


def _table_key_names(source: Path, literal_name: str) -> list[str]:
    """Return the UnitTypeId member names used as keys of a mapping literal."""
    literal = _dict_literal(source, literal_name)
    names = []
    for key in literal.keys:
        assert isinstance(key, ast.Attribute) and isinstance(key.value, ast.Name), (
            f"{literal_name} keys must reference UnitTypeId members"
        )
        assert key.value.id == "UnitTypeId", f"{literal_name} keys must reference UnitTypeId members"
        names.append(key.attr)
    return names


def _tier_of(value: float) -> str | None:
    """Tier band a value falls into, or None when it falls in no band."""
    for tier, (low, high) in STRUCTURE_VALUE_TIERS.items():
        if low <= value <= high:
            return tier
    return None


def _values_in_tier(tier: str) -> list[float]:
    """All table values whose band is the given tier."""
    return [
        value
        for value in STRUCTURE_VALUE_TABLE.values()
        if _tier_of(value) == tier
    ]


# ===== TABLE SHAPE =====


def test_table_has_rows() -> None:
    """The table is the bot's structure-value seed - an empty one scores every structure 0."""
    assert MIN_ROWS <= len(STRUCTURE_VALUE_TABLE) <= MAX_ROWS, (
        f"structure-value table has {len(STRUCTURE_VALUE_TABLE)} rows, expected {MIN_ROWS}-{MAX_ROWS}"
    )


def test_no_duplicate_keys() -> None:
    """No key may be listed twice - the second value would silently win."""
    names = _table_key_names(TABLE_SOURCE, "STRUCTURE_VALUE_TABLE")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"duplicate STRUCTURE_VALUE_TABLE keys: {duplicates}"


def test_keys_resolve_to_unit_type_ids() -> None:
    """Every key must be a real UnitTypeId member.

    A typo'd key scores 0.0 through `.get(..., 0.0)` - invisible at runtime, fatal
    to the row it was meant to express.
    """
    for name in _table_key_names(TABLE_SOURCE, "STRUCTURE_VALUE_TABLE"):
        assert name in UnitTypeId.__members__, f"UnitTypeId.{name} does not exist"


def test_runtime_keys_are_unit_type_ids() -> None:
    """The imported table must be keyed by UnitTypeId members."""
    for key in STRUCTURE_VALUE_TABLE:
        assert isinstance(key, UnitTypeId), f"bad key: {key!r}"


def test_table_is_immutable() -> None:
    """The table is reference data - a consumer that mutates it would drift the seeds."""
    assert isinstance(STRUCTURE_VALUE_TABLE, MappingProxyType), (
        "STRUCTURE_VALUE_TABLE must be a MappingProxyType (frozen mapping)"
    )


def test_values_on_one_scale() -> None:
    """All values are on the documented positive scale."""
    for key, value in STRUCTURE_VALUE_TABLE.items():
        assert isinstance(value, float), f"{key} has non-float value {value!r}"
        assert VALUE_MIN <= value <= VALUE_MAX, (
            f"{key} value {value} outside [{VALUE_MIN}, {VALUE_MAX}]"
        )
        steps = value / VALUE_STEP
        assert steps.is_integer(), f"{key} value {value} is not a multiple of {VALUE_STEP}"


# ===== STRUCTURE MEMBERSHIP =====


def test_every_key_is_a_structure() -> None:
    """Every key must be a real structure, checked against ARES' structure oracle.

    A hand-written list would drift; ALL_STRUCTURES is the same set ARES itself
    uses to decide what counts as a building.
    """
    for key in STRUCTURE_VALUE_TABLE:
        assert key in ALL_STRUCTURES, f"{key} is not a structure"


def test_no_mobile_unit_ids() -> None:
    """Mobile units must not appear - they are covered by army_value / TACTICAL_BONUS.

    The table answers "what is killing this STRUCTURE worth"; a unit row here would
    double-count the unit scoring path.
    """
    mobile = {
        UnitTypeId.ZEALOT,
        UnitTypeId.STALKER,
        UnitTypeId.IMMORTAL,
        UnitTypeId.COLOSSUS,
        UnitTypeId.MARINE,
        UnitTypeId.SCV,
        UnitTypeId.PROBE,
        UnitTypeId.DRONE,
        UnitTypeId.ZERGLING,
        UnitTypeId.ROACH,
    }
    assert not (set(STRUCTURE_VALUE_TABLE) & mobile), (
        f"mobile units in the structure table: {set(STRUCTURE_VALUE_TABLE) & mobile}"
    )
    assert UnitTypeId.ZEALOT not in STRUCTURE_VALUE_TABLE


def test_all_three_races_represented() -> None:
    """The Protoss bot faces all three races - each must have rows."""
    representatives = {
        "Terran": [
            UnitTypeId.COMMANDCENTER,
            UnitTypeId.FACTORY,
            UnitTypeId.SUPPLYDEPOT,
            UnitTypeId.BUNKER,
        ],
        "Protoss": [
            UnitTypeId.NEXUS,
            UnitTypeId.ROBOTICSFACILITY,
            UnitTypeId.PYLON,
            UnitTypeId.PHOTONCANNON,
        ],
        "Zerg": [
            UnitTypeId.HATCHERY,
            UnitTypeId.HIVE,
            UnitTypeId.CREEPTUMOR,
            UnitTypeId.SPORECRAWLER,
        ],
    }
    for race, ids in representatives.items():
        missing = [unit_id.name for unit_id in ids if unit_id not in STRUCTURE_VALUE_TABLE]
        assert not missing, f"{race} structures missing from the table: {missing}"


def test_defensive_structures_covered() -> None:
    """Static defense is the half of the table the bot must not forget."""
    static_defense = [
        UnitTypeId.BUNKER,
        UnitTypeId.PLANETARYFORTRESS,
        UnitTypeId.MISSILETURRET,
        UnitTypeId.PHOTONCANNON,
        UnitTypeId.SPINECRAWLER,
        UnitTypeId.SPORECRAWLER,
    ]
    missing = [unit_id.name for unit_id in static_defense if unit_id not in STRUCTURE_VALUE_TABLE]
    assert not missing, f"static defense missing from the table: {missing}"


# ===== TIER ORDERING =====


def test_tier_bands_are_strictly_ordered() -> None:
    """Each tier band must sit strictly above the next one, with no overlap.

    Overlapping bands would let a value belong to two tiers - the tier of a row
    would then depend on iteration order.
    """
    for higher, lower in zip(TIER_ORDER, TIER_ORDER[1:]):
        high_low, high_high = STRUCTURE_VALUE_TIERS[higher]
        low_low, low_high = STRUCTURE_VALUE_TIERS[lower]
        assert high_low > low_high, (
            f"tier {higher} ({high_low}-{high_high}) must sit strictly above "
            f"{lower} ({low_low}-{low_high})"
        )


def test_every_row_falls_in_exactly_one_tier() -> None:
    """Every value must land in a tier band - no orphan values."""
    orphans = {
        key.name: value
        for key, value in STRUCTURE_VALUE_TABLE.items()
        if _tier_of(value) is None
    }
    assert not orphans, f"values outside every tier band: {orphans}"


def test_every_tier_has_rows() -> None:
    """A tier with no rows means a whole category of structures is unranked."""
    for tier in TIER_ORDER:
        assert _values_in_tier(tier), f"tier {tier} has no rows"


def test_tier_ordering_holds_for_values() -> None:
    """Every value in a higher tier must beat every value in a lower tier.

    This is the property the tiers exist for: "production > tech > supply > misc"
    has to hold for the actual numbers, not just for the documented bands.
    """
    for higher, lower in zip(TIER_ORDER, TIER_ORDER[1:]):
        high_values = _values_in_tier(higher)
        low_values = _values_in_tier(lower)
        assert min(high_values) > max(low_values), (
            f"tier {higher} (min {min(high_values)}) must beat tier {lower} (max {max(low_values)})"
        )


def test_tier_order_covers_every_documented_tier() -> None:
    """TIER_ORDER and the band mapping must describe the same tiers."""
    assert set(TIER_ORDER) == set(STRUCTURE_VALUE_TIERS), (
        "TIER_ORDER and STRUCTURE_VALUE_TIERS disagree on the tier list"
    )


def test_anchor_structures_sit_in_their_tier() -> None:
    """The tier of the landmark structures must match the tier they are read as.

    The tier is derived from the value band, so a mis-valued main base would
    silently reclassify itself. These anchors pin the reading the plan means:
    a main base is production, a tech lab is tech, a pylon is supply, a bunker
    is misc.
    """
    anchors = {
        "production": [
            UnitTypeId.COMMANDCENTER,
            UnitTypeId.NEXUS,
            UnitTypeId.HATCHERY,
            UnitTypeId.GATEWAY,
            UnitTypeId.FACTORY,
        ],
        "tech": [
            UnitTypeId.ROBOTICSBAY,
            UnitTypeId.SPIRE,
            UnitTypeId.HIVE,
            UnitTypeId.TECHLAB,
        ],
        "supply": [
            UnitTypeId.PYLON,
            UnitTypeId.SUPPLYDEPOT,
            UnitTypeId.CREEPTUMOR,
        ],
        "misc": [
            UnitTypeId.BUNKER,
            UnitTypeId.PLANETARYFORTRESS,
            UnitTypeId.PHOTONCANNON,
            UnitTypeId.ASSIMILATOR,
        ],
    }
    for tier, ids in anchors.items():
        for unit_id in ids:
            value = STRUCTURE_VALUE_TABLE[unit_id]
            assert _tier_of(value) == tier, (
                f"{unit_id.name} sits in tier {_tier_of(value)}, expected {tier}"
            )


# ===== MODULE PURITY =====


def test_module_imports_no_bot_packages() -> None:
    """The table is pure data - it must not pull managers (or combat) into its import.

    Importing a manager here would drag the build runner and ARES wiring into every
    consumer of this table.
    """
    tree = ast.parse(TABLE_SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("bot"), (
                    f"structure_value_table must not import {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith("bot"), (
                f"structure_value_table must not import from {module}"
            )
