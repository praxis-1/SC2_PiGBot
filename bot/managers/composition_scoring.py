"""Composition scoring — the macro manager's utility-AI decision layer.

Purpose: Replace the profile → sticky-switch → nudge pipeline in the macro
         manager with a scored competition: candidate army compositions (the
         profile builds, counter-driven variants and the hold option) compete on
         the macro cycle through the shared reasoner, and the winner builds.
Key Decisions: The scoring table is the core base score, not a post-decision
               nudge. Belief confidence is a multiplier, never an additive term.
               There are no sticky switches: the stabilizers are curve steepness
               first and a scored commitment (hysteresis) bonus second.
               The cost term is the price of a mix; the economy term is the
               guard that says whether we can sustain it.
               Buildability invariant: every candidate on the ballot is fully
               buildable and the ballot is never empty — a composition
               production cannot execute never wins the competition.
Limitations: The magnitudes are seeds — Task 8 refits them from telemetry. The
               scorer re-scores on the macro cadence (a belief change or every N
               cycles), never per frame. The nearest-reachable translation runs
               in the counter-variant machinery only; the tech-proximity
               semi-tree (Task 9) is where the wider translation belongs.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sc2.constants import EQUIVALENTS_FOR_TECH_PROGRESS
from sc2.ids.unit_typeid import UnitTypeId

from ares.consts import ALL_STRUCTURES, WORKER_TYPES
from ares.dicts.cost_dict import COST_DICT
from ares.dicts.unit_data import UNIT_DATA
from ares.dicts.unit_tech_requirement import UNIT_TECH_REQUIREMENT

from bot.combat.scoring_table import SCORING_TABLE, SITUATIONAL_ROWS
from bot.constants import (
    PRODUCTION_MIN_PROPORTION,
    STRATEGY_EXPECTED_UNITS,
    get_active_profile,
)
from bot.utilities.reasoner import (
    Appraisal,
    BeliefSnapshot,
    Choice,
    Consideration,
    CurveParams,
    EconomySnapshot,
    IntelContext,
    RankedChoice,
    apply_curve,
    combine,
    load_curve_params,
    score,
)

# ===== CANDIDATE SET =====

CANDIDATE_CAP = 8
"""Maximum number of compositions that compete in one scoring round."""

MAX_COUNTER_VARIANTS = 5
"""How many counter-driven variants the builder may create before the cap bites."""

COUNTER_VARIANT_SHIFT = 0.15
"""Proportion points moved from the least effective unit to the best counter.
Provenance: the old resource-pressure nudge moved ±15% of proportion; a
candidate shifts the same order of magnitude, so the scored decision lands
where the old nudge did."""

PROFILE_SOURCE = "profile"
"""Candidate came from the active build profile."""

HOLD_SOURCE = "hold"
"""Candidate is the composition we already decided to build."""

COUNTER_SOURCE = "counter"
"""Candidate is a profile mix shifted toward the best counter."""

HOLD_NAME = "hold"
"""Name of the hold candidate."""

# ===== BUILDABILITY =====

FALLBACK_SOURCE = "fallback"
"""Candidate is the conservative mix that keeps the ballot alive."""

FALLBACK_NAME = "fallback"
"""Name of the conservative buildable candidate."""


def _build_tech_equivalents() -> dict[UnitTypeId, UnitTypeId]:
    """The base building each equivalent structure stands in for.

    A Warp Gate is a Gateway, a Phasing Warp Prism is a Warp Prism — the tech
    tree counts them as one node. The first base (alphabetically) wins, so the
    mapping is deterministic.
    """
    mapping: dict[UnitTypeId, UnitTypeId] = {}
    for base in sorted(EQUIVALENTS_FOR_TECH_PROGRESS, key=lambda item: item.name):
        equivalents = sorted(
            EQUIVALENTS_FOR_TECH_PROGRESS[base], key=lambda item: item.name
        )
        for equivalent in equivalents:
            mapping.setdefault(equivalent, base)
    return mapping


TECH_EQUIVALENTS = _build_tech_equivalents()
"""Equivalent structures folded onto their base building."""


def _tech_requirements(unit_type: UnitTypeId) -> frozenset[UnitTypeId]:
    """The buildings a unit type needs before it can be trained.

    ARES' own table — the one ``tech_ready_for_unit`` walks — with the
    equivalent structures folded onto their base, so two unit types on the same
    tech path compare equal.
    """
    return frozenset(
        TECH_EQUIVALENTS.get(building, building)
        for building in UNIT_TECH_REQUIREMENT.get(unit_type, ())
    )


def _tech_distance(ideal: UnitTypeId, other: UnitTypeId) -> int:
    """How far apart two unit types sit on the tech tree, in buildings."""
    return len(_tech_requirements(ideal) ^ _tech_requirements(other))


# ===== COST MODEL =====

SUPPLY_COST_WEIGHT = 10.0
"""Resources one supply point is worth when the cost term prices production time.
A supply point is roughly ten resources of build cost, so resource burden and
production time can be added together."""

ECONOMY_WORKER_CAPACITY = 30
"""Workers below which the economy cannot sustain a full composition."""

SATURATION_WORKERS_PER_BASE = 16
"""Workers a Protoss base needs before its minerals are worked out."""

# ===== UPGRADE SYNERGY =====

UPGRADE_SYNERGY_UNITS: dict[str, tuple[UnitTypeId, ...]] = {
    "WARPGATERESEARCH": (UnitTypeId.STALKER,),
    "CHARGE": (UnitTypeId.ZEALOT,),
    "EXTENDEDTHERMALLANCE": (UnitTypeId.HIGHTEMPLAR, UnitTypeId.ARCHON),
    "PSISTORMTECH": (UnitTypeId.HIGHTEMPLAR, UnitTypeId.ARCHON),
}
"""Upgrades and the unit types that exploit them. The tech term asks how much of
our finished research the composition actually puts on the field."""

# ===== SEED PARAMETERS =====

DEFAULT_WEIGHTS: dict[str, float] = {
    "counter": 2.0,
    "economy": 1.0,
    "tech": 0.5,
    "cost": 0.6,
    "commitment": 0.03,
}
"""Seed weights of the scoring terms.

Provenance: the counter term carries the scoring table's own magnitude
(Liquipedia-annotated pairs, −10..+12) and is the core of the score, so it
weighs more than the guard terms. The economy and cost terms are guards: they
only separate the candidates while the economy is tight (the economy term
saturates on a rich one, the cost term scales with income tightness). The
commitment weight equals the margin on purpose — the incumbent keeps the
crown exactly while the challenger's gap stays inside the margin.
Task 8 refits them."""

DEFAULT_COMMITMENT_MARGIN = 0.03
"""Score margin a challenger has to beat to take the crown from the incumbent.

Seeded between the two gaps the scorer actually produces: a 15-point mix shift
against a matching enemy is worth about 0.05-0.09 of base score, while a 5%
wobble in the believed enemy shares is worth about 0.01. The margin therefore
lets a real counter advantage take the crown and keeps belief noise from
flipping it — half a counter-table point, in the units the table speaks."""

DEFAULT_RESCORE_INTERVAL = 10
"""Macro cycles between forced re-scores (a belief change re-scores earlier)."""

DEFAULT_CURVES: dict[str, CurveParams] = {
    "counter": CurveParams(
        kind="linear", minimum=-8.0, maximum=8.0, midpoint=0.0, steepness=4.0
    ),
    "economy": CurveParams(
        kind="logistic", minimum=-600.0, maximum=600.0, midpoint=0.0, steepness=4.0
    ),
    "tech": CurveParams(
        kind="linear", minimum=0.0, maximum=1.0, midpoint=0.5, steepness=4.0
    ),
    "cost": CurveParams(
        kind="linear", minimum=0.0, maximum=560.0, midpoint=0.5, steepness=4.0
    ),
    "income": CurveParams(
        kind="logistic", minimum=0.0, maximum=2000.0, midpoint=800.0, steepness=8.0
    ),
    "commitment": CurveParams(
        kind="linear", minimum=0.0, maximum=1.0, midpoint=0.5, steepness=4.0
    ),
}
"""Fallback curves when config.yml does not carry an entry.

counter — linear over the band the table's rows actually live in. The counter
value is a value scale, not a threat signal, so it keeps its resolution across
the whole band instead of saturating like a sigmoid would.
economy — sigmoid over the affordability headroom (see EconomyFit). The
midpoint sits at break-even: a mix that costs exactly what the economy spends
per minute is neutral, a comfortable mix saturates (affordability stops
separating the candidates) and a mix that outruns the economy bites hard.
cost — linear over the price of the priciest single-unit mix.
income — sigmoid over resources per minute; it feeds the cost term's tightness
(get_economy_state calls 1600 minerals/min "full")."""


@dataclass(frozen=True)
class CompositionCandidate:
    """One army composition competing for the build slot.

    name: stable identifier — telemetry and the debug overlay read it.
    composition: the ARES composition dict ({unit type: {proportion, priority}}).
    source: where the candidate came from — profile, hold or counter.
    """

    name: str
    composition: dict[UnitTypeId, dict]
    source: str


@dataclass(frozen=True)
class CompositionContext(IntelContext):
    """IntelContext plus the candidate under evaluation.

    Considerations score a candidate, and IntelContext is the shared snapshot, so
    the candidate rides along as a subclass field — every field the reasoner
    defines stays available and `evaluate(context)` still takes one object.
    """

    candidate: CompositionCandidate | None = None


@dataclass(frozen=True)
class CompositionSelection:
    """The outcome of one scoring round."""

    candidate: CompositionCandidate | None
    ranking: list[RankedChoice]
    context: IntelContext
    candidates: list[CompositionCandidate]


# ===== SCORING MATH =====


def counter_effectiveness(
    composition: Mapping[UnitTypeId, Mapping[str, float]],
    enemy_shares: Mapping[str, float],
) -> float:
    """Expected counter value of a composition against the believed enemy army.

    For every unit type we build, the scoring table's value against each
    believed enemy type is weighted by that enemy's share of the enemy army, and
    our own proportion weights the unit type. Situational rows (fixed-count
    support value, not per-enemy damage) are excluded.

    Perf note: O(k · m) with k = unit types in the mix, m = believed enemy
    types — a handful on each side.
    """
    if not enemy_shares:
        return 0.0
    total = 0.0
    for unit_type, info in composition.items():
        proportion = float(info.get("proportion", 0.0))
        if proportion <= 0.0:
            continue
        unit_value = 0.0
        for enemy_name, share in enemy_shares.items():
            enemy_type = _unit_by_name(enemy_name)
            if enemy_type is None:
                continue
            row = (unit_type, enemy_type)
            if row in SITUATIONAL_ROWS:
                continue
            unit_value += float(share) * SCORING_TABLE.get(row, 0.0)
        total += proportion * unit_value
    return total


def composition_cost(composition: Mapping[UnitTypeId, float]) -> float:
    """Resource and production-time burden of one unit of a composition.

    Resources come from ARES' cost table, supply from its unit data — supply is
    production time, priced at SUPPLY_COST_WEIGHT resources a point.
    """
    total = 0.0
    for unit_type, info in composition.items():
        proportion = float(info.get("proportion", 0.0))
        if proportion <= 0.0:
            continue
        cost = COST_DICT.get(unit_type)
        if cost is None:
            continue
        supply = float(UNIT_DATA.get(unit_type, {}).get("supply", 0))
        total += proportion * (
            cost.minerals + cost.vespene + SUPPLY_COST_WEIGHT * supply
        )
    return total


def composition_appetite(composition: Mapping[UnitTypeId, float]) -> float:
    """What one unit of a composition costs in resources (no supply term).

    This is the economy term's question — can the economy sustain this mix —
    while the cost term asks the heavier question of resources *and* time.
    """
    total = 0.0
    for unit_type, info in composition.items():
        proportion = float(info.get("proportion", 0.0))
        if proportion <= 0.0:
            continue
        cost = COST_DICT.get(unit_type)
        if cost is None:
            continue
        total += proportion * (cost.minerals + cost.vespene)
    return total


def _unit_by_name(name: str) -> UnitTypeId | None:
    """Resolve a unit type name — the belief snapshot speaks names, not ids."""
    try:
        return UnitTypeId[name]
    except KeyError:
        return None


def _candidate_of(context: IntelContext) -> CompositionCandidate | None:
    """The candidate a consideration is scoring (None for a plain IntelContext)."""
    return getattr(context, "candidate", None)


# ===== CONSIDERATIONS =====


class CounterEffectiveness:
    """Σ counter effectiveness of our units against the believed enemy army.

    The scoring table is the core base score. The raw sum goes through the
    counter curve — a linear value scale over the table's band — so a mix that
    counters well scores above a neutral mix and a mix that is bad against the
    enemy scores below it.
    """

    def __init__(self) -> None:
        self._weight = DEFAULT_WEIGHTS["counter"]
        self._curve = DEFAULT_CURVES["counter"]

    def load(self, params: Mapping[str, Any]) -> None:
        self._weight = float(params.get("weight", self._weight))
        self._curve = params.get("curve", self._curve)

    def evaluate(self, context: IntelContext) -> Appraisal:
        candidate = _candidate_of(context)
        if candidate is None:
            return Appraisal()
        raw = counter_effectiveness(
            candidate.composition, context.belief.enemy_composition
        )
        shaped = apply_curve(raw, self._curve)
        return Appraisal(base_score=self._weight * (shaped - 0.5), multiplier=1.0)


class EconomyFit:
    """Can the economy sustain this composition?

    The raw input is the affordability headroom: what the economy can spend per
    minute (income, scaled by saturation and by how many workers we have)
    minus what one army member of the mix costs. The headroom answers a
    concrete question — can we sustain a stream of this composition? — and the
    curve saturates: a rich economy's headroom runs past the curve's band, so
    affordability stops separating the candidates and the counter term decides
    alone. A mix that outruns the economy bites hard.
    """

    def __init__(self) -> None:
        self._weight = DEFAULT_WEIGHTS["economy"]
        self._curve = DEFAULT_CURVES["economy"]

    def load(self, params: Mapping[str, Any]) -> None:
        self._weight = float(params.get("weight", self._weight))
        self._curve = params.get("curve", self._curve)

    def evaluate(self, context: IntelContext) -> Appraisal:
        candidate = _candidate_of(context)
        if candidate is None:
            return Appraisal()
        capacity = economy_capacity(context.economy)
        if capacity <= 0.0:
            return Appraisal()
        appetite = composition_appetite(candidate.composition)
        shaped = apply_curve(capacity - appetite, self._curve)
        return Appraisal(base_score=self._weight * (shaped - 0.5), multiplier=1.0)


class TechSynergy:
    """How much of our finished tech the composition actually exploits.

    The raw input is the share of our researched upgrades that the mix's units
    benefit from. Nothing researched means nothing to exploit, so the term
    stays out of the score instead of punishing every candidate equally.
    """

    def __init__(self) -> None:
        self._weight = DEFAULT_WEIGHTS["tech"]
        self._curve = DEFAULT_CURVES["tech"]

    def load(self, params: Mapping[str, Any]) -> None:
        self._weight = float(params.get("weight", self._weight))
        self._curve = params.get("curve", self._curve)

    def evaluate(self, context: IntelContext) -> Appraisal:
        candidate = _candidate_of(context)
        if candidate is None:
            return Appraisal()
        progress = context.tech_progress
        researched = sum(progress.values())
        if researched <= 0.0:
            return Appraisal()
        exploited = 0.0
        for unit_type, info in candidate.composition.items():
            proportion = float(info.get("proportion", 0.0))
            if proportion <= 0.0:
                continue
            bonus = sum(
                float(value)
                for name, value in progress.items()
                if unit_type in UPGRADE_SYNERGY_UNITS.get(name, ())
            )
            exploited += proportion * bonus
        shaped = apply_curve(exploited / researched, self._curve)
        baseline = apply_curve(0.0, self._curve)
        return Appraisal(base_score=self._weight * (shaped - baseline), multiplier=1.0)


class CostBurden:
    """The resource-and-time burden of a composition — a negative base.

    The raw input is what one army member of the mix costs in resources plus
    supply (supply is production time, priced in resources). The burden is
    scaled by economy tightness: a rich economy can afford the heavy units, so
    the burden stops counting and the counter term decides alone.
    """

    def __init__(self) -> None:
        self._weight = DEFAULT_WEIGHTS["cost"]
        self._curve = DEFAULT_CURVES["cost"]
        self._income_curve = DEFAULT_CURVES["income"]

    def load(self, params: Mapping[str, Any]) -> None:
        self._weight = float(params.get("weight", self._weight))
        self._curve = params.get("curve", self._curve)
        self._income_curve = params.get("income_curve", self._income_curve)

    def evaluate(self, context: IntelContext) -> Appraisal:
        candidate = _candidate_of(context)
        if candidate is None:
            return Appraisal()
        shaped = apply_curve(composition_cost(candidate.composition), self._curve)
        income = context.economy.mineral_income + context.economy.gas_income
        tightness = 1.0 - apply_curve(income, self._income_curve)
        return Appraisal(base_score=-self._weight * tightness * shaped, multiplier=1.0)


class BeliefConfidence:
    """How much the belief layer trusts the picture — a multiplier, never a term.

    Expected utility, not a reward: a stale picture scales every candidate's
    score down instead of adding a constant to one of them.
    """

    def load(self, params: Mapping[str, Any]) -> None:
        """No tunable parameters — the confidence comes from the belief layer."""

    def evaluate(self, context: IntelContext) -> Appraisal:
        confidence = min(1.0, max(0.0, float(context.belief.confidence)))
        return Appraisal(base_score=0.0, multiplier=confidence, veto=False)


class Commitment:
    """Scored hysteresis for the composition we are already building.

    The incumbent gets a bonus in the score — it is not a latch. The bonus
    holds the crown while a challenger is at most `margin` behind and fades
    over the next margin of score, so the switch happens as the challenger's
    advantage grows instead of at a cliff. The curve shape decides how long the
    bonus holds: a convex curve holds longer than a linear one.
    """

    def __init__(self) -> None:
        self._weight = DEFAULT_WEIGHTS["commitment"]
        self._margin = DEFAULT_COMMITMENT_MARGIN
        self._curve = DEFAULT_CURVES["commitment"]

    def load(self, params: Mapping[str, Any]) -> None:
        self._weight = float(params.get("weight", self._weight))
        self._margin = float(params.get("margin", self._margin))
        self._curve = params.get("curve", self._curve)

    def evaluate(self, context: IntelContext) -> Appraisal:
        """The bonus the incumbent earns — `bonus_for_gap` applies the challenger's cut."""
        return Appraisal(base_score=self._weight, multiplier=1.0, veto=False)

    def bonus_for_gap(self, gap: float) -> float:
        """The bonus after the challenger's gap is known.

        The gap is how far the best challenger leads the incumbent in base score.
        A gap inside the margin keeps the full bonus; past it the bonus fades to
        zero over the next margin.
        """
        fade = max(self._margin, 1e-9)
        headroom = min(1.0, max(0.0, (self._margin + fade - gap) / fade))
        return self._weight * apply_curve(headroom, self._curve)


def economy_capacity(economy: EconomySnapshot) -> float:
    """What the economy can spend per minute on one army member's upkeep.

    Income, scaled by how much of the base is worked out and by how many
    workers we have (a base without workers produces nothing). It is the
    denominator of the economy term's slack.
    """
    return (
        (economy.mineral_income + economy.gas_income)
        * (0.5 + 0.5 * economy.saturation)
        * min(1.0, economy.worker_count / float(ECONOMY_WORKER_CAPACITY))
    )


# ===== CANDIDATE BUILDER =====


def build_candidates(
    profile_compositions: Sequence[tuple[str, dict]],
    enemy_shares: Mapping[str, float],
    hold_composition: Mapping[UnitTypeId, dict] | None = None,
    tech_ready: Callable[[UnitTypeId], bool] | None = None,
    cap: int = CANDIDATE_CAP,
    max_variants: int = MAX_COUNTER_VARIANTS,
) -> list[CompositionCandidate]:
    """Build the ballot of compositions that compete this cycle.

    The ballot is the profile's compositions, the hold option and the
    counter-driven variants, capped at `cap`. The order is fixed — profile
    builds first, then hold, then the variants by counter value — so a tie
    keeps the profile's default build.

    Every candidate on the ballot is buildable. Each composition is filtered
    through the tech we actually have and re-normalized, so a composition
    production cannot execute never reaches the scoring — the 2026-11-09 gate
    failure crowned a robo-centric mix with no Robotics tech and production
    starved on it. The ballot is never empty: the conservative fallback stands
    in when nothing else survives.

    `hold_composition` is the composition we already decided to build (the last
    round's winner). It is the defensive option: it needs no new tech and no
    composition switch. It is deliberately *not* the army's current mix — that
    is a transient state, and a cheap early army would win the price terms and
    stall the build's intended composition.
    """
    candidates: list[CompositionCandidate] = []
    signatures: set[tuple] = set()
    base = profile_compositions[0][1] if profile_compositions else {}

    known_types: set[UnitTypeId] = set(base)
    for _, composition in profile_compositions:
        known_types |= set(composition)
    values = {
        unit_type: _unit_counter_value(unit_type, enemy_shares)
        for unit_type in known_types
    }
    buildable = {
        unit_type
        for unit_type in known_types
        if tech_ready is None or tech_ready(unit_type)
    }
    translation = _translation_map(
        known_types, _profile_vocabulary(profile_compositions), buildable, values
    )

    def _add(name: str, composition: dict, source: str) -> None:
        if not composition:
            return
        signature = _composition_signature(composition)
        if signature in signatures:
            return
        signatures.add(signature)
        candidates.append(
            CompositionCandidate(
                name=name,
                composition=_with_counter_priorities(composition, values),
                source=source,
            )
        )

    for name, composition in profile_compositions:
        _add(name, _buildable_composition(composition, buildable), PROFILE_SOURCE)

    if hold_composition:
        hold = _hold_composition(hold_composition, base, buildable)
        if hold:
            # The hold candidate is never deduplicated: it means "keep building
            # what we decided to build", and the cap must not squeeze the
            # defensive option out of the ballot.
            candidates.append(
                CompositionCandidate(
                    name=HOLD_NAME,
                    composition=_with_counter_priorities(hold, values),
                    source=HOLD_SOURCE,
                )
            )

    for unit_type, variant in _counter_variants(
        _buildable_composition(base, buildable),
        values,
        buildable,
        translation,
        max_variants,
    ):
        _add(f"counter_{unit_type.name}", variant, COUNTER_SOURCE)

    if not candidates:
        _add(FALLBACK_NAME, _fallback_composition(base, buildable), FALLBACK_SOURCE)

    return candidates[:cap]


def _unit_counter_value(
    unit_type: UnitTypeId, enemy_shares: Mapping[str, float]
) -> float:
    """The counter value of one unit type against the believed enemy army."""
    return counter_effectiveness(
        {unit_type: {"proportion": 1.0, "priority": 0}}, enemy_shares
    )


def _profile_vocabulary(
    profile_compositions: Sequence[tuple[str, dict]],
) -> set[UnitTypeId]:
    """The unit types the build actually fields — a share at or above the floor.

    Only these may stand in for a unit we cannot build: substituting a unit
    type the build does not field would be a strategy change, not a
    translation.
    """
    vocabulary: set[UnitTypeId] = set()
    for _, composition in profile_compositions:
        for unit_type, info in composition.items():
            if float(info["proportion"]) >= PRODUCTION_MIN_PROPORTION:
                vocabulary.add(unit_type)
    return vocabulary


def _translation_map(
    unit_types: Iterable[UnitTypeId],
    vocabulary: Iterable[UnitTypeId],
    buildable: set[UnitTypeId],
    values: Mapping[UnitTypeId, float],
) -> dict[UnitTypeId, UnitTypeId]:
    """The nearest buildable stand-in for every unit type we cannot build yet.

    The counter table's ideal is often ahead of our tech (a Colossus before
    the Robotics Bay). ARES holds the prerequisites, so "nearest" is a
    question the tech tree answers: the buildable unit type whose buildings
    differ least from the ideal's. The stand-in inherits the ideal's place in
    the ranking, so the mix moves toward the counter line as far as we can
    reach instead of the ideal being dropped on the floor.

    Ties break on the counter value (the better counter stands in), then on
    price (the cheaper stand-in is the more conservative one) and finally on
    the unit name, so the map is deterministic.
    """
    stand_ins = sorted(set(vocabulary) & buildable, key=lambda item: item.name)
    translation: dict[UnitTypeId, UnitTypeId] = {}
    for unit_type in unit_types:
        if unit_type in buildable:
            continue
        candidates = [other for other in stand_ins if other != unit_type]
        if not candidates:
            continue
        translation[unit_type] = min(
            candidates,
            key=lambda other: (
                _tech_distance(unit_type, other),
                -values.get(other, 0.0),
                _unit_price(other),
                other.name,
            ),
        )
    return translation


def _unit_price(unit_type: UnitTypeId) -> float:
    """What one unit costs in resources — ARES' cost table."""
    cost = COST_DICT.get(unit_type)
    if cost is None:
        return 0.0
    return float(cost.minerals + cost.vespene)


def _buildable_composition(
    composition: Mapping[UnitTypeId, dict], buildable: set[UnitTypeId]
) -> dict:
    """The part of a composition production can actually execute.

    A unit type we have no tech for is dropped, and a unit type below the
    production floor is dropped (it is not worth a production slot). The rest
    is re-normalized: ARES asserts the proportions sum to 1.0, so a filtered
    composition has to be re-balanced before it goes on the ballot.
    """
    filtered = {
        unit_type: {
            "proportion": float(info["proportion"]),
            "priority": int(info["priority"]),
        }
        for unit_type, info in composition.items()
        if unit_type in buildable
        and float(info["proportion"]) >= PRODUCTION_MIN_PROPORTION
    }
    if not filtered:
        return {}
    return _normalize_composition(filtered)


def _conservative_buildable(
    composition: Mapping[UnitTypeId, dict], buildable: set[UnitTypeId]
) -> dict:
    """One cheap unit type we can build — the conservative stand-in.

    Used when filtering would empty a composition: the ballot has to keep a
    candidate, and the cheapest thing we can actually produce is the safest
    thing to keep building.
    """
    fielded = [
        unit_type
        for unit_type, info in composition.items()
        if unit_type in buildable
        and float(info["proportion"]) >= PRODUCTION_MIN_PROPORTION
    ]
    pool = fielded or [unit_type for unit_type in composition if unit_type in buildable]
    if not pool:
        return {}
    cheapest = min(pool, key=lambda unit_type: (_unit_price(unit_type), unit_type.name))
    return {cheapest: {"proportion": 1.0, "priority": 0}}


def _fallback_composition(
    base: Mapping[UnitTypeId, dict], buildable: set[UnitTypeId]
) -> dict:
    """The candidate that keeps the ballot alive when nothing else survives.

    Two tiers. The cheapest unit type of the build's own mix we can produce,
    or — when nothing is buildable at all — the profile's mix as it is. In
    that degenerate state every composition is equally unexecutable and an
    empty composition would break ARES' proportion assert, so the profile's
    own mix stands in and production stays idle until tech arrives.
    """
    conservative = _conservative_buildable(base, buildable)
    if conservative:
        return conservative
    if not base:
        return {}
    return _normalize_composition(
        {unit_type: dict(info) for unit_type, info in base.items()}
    )


def _hold_composition(
    hold: Mapping[UnitTypeId, dict],
    base: Mapping[UnitTypeId, dict],
    buildable: set[UnitTypeId],
) -> dict:
    """The composition we already build, restricted to what we can build.

    Continuity is the point: the hold candidate needs no new tech and no
    composition switch, and the priorities stay the profile's build order.
    The hold is filtered like every other candidate — but it must never die.
    When the build's base mix does not field the incumbent's units, the
    incumbent's own buildable part stands in; when even that is empty (the
    incumbent fields nothing we can build), the build's own cheapest buildable
    unit keeps the defensive option alive.
    """
    composition = {
        unit_type: info for unit_type, info in hold.items() if unit_type in base
    }
    filtered = _buildable_composition(composition, buildable)
    if filtered:
        return filtered
    filtered = _buildable_composition(hold, buildable)
    if filtered:
        return filtered
    conservative = _conservative_buildable(hold, buildable)
    if conservative:
        return conservative
    return _conservative_buildable(base, buildable)


def _counter_variants(
    base: Mapping[UnitTypeId, dict],
    values: Mapping[UnitTypeId, float],
    buildable: set[UnitTypeId],
    translation: Mapping[UnitTypeId, UnitTypeId],
    max_variants: int,
) -> list[tuple[UnitTypeId, dict]]:
    """Profile mixes shifted toward the units that counter the believed enemy.

    The ideal counter is the unit type the scoring table likes best. When we
    cannot build it yet, the walk does not give up: it lands on the nearest
    buildable approximation — the stand-in that sits closest on ARES' tech
    tree — and shifts toward that instead of dropping the ideal. Every shift
    target is buildable, so every variant is a composition production can
    execute.
    """
    if not base or not values:
        return []
    ranked = sorted(values.items(), key=lambda pair: (-pair[1], pair[0].name))
    variants: list[tuple[UnitTypeId, dict]] = []
    targets: set[UnitTypeId] = set()
    for ideal, value in ranked:
        if len(variants) >= max_variants:
            break
        if value <= 0.0:
            break
        target = ideal if ideal in buildable else translation.get(ideal)
        if target is None or target in targets:
            continue
        if values.get(target, 0.0) <= 0.0:
            # The stand-in does not counter anything — shifting toward it would
            # buy nothing.
            continue
        variant = _shift_proportion(base, target, values, buildable)
        if variant is None:
            continue
        targets.add(target)
        variants.append((target, variant))
    return variants


def _shift_proportion(
    base: Mapping[UnitTypeId, dict],
    target: UnitTypeId,
    values: Mapping[UnitTypeId, float],
    buildable: set[UnitTypeId],
) -> dict | None:
    """Move proportion from the least effective unit we can build to the counter."""
    donors = [
        unit_type
        for unit_type, info in base.items()
        if float(info["proportion"]) > PRODUCTION_MIN_PROPORTION
        and unit_type != target
        and unit_type in buildable
    ]
    if not donors:
        return None
    donor = min(donors, key=lambda unit_type: (values.get(unit_type, 0.0), unit_type.name))
    donor_proportion = float(base[donor]["proportion"])
    amount = min(COUNTER_VARIANT_SHIFT, donor_proportion - PRODUCTION_MIN_PROPORTION)
    if amount <= 0.0:
        return None
    if target not in base and amount < PRODUCTION_MIN_PROPORTION:
        return None
    variant = {
        unit_type: {
            "proportion": float(info["proportion"]),
            "priority": int(info["priority"]),
        }
        for unit_type, info in base.items()
    }
    variant[donor]["proportion"] = donor_proportion - amount
    if target in variant:
        variant[target]["proportion"] += amount
    else:
        variant[target] = {
            "proportion": amount,
            "priority": max(info["priority"] for info in variant.values()) + 1,
        }
    return _normalize_composition(variant)


def _with_counter_priorities(
    composition: Mapping[UnitTypeId, dict], values: Mapping[UnitTypeId, float]
) -> dict:
    """Build the most effective unit first — the priority order the table implies.

    The old nudge did the same after the proportions moved; here it is part of
    the candidate, so the winner arrives with its build order already set.
    """
    ordered = sorted(
        composition,
        key=lambda unit_type: (-values.get(unit_type, 0.0), int(composition[unit_type]["priority"])),
    )
    return {
        unit_type: {
            "proportion": float(composition[unit_type]["proportion"]),
            "priority": index,
        }
        for index, unit_type in enumerate(ordered)
    }


def _composition_signature(composition: Mapping[UnitTypeId, dict]) -> tuple:
    """A comparable fingerprint of a composition — used to drop duplicates."""
    return tuple(
        sorted(
            ((unit_type, round(float(info["proportion"]), 4))
             for unit_type, info in composition.items()),
            key=lambda pair: pair[0].name,
        )
    )


def _normalize_composition(composition: Mapping[UnitTypeId, dict]) -> dict:
    """Round the proportions and make them sum to exactly 1.0 (ARES asserts it)."""
    total = sum(float(info["proportion"]) for info in composition.values())
    if total <= 0.0:
        return {unit_type: dict(info) for unit_type, info in composition.items()}
    normalized = {
        unit_type: {
            "proportion": round(float(info["proportion"]) / total, 4),
            "priority": int(info["priority"]),
        }
        for unit_type, info in composition.items()
    }
    residual = round(
        1.0 - sum(info["proportion"] for info in normalized.values()), 4
    )
    if normalized:
        holder = max(normalized, key=lambda unit_type: normalized[unit_type]["proportion"])
        normalized[holder]["proportion"] = round(
            normalized[holder]["proportion"] + residual, 4
        )
    return normalized


# ===== SCORER =====


class CompositionScorer:
    """Scores composition candidates and hands the winner to the macro manager.

    The scorer owns the reasoner wiring: it builds the IntelContext from the
    belief layer on the macro cadence, builds the candidate ballot, ranks the
    candidates through the canonical runner and keeps the incumbent so the
    commitment bonus can hold the crown.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        considerations: Sequence[Consideration] | None = None,
        commitment: Commitment | None = None,
    ) -> None:
        self._config = config or {}
        self._considerations = (
            list(considerations)
            if considerations is not None
            else _build_considerations(self._config)
        )
        self._commitment = (
            commitment if commitment is not None else _build_commitment(self._config)
        )
        self._interval = _rescore_interval(self._config)
        self._cycle = 0
        self._last_rescore = 0
        self._signature: tuple | None = None
        self._selection: CompositionSelection | None = None
        self._incumbent: str | None = None
        self._incumbent_composition: dict[UnitTypeId, dict] | None = None

    @property
    def considerations(self) -> tuple[Consideration, ...]:
        """The terms every candidate is scored on."""
        return tuple(self._considerations)

    @property
    def incumbent(self) -> str | None:
        """The candidate that currently holds the crown."""
        return self._incumbent

    def rank(
        self,
        candidates: Sequence[CompositionCandidate],
        context: IntelContext,
        incumbent: str | None = None,
    ) -> list[RankedChoice]:
        """Rank the candidates on the canonical score.

        The incumbent's commitment bonus needs the challenger's gap, so the
        ranking runs in two steps: the terms first, then the bonus, then the
        canonical runner over the finished choices.
        """
        choices: list[Choice] = []
        base_sums: list[float] = []
        for candidate in candidates:
            appraisals = tuple(
                consideration.evaluate(_candidate_context(context, candidate))
                for consideration in self._considerations
            )
            choices.append(Choice(name=candidate.name, appraisals=appraisals))
            base_sums.append(combine(appraisals).base_score)

        index = self._incumbent_index(candidates, incumbent)
        if index is not None:
            bonus = self._commitment.bonus_for_gap(
                _challenger_gap(base_sums, index)
            )
            if bonus:
                choices[index] = Choice(
                    name=choices[index].name,
                    appraisals=choices[index].appraisals
                    + (Appraisal(base_score=bonus),),
                )

        return score(choices)

    def select(self, bot) -> CompositionSelection | None:
        """Pick the composition to build, re-scoring on the macro cadence.

        A re-score happens when the belief picture changes or every
        `RescoreInterval` cycles — the composition is a macro decision, not a
        per-frame one.
        """
        self._cycle += 1
        observed = self._observed_shares(bot)
        confidence = self._confidence(bot)
        signature = (
            tuple(sorted((name, round(share, 2)) for name, share in observed.items())),
            round(confidence, 2),
        )
        due = (
            self._selection is None
            or self._cycle - self._last_rescore >= self._interval
            or signature != self._signature
        )
        if due:
            self._rescore(bot, observed, confidence, signature)
        return self._selection

    def build_context(
        self,
        bot,
        observed: Mapping[str, float] | None = None,
        confidence: float | None = None,
        predicted: Mapping[str, float] | None = None,
    ) -> IntelContext:
        """Build the snapshot every consideration scores against.

        The believed enemy army blends what we observed with what the strategy
        belief predicts, weighted by how much the belief layer trusts itself.
        """
        observed = self._observed_shares(bot) if observed is None else observed
        confidence = self._confidence(bot) if confidence is None else confidence
        predicted = self._predicted_shares(bot) if predicted is None else predicted
        return IntelContext(
            belief=BeliefSnapshot(
                enemy_composition=_blend_shares(observed, predicted, confidence),
                strategy=self._strategy_probs(bot),
                confidence=confidence,
            ),
            intel_urgency=float(getattr(bot, "_intel_urgency", 0.0)),
            intel_freshness=confidence,
            economy=self._economy(bot),
            tech_progress=self._tech_progress(bot),
        )

    def _rescore(
        self,
        bot,
        observed: Mapping[str, float],
        confidence: float,
        signature: tuple,
    ) -> None:
        """Run one scoring round and cache the winner."""
        context = self.build_context(
            bot, observed=observed, confidence=confidence
        )
        profile = get_active_profile(bot)
        profile_compositions = [
            ("profile_0", profile.army_composition_0),
            ("profile_1", profile.army_composition_1),
        ]
        if profile.army_composition_2:
            profile_compositions.append(("profile_2", profile.army_composition_2))
        candidates = build_candidates(
            profile_compositions,
            context.belief.enemy_composition,
            hold_composition=self._incumbent_composition,
            tech_ready=bot.tech_ready_for_unit,
        )
        ranking = self.rank(candidates, context, incumbent=self._incumbent)
        by_name = {candidate.name: candidate for candidate in candidates}
        winner = by_name.get(ranking[0].choice.name) if ranking else None
        self._incumbent = winner.name if winner else None
        self._incumbent_composition = winner.composition if winner else None
        self._selection = CompositionSelection(
            candidate=winner,
            ranking=ranking,
            context=context,
            candidates=candidates,
        )
        self._signature = signature
        self._last_rescore = self._cycle

    @staticmethod
    def _incumbent_index(
        candidates: Sequence[CompositionCandidate], incumbent: str | None
    ) -> int | None:
        """Where the incumbent sits on the ballot.

        By name if the incumbent is still there; otherwise the hold candidate,
        which carries the incumbent's composition.
        """
        if incumbent is None:
            return None
        for position, candidate in enumerate(candidates):
            if candidate.name == incumbent:
                return position
        for position, candidate in enumerate(candidates):
            if candidate.source == HOLD_SOURCE:
                return position
        return None

    # ----- CONTEXT SOURCES -----

    @staticmethod
    def _observed_shares(bot) -> dict[str, float]:
        """Probability-weighted shares of the enemy army, by unit type name."""
        belief = getattr(bot, "belief_state", None)
        composition = getattr(belief, "composition", None)
        if composition is None:
            return {}
        weighted = composition.get_weighted_army(
            exclude_workers=True,
            exclude_structures=True,
            exclude_ignored=True,
            min_confidence=0.01,
        )
        counts: dict[str, float] = {}
        for item in weighted:
            name = item.unit.type_id.name
            counts[name] = counts.get(name, 0.0) + float(item.confidence)
        total = sum(counts.values())
        if total <= 0.0:
            return {}
        return {name: count / total for name, count in counts.items()}

    @staticmethod
    def _confidence(bot) -> float:
        """How much the belief layer trusts the picture (0-1)."""
        belief = getattr(bot, "belief_state", None)
        composition = getattr(belief, "composition", None)
        if composition is not None:
            return min(1.0, max(0.0, float(composition.freshness)))
        from bot.intel import get_enemy_intel_quality

        return min(1.0, max(0.0, float(get_enemy_intel_quality(bot)["freshness"])))

    @staticmethod
    def _predicted_shares(bot) -> dict[str, float]:
        """What the strategy belief says we are about to face."""
        belief = getattr(bot, "belief_state", None)
        strategy = getattr(belief, "strategy", None)
        prediction = getattr(strategy, "last_prediction", None)
        if prediction is None:
            return {}
        race = getattr(bot, "enemy_race", None)
        counts: dict[str, float] = {}
        for category, probability in prediction.probs.items():
            expected = STRATEGY_EXPECTED_UNITS.get((category, race))
            if not expected:
                continue
            for unit_type, share in expected.items():
                if (
                    unit_type in WORKER_TYPES
                    or unit_type in ALL_STRUCTURES
                    or unit_type == UnitTypeId.MULE
                ):
                    continue
                name = unit_type.name
                counts[name] = counts.get(name, 0.0) + float(probability) * float(share)
        total = sum(counts.values())
        if total <= 0.0:
            return {}
        return {name: count / total for name, count in counts.items()}

    @staticmethod
    def _strategy_probs(bot) -> dict[str, float]:
        """The strategy belief's category probabilities, by label."""
        belief = getattr(bot, "belief_state", None)
        strategy = getattr(belief, "strategy", None)
        prediction = getattr(strategy, "last_prediction", None)
        if prediction is None:
            return {}
        return {
            category.value: float(probability)
            for category, probability in prediction.probs.items()
        }

    @staticmethod
    def _economy(bot) -> EconomySnapshot:
        """Our economy: workers, income and how much of the base is worked out."""
        workers = bot.workers.amount
        bases = max(1, len(bot.townhalls))
        saturation = min(
            1.0, workers / float(SATURATION_WORKERS_PER_BASE * bases)
        )
        return EconomySnapshot(
            worker_count=workers,
            mineral_income=float(bot.state.score.collection_rate_minerals),
            gas_income=float(bot.state.score.collection_rate_vespene),
            saturation=saturation,
        )

    @staticmethod
    def _tech_progress(bot) -> dict[str, float]:
        """The finished upgrades a composition could exploit, by name."""
        upgrades = getattr(bot.state, "upgrades", None) or set()
        return {
            upgrade.name: 1.0
            for upgrade in upgrades
            if upgrade.name in UPGRADE_SYNERGY_UNITS
        }


# ===== HELPERS =====


def _candidate_context(
    context: IntelContext, candidate: CompositionCandidate
) -> CompositionContext:
    """The shared snapshot plus the candidate under evaluation."""
    return CompositionContext(
        belief=context.belief,
        threat_severity=context.threat_severity,
        intel_urgency=context.intel_urgency,
        intel_freshness=context.intel_freshness,
        economy=context.economy,
        tech_progress=context.tech_progress,
        candidate=candidate,
    )


def _challenger_gap(base_sums: Sequence[float], incumbent_index: int) -> float:
    """How far the best challenger leads the incumbent (0 when it stands alone)."""
    others = [
        value for index, value in enumerate(base_sums) if index != incumbent_index
    ]
    if not others:
        return 0.0
    return max(others) - base_sums[incumbent_index]


def _blend_shares(
    observed: Mapping[str, float],
    predicted: Mapping[str, float],
    confidence: float,
) -> dict[str, float]:
    """Blend the observed army with the strategy prior by how much we trust it."""
    if not observed:
        return dict(predicted)
    if not predicted:
        return dict(observed)
    blended = {
        name: confidence * observed.get(name, 0.0)
        + (1.0 - confidence) * predicted.get(name, 0.0)
        for name in set(observed) | set(predicted)
    }
    total = sum(blended.values())
    if total <= 0.0:
        return {}
    return {name: value / total for name, value in blended.items()}


def _build_considerations(config: Mapping[str, Any]) -> list[Consideration]:
    """The standard term set, loaded from config.yml."""
    weights = _weights(config)
    counter = CounterEffectiveness()
    counter.load({"weight": weights["counter"], "curve": _curve(config, "counter")})
    economy = EconomyFit()
    economy.load({"weight": weights["economy"], "curve": _curve(config, "economy")})
    tech = TechSynergy()
    tech.load({"weight": weights["tech"], "curve": _curve(config, "tech")})
    cost = CostBurden()
    cost.load(
        {
            "weight": weights["cost"],
            "curve": _curve(config, "cost"),
            "income_curve": _curve(config, "income"),
        }
    )
    belief = BeliefConfidence()
    belief.load({})
    return [counter, economy, tech, cost, belief]


def _build_commitment(config: Mapping[str, Any]) -> Commitment:
    """The commitment term, loaded from config.yml."""
    weights = _weights(config)
    commitment = Commitment()
    commitment.load(
        {
            "weight": weights["commitment"],
            "margin": _commitment_margin(config),
            "curve": _curve(config, "commitment"),
        }
    )
    return commitment


def _weights(config: Mapping[str, Any]) -> dict[str, float]:
    """The term weights — config.yml overrides the seeds."""
    weights = dict(DEFAULT_WEIGHTS)
    raw = _lookup(config, "UtilityAI", "Composition", "Weights")
    if isinstance(raw, Mapping):
        for name, value in raw.items():
            try:
                weights[name] = float(value)
            except (TypeError, ValueError):
                continue
    return weights


def _commitment_margin(config: Mapping[str, Any]) -> float:
    """The score margin a challenger must beat to take the crown."""
    raw = _lookup(config, "UtilityAI", "Composition", "CommitmentMargin")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return DEFAULT_COMMITMENT_MARGIN


def _curve(config: Mapping[str, Any], name: str) -> CurveParams:
    """One named response curve — config.yml owns the personality."""
    entry = _lookup(config, "UtilityAI", "Curves", name)
    if not isinstance(entry, Mapping):
        return DEFAULT_CURVES[name]
    return load_curve_params(config, name)


def _rescore_interval(config: Mapping[str, Any]) -> int:
    """How many macro cycles pass between forced re-scores."""
    raw = _lookup(config, "UtilityAI", "RescoreInterval")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_RESCORE_INTERVAL


def _lookup(config: Mapping[str, Any], *keys: str) -> Any:
    """Walk a nested mapping, tolerating a missing or malformed section."""
    current: Any = config
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current
