"""Unit tests for the composition scoring subsystem (bot/managers/composition_scoring.py).

Purpose: pin the composition-selection math before it drives production — the
    counter sums that come out of SCORING_TABLE, the signed cost base, the
    belief-confidence multiplier, the incumbent hysteresis, the candidate cap
    and the deterministic tie order.
Key Decisions: the tests construct IntelContext and the candidates directly, so
    no game, no bot instance and no SC2 binary is needed; the shipped
    config.yml is the source of the curve shapes and the term weights.
Limitations: the tests pin the arithmetic and the shapes, not the personality
    values themselves — Task 8 refits those from telemetry.
"""

import json
from math import isclose
from pathlib import Path

import pytest
import yaml

from bot.combat.scoring_table import SCORING_TABLE
from bot.managers.composition_scoring import (
    CANDIDATE_CAP,
    BeliefConfidence,
    Commitment,
    CompositionCandidate,
    CompositionContext,
    CompositionScorer,
    CounterEffectiveness,
    CostBurden,
    EconomyFit,
    TechSynergy,
    _buildable_composition,
    _tech_distance,
    _translation_map,
    build_candidates,
    composition_cost,
    counter_effectiveness,
)
from bot.utilities.reasoner import (
    Appraisal,
    BeliefSnapshot,
    Choice,
    CurveParams,
    EconomySnapshot,
    IntelContext,
    apply_curve,
    combine,
    load_curve_params,
    score,
)
from sc2.ids.unit_typeid import UnitTypeId

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((REPO_ROOT / "config.yml").read_text(encoding="utf-8"))

REMOVED_SWITCHES = (
    "_economy_switch_triggered",
    "economy_switch_threshold",
    "archon_switch_threshold",
)
"""The one-way economy switch and the archon percentage switch — Task 4 removes
them, so the tree must not mention them any more."""


def _config_curve(name: str) -> CurveParams:
    """The shipped curve parameters — config.yml is the personality's home."""
    return load_curve_params(CONFIG, name)


def _weight(name: str) -> float:
    """The shipped weight of one scoring term."""
    return float(CONFIG["UtilityAI"]["Composition"]["Weights"][name])


def _composition(*pairs: tuple[UnitTypeId, float]) -> dict:
    """An ARES-shaped composition: {unit type: {proportion, priority}}."""
    return {
        unit_type: {"proportion": proportion, "priority": index}
        for index, (unit_type, proportion) in enumerate(pairs)
    }


def _context(
    candidate: CompositionCandidate | None = None,
    enemy_composition: dict | None = None,
    confidence: float = 1.0,
    economy: EconomySnapshot | None = None,
    tech_progress: dict | None = None,
) -> CompositionContext:
    """An IntelContext with the candidate attached — the scorer's test seam."""
    return CompositionContext(
        belief=BeliefSnapshot(
            enemy_composition=enemy_composition or {}, confidence=confidence
        ),
        economy=economy or EconomySnapshot(),
        tech_progress=tech_progress or {},
        candidate=candidate,
    )


def _candidate(name: str, *pairs: tuple[UnitTypeId, float]) -> CompositionCandidate:
    return CompositionCandidate(
        name=name, composition=_composition(*pairs), source="profile"
    )


# ===== COUNTER EFFECTIVENESS =====


def test_counter_effectiveness_reads_the_scoring_table() -> None:
    """The core base score is the scoring table — nothing else."""
    composition = _composition((UnitTypeId.IMMORTAL, 1.0))
    shares = {"ROACH": 1.0}
    assert counter_effectiveness(composition, shares) == pytest.approx(
        SCORING_TABLE[(UnitTypeId.IMMORTAL, UnitTypeId.ROACH)]
    )


def test_counter_effectiveness_weights_the_enemy_shares() -> None:
    """A mixed enemy army is scored by its shares, not by unit counts."""
    composition = _composition((UnitTypeId.IMMORTAL, 1.0))
    shares = {"ROACH": 0.75, "RAVAGER": 0.25}
    expected = (
        0.75 * SCORING_TABLE[(UnitTypeId.IMMORTAL, UnitTypeId.ROACH)]
        + 0.25 * SCORING_TABLE[(UnitTypeId.IMMORTAL, UnitTypeId.RAVAGER)]
    )
    assert counter_effectiveness(composition, shares) == pytest.approx(expected)


def test_counter_effectiveness_weights_our_own_proportions() -> None:
    """Our mix decides how much of the counter value we actually field."""
    composition = _composition((UnitTypeId.IMMORTAL, 0.5), (UnitTypeId.ZEALOT, 0.5))
    shares = {"ZERGLING": 1.0}
    expected = 0.5 * (
        SCORING_TABLE.get((UnitTypeId.IMMORTAL, UnitTypeId.ZERGLING), 0.0)
        + SCORING_TABLE.get((UnitTypeId.ZEALOT, UnitTypeId.ZERGLING), 0.0)
    )
    assert counter_effectiveness(composition, shares) == pytest.approx(expected)


def test_counter_effectiveness_excludes_situational_rows() -> None:
    """Fixed-count support rows (Sentry vs lings) are not per-enemy damage."""
    from bot.combat.scoring_table import SITUATIONAL_ROWS

    assert (UnitTypeId.SENTRY, UnitTypeId.ZERGLING) in SITUATIONAL_ROWS
    composition = _composition((UnitTypeId.SENTRY, 1.0))
    assert counter_effectiveness(composition, {"ZERGLING": 1.0}) == pytest.approx(0.0)


def test_counter_effectiveness_without_intel_is_zero() -> None:
    """No enemy picture → no counter claim. The term stays silent, not negative."""
    composition = _composition((UnitTypeId.IMMORTAL, 1.0))
    assert counter_effectiveness(composition, {}) == pytest.approx(0.0)


# ===== TERM MATH =====


def test_counter_term_is_the_curved_table_sum() -> None:
    """The counter term is the curve of the table sum, weighted — no raw values."""
    consideration = CounterEffectiveness()
    consideration.load({"weight": _weight("counter"), "curve": _config_curve("counter")})

    candidate = _candidate("immortals", (UnitTypeId.IMMORTAL, 1.0))
    context = _context(candidate=candidate, enemy_composition={"ROACH": 1.0})
    appraisal = consideration.evaluate(context)

    raw = SCORING_TABLE[(UnitTypeId.IMMORTAL, UnitTypeId.ROACH)]
    expected = _weight("counter") * (
        apply_curve(raw, _config_curve("counter")) - 0.5
    )
    assert appraisal.base_score == pytest.approx(expected)


def test_counter_curve_is_centred_on_the_neutral_matchup() -> None:
    """A raw counter value of zero is the curve's midpoint → no score either way."""
    curve = _config_curve("counter")
    assert apply_curve(0.0, curve) == pytest.approx(0.5)


def test_cost_term_is_a_negative_base() -> None:
    """Cost is a negative base — a burden, never a veto."""
    consideration = CostBurden()
    consideration.load({"weight": _weight("cost"), "curve": _config_curve("cost")})

    candidate = _candidate("zealots", (UnitTypeId.ZEALOT, 1.0))
    context = _context(candidate=candidate)
    appraisal = consideration.evaluate(context)

    assert appraisal.base_score <= 0.0
    assert appraisal.veto is False


def test_cost_term_charges_the_heavier_mix_more() -> None:
    """The resource + production-time burden separates the mixes."""
    consideration = CostBurden()
    consideration.load({"weight": _weight("cost"), "curve": _config_curve("cost")})

    cheap = _candidate("cheap", (UnitTypeId.ZEALOT, 1.0))
    heavy = _candidate("heavy", (UnitTypeId.COLOSSUS, 1.0))
    context = _context()

    cheap_score = consideration.evaluate(_context(candidate=cheap)).base_score
    heavy_score = consideration.evaluate(_context(candidate=heavy)).base_score
    assert heavy_score < cheap_score


def test_cost_term_fades_as_the_economy_warms_up() -> None:
    """A rich economy stops caring what a unit costs — the burden is scaled."""
    consideration = CostBurden()
    consideration.load({"weight": _weight("cost"), "curve": _config_curve("cost")})

    candidate = _candidate("colossi", (UnitTypeId.COLOSSUS, 1.0))
    poor = _context(
        candidate=candidate, economy=EconomySnapshot(mineral_income=100.0)
    )
    rich = _context(
        candidate=candidate, economy=EconomySnapshot(mineral_income=2000.0)
    )
    assert abs(consideration.evaluate(rich).base_score) < abs(
        consideration.evaluate(poor).base_score
    )


def test_composition_cost_counts_resources_and_supply() -> None:
    """Time to produce is part of the burden: supply is priced in resources."""
    zealot_cost = composition_cost(_composition((UnitTypeId.ZEALOT, 1.0)))
    colossus_cost = composition_cost(_composition((UnitTypeId.COLOSSUS, 1.0)))
    assert zealot_cost > 0.0
    assert colossus_cost > zealot_cost


def test_economy_term_penalises_a_mix_the_economy_cannot_pay_for() -> None:
    """When the bank is empty, the cheap mix outscores the expensive one."""
    consideration = EconomyFit()
    consideration.load(
        {"weight": _weight("economy"), "curve": _config_curve("economy")}
    )

    cheap = _candidate("cheap", (UnitTypeId.ZEALOT, 1.0))
    heavy = _candidate("heavy", (UnitTypeId.COLOSSUS, 1.0))
    poor = EconomySnapshot(worker_count=12, mineral_income=200.0, saturation=0.5)

    cheap_score = consideration.evaluate(
        _context(candidate=cheap, economy=poor)
    ).base_score
    heavy_score = consideration.evaluate(
        _context(candidate=heavy, economy=poor)
    ).base_score
    assert cheap_score > heavy_score


def test_economy_term_saturates_when_the_economy_is_rich() -> None:
    """A rich economy cannot separate the mixes — affordability is not a choice."""
    consideration = EconomyFit()
    consideration.load(
        {"weight": _weight("economy"), "curve": _config_curve("economy")}
    )

    cheap = _candidate("cheap", (UnitTypeId.ZEALOT, 1.0))
    heavy = _candidate("heavy", (UnitTypeId.COLOSSUS, 1.0))
    rich = EconomySnapshot(
        worker_count=50, mineral_income=2000.0, gas_income=400.0, saturation=1.0
    )

    cheap_score = consideration.evaluate(_context(candidate=cheap, economy=rich)).base_score
    heavy_score = consideration.evaluate(_context(candidate=heavy, economy=rich)).base_score
    assert cheap_score == pytest.approx(heavy_score)


def test_tech_term_rewards_the_upgrades_the_mix_uses() -> None:
    """A composition that exploits our researched upgrades scores higher."""
    consideration = TechSynergy()
    consideration.load({"weight": _weight("tech"), "curve": _config_curve("tech")})

    zealots = _candidate("zealots", (UnitTypeId.ZEALOT, 1.0))
    immortals = _candidate("immortals", (UnitTypeId.IMMORTAL, 1.0))
    context = _context(tech_progress={"CHARGE": 1.0})

    assert consideration.evaluate(
        _context(candidate=zealots, tech_progress={"CHARGE": 1.0})
    ).base_score > consideration.evaluate(
        _context(candidate=immortals, tech_progress={"CHARGE": 1.0})
    ).base_score


def test_tech_term_is_silent_without_researched_upgrades() -> None:
    """Nothing researched → nothing to exploit → the term stays out of the way."""
    consideration = TechSynergy()
    consideration.load({"weight": _weight("tech"), "curve": _config_curve("tech")})

    candidate = _candidate("zealots", (UnitTypeId.ZEALOT, 1.0))
    assert consideration.evaluate(_context(candidate=candidate)).base_score == pytest.approx(
        0.0
    )


# ===== BELIEF CONFIDENCE =====


def test_belief_confidence_is_a_multiplier_not_a_term() -> None:
    """Confidence scales the score — it never adds to it."""
    consideration = BeliefConfidence()
    consideration.load({})
    appraisal = consideration.evaluate(_context(confidence=0.4))

    assert appraisal.base_score == pytest.approx(0.0)
    assert appraisal.multiplier == pytest.approx(0.4)


def test_the_canonical_score_is_the_base_sum_times_the_confidence() -> None:
    """score = (Σ base_scores) × confidence — the plan's canonical form."""
    confidence = 0.25
    base_sum = 2.0
    ranked = score(
        [
            Choice(
                name="candidate",
                appraisals=(
                    Appraisal(base_score=base_sum),
                    Appraisal(base_score=0.0, multiplier=confidence),
                ),
            )
        ]
    )
    assert ranked[0].score == pytest.approx(base_sum * confidence)


# ===== CANDIDATES =====


def test_candidates_are_capped() -> None:
    """At most eight compositions compete in one cycle."""
    profiles = [
        ("profile_0", _composition((UnitTypeId.ZEALOT, 1.0))),
        ("profile_1", _composition((UnitTypeId.ZEALOT, 0.5), (UnitTypeId.STALKER, 0.5))),
        ("profile_2", _composition((UnitTypeId.ZEALOT, 0.25), (UnitTypeId.STALKER, 0.75))),
    ]
    shares = {
        "ROACH": 0.3,
        "ZERGLING": 0.3,
        "HYDRALISK": 0.2,
        "MUTALISK": 0.1,
        "BANELETING": 0.1,
    }
    candidates = build_candidates(profiles, shares, tech_ready=lambda unit: True)
    assert len(candidates) <= CANDIDATE_CAP


def test_the_hold_candidate_is_always_on_the_ballot() -> None:
    """The defensive option must never be squeezed out by the cap."""
    profiles = [
        ("profile_0", _composition((UnitTypeId.ZEALOT, 1.0))),
        ("profile_1", _composition((UnitTypeId.ZEALOT, 0.5), (UnitTypeId.STALKER, 0.5))),
        ("profile_2", _composition((UnitTypeId.ZEALOT, 0.25), (UnitTypeId.STALKER, 0.75))),
    ]
    shares = {
        "ROACH": 0.3,
        "ZERGLING": 0.3,
        "HYDRALISK": 0.2,
        "MUTALISK": 0.1,
        "BANELETING": 0.1,
    }
    hold = {UnitTypeId.ZEALOT: {"proportion": 1.0, "priority": 0}}
    candidates = build_candidates(
        profiles, shares, hold_composition=hold, tech_ready=lambda unit: True
    )
    assert any(candidate.source == "hold" for candidate in candidates)


def test_candidate_order_is_deterministic() -> None:
    """Same inputs → same ballot, so the argmax is reproducible run to run."""
    profiles = [
        ("profile_0", _composition((UnitTypeId.IMMORTAL, 0.5), (UnitTypeId.ZEALOT, 0.5))),
        ("profile_1", _composition((UnitTypeId.IMMORTAL, 0.8), (UnitTypeId.ZEALOT, 0.2))),
    ]
    shares = {"ROACH": 0.7, "ZERGLING": 0.3}
    first = build_candidates(profiles, shares, tech_ready=lambda unit: True)
    second = build_candidates(profiles, shares, tech_ready=lambda unit: True)
    assert [candidate.name for candidate in first] == [
        candidate.name for candidate in second
    ]


def test_a_counter_variant_shifts_proportion_toward_the_best_counter() -> None:
    """The counter table moves real proportion, not just a priority."""
    profiles = [
        ("profile_0", _composition((UnitTypeId.ZEALOT, 0.5), (UnitTypeId.IMMORTAL, 0.5))),
    ]
    shares = {"ROACH": 1.0}
    candidates = build_candidates(profiles, shares, tech_ready=lambda unit: True)

    variants = [c for c in candidates if c.source == "counter"]
    assert variants, "a roach army must produce a counter variant"
    variant = variants[0]
    base = profiles[0][1]
    assert variant.composition[UnitTypeId.IMMORTAL]["proportion"] > base[
        UnitTypeId.IMMORTAL
    ]["proportion"]
    assert variant.composition[UnitTypeId.ZEALOT]["proportion"] < base[
        UnitTypeId.ZEALOT
    ]["proportion"]


def test_candidates_are_valid_ares_compositions() -> None:
    """ARES asserts the proportions sum to 1.0 and the priorities are in range."""
    profiles = [
        ("profile_0", _composition((UnitTypeId.ZEALOT, 0.5), (UnitTypeId.IMMORTAL, 0.5))),
        ("profile_1", _composition((UnitTypeId.ZEALOT, 0.3), (UnitTypeId.IMMORTAL, 0.7))),
    ]
    shares = {"ROACH": 0.6, "ZERGLING": 0.4}
    hold = {UnitTypeId.ZEALOT: {"proportion": 1.0, "priority": 0}}
    candidates = build_candidates(
        profiles, shares, hold_composition=hold, tech_ready=lambda u: True
    )
    assert candidates
    for candidate in candidates:
        proportions = [
            float(info["proportion"]) for info in candidate.composition.values()
        ]
        assert isclose(sum(proportions), 1.0, abs_tol=1e-3)
        assert all(value >= 0.0 for value in proportions)
        for info in candidate.composition.values():
            assert 0 <= int(info["priority"]) < 11


def test_variants_only_use_units_we_can_build() -> None:
    """A counter we cannot produce is not a candidate."""
    profiles = [
        ("profile_0", _composition((UnitTypeId.ZEALOT, 1.0))),
    ]
    shares = {"ROACH": 1.0}
    candidates = build_candidates(
        profiles, shares, tech_ready=lambda unit: unit == UnitTypeId.ZEALOT
    )
    for candidate in candidates:
        if candidate.source != "counter":
            continue
        assert set(candidate.composition) == {UnitTypeId.ZEALOT}


# ===== BUILDABILITY INVARIANT =====

ROBO_ARMY = {
    UnitTypeId.IMMORTAL: {"proportion": 0.15, "priority": 1},
    UnitTypeId.COLOSSUS: {"proportion": 0.15, "priority": 4},
    UnitTypeId.HIGHTEMPLAR: {"proportion": 0.4, "priority": 0},
    UnitTypeId.DISRUPTOR: {"proportion": 0.1, "priority": 5},
    UnitTypeId.STALKER: {"proportion": 0.0, "priority": 2},
    UnitTypeId.ZEALOT: {"proportion": 0.2, "priority": 3},
}
"""The robo-centric mix of the standard PvT profile — the composition that
won the 2026-11-09 gate with 74% of its counter score coming from units the
bot had no tech for."""

TECH_SCENARIOS = {
    "no tech": set(),
    "gateway only": {UnitTypeId.ZEALOT},
    "gateway + cyber core": {UnitTypeId.ZEALOT, UnitTypeId.STALKER},
    "gateway + cyber core + robo": {
        UnitTypeId.ZEALOT,
        UnitTypeId.STALKER,
        UnitTypeId.IMMORTAL,
    },
    "full tech": set(ROBO_ARMY),
}
"""The tech states a Protoss build walks through on its way to the robo army."""

BUILDABLE_SCENARIOS = {name: ready for name, ready in TECH_SCENARIOS.items() if ready}
"""The tech states where the bot can build something at all.

The degenerate state (no tech) is outside the buildability contract on
purpose: nothing is executable there, so the ballot carries the profile's own
mix as an inert placeholder instead of a buildable composition.
"""


def _robo_profiles() -> list[tuple[str, dict]]:
    """The standard profile's two compositions."""
    return [("profile_0", dict(ROBO_ARMY)), ("profile_1", dict(ROBO_ARMY))]


def _ready_predicate(ready: set[UnitTypeId]):
    """A tech_ready predicate for one tech state."""
    return lambda unit_type: unit_type in ready


@pytest.mark.parametrize("scenario", sorted(BUILDABLE_SCENARIOS))
def test_every_candidate_on_the_ballot_is_fully_buildable(scenario: str) -> None:
    """The invariant: nothing on the ballot needs tech the bot does not have.

    A candidate production cannot execute must never win the competition —
    the 2026-11-09 gate failure crowned a robo mix with no Robotics tech and
    production starved on it.
    """
    ready = BUILDABLE_SCENARIOS[scenario]
    candidates = build_candidates(
        _robo_profiles(),
        {"ROACH": 0.6, "HIGHTEMPLAR": 0.4},
        tech_ready=_ready_predicate(ready),
    )
    assert candidates
    for candidate in candidates:
        unbuildable = set(candidate.composition) - ready
        assert not unbuildable, (
            f"{candidate.name} carries {sorted(unit.name for unit in unbuildable)} "
            f"without the tech to build them"
        )


@pytest.mark.parametrize("scenario", sorted(TECH_SCENARIOS))
def test_the_ballot_is_never_empty(scenario: str) -> None:
    """The counter walk always lands on something — an empty ballot has no winner."""
    candidates = build_candidates(
        _robo_profiles(),
        {"ROACH": 1.0},
        tech_ready=_ready_predicate(TECH_SCENARIOS[scenario]),
    )
    assert candidates, "the ballot must never be empty"


def test_the_degenerate_no_tech_ballot_still_lands_on_something() -> None:
    """With nothing buildable the ballot carries the profile's own mix.

    Every composition is equally unexecutable in that state and an empty
    composition would break ARES' proportion assert, so the profile's mix
    stands in — production stays idle until the tech arrives.
    """
    candidates = build_candidates(
        _robo_profiles(), {"ROACH": 1.0}, tech_ready=lambda unit_type: False
    )
    assert candidates
    assert candidates[0].source == "fallback"
    assert set(candidates[0].composition) == set(ROBO_ARMY)
    proportions = [
        float(info["proportion"]) for info in candidates[0].composition.values()
    ]
    assert isclose(sum(proportions), 1.0, abs_tol=1e-3)


@pytest.mark.parametrize("scenario", sorted(BUILDABLE_SCENARIOS))
def test_every_candidate_keeps_the_ares_proportion_contract(scenario: str) -> None:
    """A filtered composition is re-normalized: ARES asserts the sum is 1.0.

    The proportions of a filtered composition no longer add up to 1.0, and a
    composition that does not add up to 1.0 crashes the SpawnController
    mid-game — so the re-normalization is mandatory, not cosmetic.
    """
    candidates = build_candidates(
        _robo_profiles(),
        {"ROACH": 0.6, "HIGHTEMPLAR": 0.4},
        hold_composition={UnitTypeId.ZEALOT: {"proportion": 1.0, "priority": 0}},
        tech_ready=_ready_predicate(BUILDABLE_SCENARIOS[scenario]),
    )
    assert candidates
    for candidate in candidates:
        proportions = [
            float(info["proportion"]) for info in candidate.composition.values()
        ]
        assert isclose(
            sum(proportions), 1.0, abs_tol=1e-3
        ), f"{candidate.name} sums to {sum(proportions)}"
        assert all(value >= 0.0 for value in proportions)
        for info in candidate.composition.values():
            assert 0 <= int(info["priority"]) < 11


def test_the_robo_mix_never_reaches_the_ballot_without_robotics() -> None:
    """The gate-failure pin: no Robotics tech, no Robo units on the ballot."""
    candidates = build_candidates(
        _robo_profiles(),
        {"ROACH": 0.6, "HIGHTEMPLAR": 0.4},
        tech_ready=lambda unit_type: unit_type == UnitTypeId.ZEALOT,
    )
    for candidate in candidates:
        assert not {
            UnitTypeId.COLOSSUS,
            UnitTypeId.HIGHTEMPLAR,
            UnitTypeId.DISRUPTOR,
            UnitTypeId.IMMORTAL,
        } & set(candidate.composition)


def test_the_hold_candidate_survives_filtering() -> None:
    """The defensive option must not die when its units become unbuildable."""
    profiles = [("profile_0", _composition((UnitTypeId.ZEALOT, 1.0)))]
    hold = {UnitTypeId.COLOSSUS: {"proportion": 1.0, "priority": 0}}
    candidates = build_candidates(
        profiles,
        {"ROACH": 1.0},
        hold_composition=hold,
        tech_ready=lambda unit_type: unit_type == UnitTypeId.ZEALOT,
    )
    holds = [candidate for candidate in candidates if candidate.source == "hold"]
    assert holds, "the hold candidate must survive filtering"
    assert holds[0].composition, "the hold candidate must never be empty"
    assert set(holds[0].composition) == {UnitTypeId.ZEALOT}


def test_the_hold_candidate_keeps_the_incumbents_buildable_units() -> None:
    """The hold stays inside the build's vocabulary, proportions intact."""
    profiles = [
        ("profile_0", _composition((UnitTypeId.ZEALOT, 0.5), (UnitTypeId.STALKER, 0.5)))
    ]
    hold = {
        UnitTypeId.ZEALOT: {"proportion": 0.5, "priority": 0},
        UnitTypeId.STALKER: {"proportion": 0.5, "priority": 1},
    }
    candidates = build_candidates(
        profiles,
        {"ROACH": 1.0},
        hold_composition=hold,
        tech_ready=lambda unit_type: True,
    )
    holds = [candidate for candidate in candidates if candidate.source == "hold"]
    assert holds
    assert set(holds[0].composition) == {UnitTypeId.ZEALOT, UnitTypeId.STALKER}


def test_the_counter_variant_lands_on_the_nearest_buildable_stand_in() -> None:
    """An ideal counter we cannot build translates to its nearest stand-in.

    The High Templar is the best counter against a mixed Protoss/Terran army,
    but without a Templar Archive the walk does not give up: it lands on the
    Stalker, the buildable unit type that sits closest to the High Templar on
    ARES' tech tree (the Stalker needs Gateway + Cyber Core, the High
    Templar two buildings more), and shifts the mix toward it.
    """
    profiles = [
        (
            "profile_0",
            _composition(
                (UnitTypeId.HIGHTEMPLAR, 0.3),
                (UnitTypeId.STALKER, 0.35),
                (UnitTypeId.ZEALOT, 0.35),
            ),
        ),
    ]
    shares = {"MUTALISK": 0.5, "MARINE": 0.5}
    candidates = build_candidates(
        profiles,
        shares,
        tech_ready=lambda unit_type: unit_type
        in {UnitTypeId.ZEALOT, UnitTypeId.STALKER},
    )
    variants = [candidate for candidate in candidates if candidate.source == "counter"]
    assert variants, "the unbuildable ideal must still land on a stand-in"
    variant = variants[0].composition
    assert UnitTypeId.STALKER in variant
    assert UnitTypeId.HIGHTEMPLAR not in variant

    # The variant is built from the buildable part of the profile — the High
    # Templar is gone and the rest re-balanced — so the shift is measured
    # against that base, not against the raw profile.
    buildable_base = _buildable_composition(
        profiles[0][1], {UnitTypeId.ZEALOT, UnitTypeId.STALKER}
    )
    assert (
        variant[UnitTypeId.STALKER]["proportion"]
        > buildable_base[UnitTypeId.STALKER]["proportion"]
    )
    assert (
        variant[UnitTypeId.ZEALOT]["proportion"]
        < buildable_base[UnitTypeId.ZEALOT]["proportion"]
    )


def test_the_stand_in_is_the_nearest_unit_on_the_tech_tree() -> None:
    """The tech tree decides who stands in — the counter value only breaks ties.

    A Colossus we cannot build stands in the Immortal (one building away on
    ARES' tech tree), not the Zealot — even when the Zealot would counter the
    enemy better.
    """
    translation = _translation_map(
        [UnitTypeId.COLOSSUS, UnitTypeId.IMMORTAL, UnitTypeId.ZEALOT],
        {UnitTypeId.COLOSSUS, UnitTypeId.IMMORTAL, UnitTypeId.ZEALOT},
        {UnitTypeId.IMMORTAL, UnitTypeId.ZEALOT},
        {UnitTypeId.IMMORTAL: 0.0, UnitTypeId.ZEALOT: 10.0},
    )
    assert translation[UnitTypeId.COLOSSUS] == UnitTypeId.IMMORTAL
    assert UnitTypeId.IMMORTAL not in translation, "a buildable unit needs no stand-in"


def test_the_translation_is_data_driven_from_the_ares_tech_tree() -> None:
    """The stand-in distance is measured on ARES' own tech requirements.

    The Immortal needs Gateway + Cyber Core + Robotics Facility; the Stalker
    needs Gateway + Cyber Core. The Immortal is the nearer neighbour of a
    Colossus (one building away), so it is the one that stands in.
    """
    assert _tech_distance(UnitTypeId.COLOSSUS, UnitTypeId.IMMORTAL) == 1
    assert _tech_distance(UnitTypeId.COLOSSUS, UnitTypeId.STALKER) == 2
    assert _tech_distance(UnitTypeId.COLOSSUS, UnitTypeId.ZEALOT) == 3
    # A Warp Gate is a Gateway — the tech tree counts them as one node.
    assert _tech_distance(UnitTypeId.STALKER, UnitTypeId.ZEALOT) == 1


def test_the_nearest_stand_in_wins_over_the_better_counter() -> None:
    """Proximity decides the stand-in; the counter value only breaks ties.

    The Stalker counters the enemy better than the Immortal here, but the
    Immortal is the neighbour of the ideal on the tech tree — the stand-in
    has to sit on the path we are investing in, not on the better matchup.
    """
    translation = _translation_map(
        [UnitTypeId.COLOSSUS],
        {
            UnitTypeId.COLOSSUS,
            UnitTypeId.IMMORTAL,
            UnitTypeId.STALKER,
            UnitTypeId.ZEALOT,
        },
        {UnitTypeId.IMMORTAL, UnitTypeId.STALKER, UnitTypeId.ZEALOT},
        {UnitTypeId.STALKER: 10.0, UnitTypeId.IMMORTAL: 0.0, UnitTypeId.ZEALOT: 0.0},
    )
    assert translation[UnitTypeId.COLOSSUS] == UnitTypeId.IMMORTAL


def test_only_the_builds_own_vocabulary_can_stand_in() -> None:
    """A stand-in the build does not field would be a strategy change."""
    translation = _translation_map(
        [UnitTypeId.COLOSSUS],
        {UnitTypeId.COLOSSUS, UnitTypeId.ZEALOT},
        {UnitTypeId.ZEALOT, UnitTypeId.IMMORTAL},
        {},
    )
    assert translation[UnitTypeId.COLOSSUS] == UnitTypeId.ZEALOT


# ===== RANKING =====


class _FixedConsideration:
    """A consideration with a per-candidate base — isolates the runner from the terms."""

    def __init__(self, bases: dict[str, float]) -> None:
        self._bases = bases

    def load(self, params) -> None:
        """No tunable parameters — the bases are fixed."""

    def evaluate(self, context) -> Appraisal:
        candidate = getattr(context, "candidate", None)
        name = candidate.name if candidate is not None else ""
        return Appraisal(base_score=self._bases.get(name, 0.0))


def _scorer_with_fixed_bases(bases: dict[str, float], **commitment_params) -> CompositionScorer:
    """A scorer whose terms are fixed numbers — the runner is under test.

    The belief-confidence term rides along so the ranking keeps the canonical
    shape: (Σ bases) × confidence.
    """
    considerations = [_FixedConsideration(bases), BeliefConfidence()]
    commitment = Commitment()
    commitment.load(commitment_params)
    return CompositionScorer(considerations=considerations, commitment=commitment)


def test_ties_keep_the_candidate_order() -> None:
    """Equal scores → the first candidate wins (stable sort), so ties are stable."""
    scorer = _scorer_with_fixed_bases({"first": 1.0, "second": 1.0})
    candidates = [
        _candidate("first", (UnitTypeId.ZEALOT, 1.0)),
        _candidate("second", (UnitTypeId.ZEALOT, 1.0)),
    ]
    ranking = scorer.rank(candidates, _context())
    assert ranking[0].choice.name == "first"


def test_the_incumbent_keeps_the_crown_inside_the_margin() -> None:
    """Hysteresis: a challenger has to beat the margin, not just tie the score."""
    scorer = _scorer_with_fixed_bases(
        {"incumbent": 1.0, "challenger": 1.2},
        weight=0.25,
        margin=0.25,
        curve=_config_curve("commitment"),
    )
    candidates = [
        _candidate("incumbent", (UnitTypeId.ZEALOT, 1.0)),
        _candidate("challenger", (UnitTypeId.ZEALOT, 1.0)),
    ]
    ranking = scorer.rank(candidates, _context(), incumbent="incumbent")
    assert ranking[0].choice.name == "incumbent"


def test_a_challenger_takes_the_crown_past_the_margin() -> None:
    """Past the margin the commitment fades and the challenger wins."""
    scorer = _scorer_with_fixed_bases(
        {"incumbent": 1.0, "challenger": 1.5},
        weight=0.25,
        margin=0.25,
        curve=_config_curve("commitment"),
    )
    candidates = [
        _candidate("incumbent", (UnitTypeId.ZEALOT, 1.0)),
        _candidate("challenger", (UnitTypeId.ZEALOT, 1.0)),
    ]
    ranking = scorer.rank(candidates, _context(), incumbent="incumbent")
    assert ranking[0].choice.name == "challenger"


def test_the_commitment_is_a_bonus_not_a_sticky_flag() -> None:
    """The incumbent gets no protection when there is no challenger."""
    scorer = _scorer_with_fixed_bases(
        {"incumbent": 1.0},
        weight=0.25,
        margin=0.25,
        curve=_config_curve("commitment"),
    )
    candidates = [_candidate("incumbent", (UnitTypeId.ZEALOT, 1.0))]
    ranking = scorer.rank(candidates, _context(), incumbent="incumbent")
    assert ranking[0].score == pytest.approx(1.25)


def test_the_incumbent_bonus_fades_with_the_challengers_gap() -> None:
    """The bonus is scored, not a latch — it shrinks as the gap grows."""
    commitment = Commitment()
    commitment.load(
        {"weight": 0.25, "margin": 0.25, "curve": _config_curve("commitment")}
    )
    assert commitment.bonus_for_gap(0.0) == pytest.approx(0.25)
    assert commitment.bonus_for_gap(0.25) == pytest.approx(0.25)
    assert commitment.bonus_for_gap(0.5) == pytest.approx(0.0)


def test_ranking_is_deterministic() -> None:
    """Same candidates, same context → same winner, every time."""
    scorer = _scorer_with_fixed_bases({"a": 0.4, "b": 0.9, "c": 0.4})
    candidates = [
        _candidate("a", (UnitTypeId.ZEALOT, 1.0)),
        _candidate("b", (UnitTypeId.ZEALOT, 1.0)),
        _candidate("c", (UnitTypeId.ZEALOT, 1.0)),
    ]
    context = _context()
    first = [(entry.choice.name, entry.score) for entry in scorer.rank(candidates, context)]
    second = [(entry.choice.name, entry.score) for entry in scorer.rank(candidates, context)]
    assert first == second
    assert first[0][0] == "b"


def test_the_confidence_multiplier_scales_the_ranking() -> None:
    """A half-trusted picture halves the reported utility of every candidate."""
    scorer = _scorer_with_fixed_bases({"a": 1.0})
    candidates = [_candidate("a", (UnitTypeId.ZEALOT, 1.0))]
    full = scorer.rank(candidates, _context(confidence=1.0))[0].score
    half = scorer.rank(candidates, _context(confidence=0.5))[0].score
    assert half == pytest.approx(full * 0.5)


def test_the_scored_terms_use_the_reasoner_runner() -> None:
    """The composition score is the reasoner's canonical combination."""
    scorer = CompositionScorer(config=CONFIG)
    candidates = [
        _candidate("immortals", (UnitTypeId.IMMORTAL, 1.0)),
        _candidate("zealots", (UnitTypeId.ZEALOT, 1.0)),
    ]
    context = _context(enemy_composition={"ROACH": 1.0})
    ranking = scorer.rank(candidates, context)

    winner = next(
        candidate
        for candidate in candidates
        if candidate.name == ranking[0].choice.name
    )
    expected = combine(
        tuple(
            consideration.evaluate(
                _context(candidate=winner, enemy_composition={"ROACH": 1.0})
            )
            for consideration in scorer.considerations
        )
    )
    assert ranking[0].score == pytest.approx(expected.base_score * expected.multiplier)


# ===== SWITCH REMOVAL =====


def test_the_sticky_switches_are_gone() -> None:
    """The one-way economy switch and the archon threshold switch are removed."""
    macro_source = (REPO_ROOT / "bot" / "managers" / "macro.py").read_text(
        encoding="utf-8"
    )
    constants_source = (REPO_ROOT / "bot" / "constants.py").read_text(encoding="utf-8")
    for symbol in REMOVED_SWITCHES:
        assert symbol not in macro_source, f"{symbol} still lives in macro.py"
        assert symbol not in constants_source, f"{symbol} still lives in constants.py"


def test_the_bot_state_has_no_economy_switch_flag() -> None:
    """No one-way flag is left on the bot either."""
    bot_source = (REPO_ROOT / "bot" / "bot.py").read_text(encoding="utf-8")
    assert "_economy_switch_triggered" not in bot_source


# ===== FALLBACK ALARM =====


@pytest.fixture
def telem(monkeypatch) -> None:
    """Put the telemetry channel in a live state that prints to stdout.

    The alarm rides the bot's own TELEM channel, so the test reads that
    channel instead of the emitter's internals — no game, no file I/O.
    """
    from bot.utilities import telemetry

    monkeypatch.setattr(telemetry, "_initialized", True)
    monkeypatch.setattr(
        telemetry,
        "_event_context",
        {"env": "ladder", "match_id": "unit-test", "schema_version": 1},
    )


def _telem_events(capsys) -> list[dict]:
    """The TELEM records printed on the channel since the last read."""
    printed = capsys.readouterr().out
    return [
        json.loads(line[len("TELEM ") :])
        for line in printed.splitlines()
        if line.startswith("TELEM ")
    ]


def test_the_fallback_alarm_fires_when_the_conservative_stand_in_takes_over(
    telem, capsys
) -> None:
    """The seed judgment standing in for a real measure is an alarm.

    The mix has nothing we can build above the production floor, so the
    conservative stand-in pins the build to the one unit type we can produce.
    """
    profiles = [
        (
            "profile_0",
            {
                UnitTypeId.STALKER: {"proportion": 0.0, "priority": 0},
                UnitTypeId.IMMORTAL: {"proportion": 1.0, "priority": 1},
            },
        )
    ]
    build_candidates(
        profiles,
        {"ROACH": 1.0},
        tech_ready=lambda unit_type: unit_type == UnitTypeId.STALKER,
        game_time=42.0,
    )

    events = _telem_events(capsys)
    assert len(events) == 1
    event = events[0]
    assert event["subsystem"] == "macro"
    assert event["action"] == "fallback_engaged"
    assert event["reason"] == "conservative_buildable"
    assert event["path"] == "base"
    assert event["source"] == "fallback"
    assert event["unit"] == "STALKER"
    assert event["ts"] == 42.0


def test_the_fallback_alarm_names_the_hold_path(telem, capsys) -> None:
    """The defensive option's own fallback is alarmed under its own path.

    The incumbent's composition cannot be built, so the conservative stand-in
    fills the hold candidate — the hold path's own last resort.
    """
    profiles = [
        (
            "profile_0",
            {
                UnitTypeId.STALKER: {"proportion": 0.0, "priority": 0},
                UnitTypeId.IMMORTAL: {"proportion": 1.0, "priority": 1},
            },
        )
    ]
    hold = {
        UnitTypeId.STALKER: {"proportion": 0.0, "priority": 0},
        UnitTypeId.IMMORTAL: {"proportion": 1.0, "priority": 1},
    }
    build_candidates(
        profiles,
        {"ROACH": 1.0},
        hold_composition=hold,
        tech_ready=lambda unit_type: unit_type == UnitTypeId.STALKER,
        game_time=51.5,
    )

    events = _telem_events(capsys)
    assert len(events) == 1
    assert events[0]["action"] == "fallback_engaged"
    assert events[0]["reason"] == "conservative_buildable"
    assert events[0]["path"] == "hold"
    assert events[0]["source"] == "hold"
    assert events[0]["unit"] == "STALKER"
    assert events[0]["ts"] == 51.5


def test_the_fallback_alarm_fires_when_nothing_is_buildable(telem, capsys) -> None:
    """The degenerate state — nothing we can produce at all — is alarmed too.

    The profile's own mix stands in as an inert placeholder there, which is
    the last resort of the last resort — it must not be a silent one.
    """
    build_candidates(
        _robo_profiles(),
        {"ROACH": 1.0},
        tech_ready=lambda unit_type: False,
        game_time=12.0,
    )

    events = _telem_events(capsys)
    assert len(events) == 1
    assert events[0]["action"] == "fallback_engaged"
    assert events[0]["reason"] == "nothing_buildable"
    assert events[0]["path"] == "base"
    assert events[0]["source"] == "fallback"
    assert events[0]["unit"] is None
    assert events[0]["ts"] == 12.0


def test_the_fallback_alarm_stays_quiet_when_the_primary_path_stands(
    telem, capsys
) -> None:
    """A buildable mix is the primary path — no fallback, no alarm."""
    profiles = [("profile_0", _composition((UnitTypeId.ZEALOT, 1.0)))]
    build_candidates(
        profiles,
        {"ROACH": 1.0},
        tech_ready=lambda unit_type: unit_type == UnitTypeId.ZEALOT,
        game_time=30.0,
    )

    assert _telem_events(capsys) == []
