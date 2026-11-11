---
project: PiG_Bot
version: 1.0
author: Praxis
goal: "Lay the bot-wide utility-AI foundation (reasoner library + intel context), then convert Macro, Reaction, and Scouting as first adopters. Phases 2-3 extend to combat, structures, expansion, micro."
status: draft-for-review
priority: high

---

# Utility AI Refactor Plan — PiG_Bot

<!--
Purpose: Convert PiG_Bot's threshold-based decisions to scored utility-AI across managers and combat targeting, coordinated through one intel picture.
Key Decisions: Canonical score = (Σ base scores) × (Π multipliers), any veto ⇒ 0; curve steepness is the primary anti-oscillation tool with scored hysteresis as fallback — never sticky switches; beliefs enter as multipliers, not additive terms; magnitudes hand-seeded (Liquipedia provenance) and refit from telemetry in Task 8.
Limitations: Phase 1 covers the three managers plus squad attack-targeting (Task 10); engagement/retreat, chrono, expansion, defender allocation, mass recall and per-unit micro remain Adoption Roadmap rows; fitted weights await Task 8's corpus fit.
-->

## Goal

Lay the utility-AI foundation for the whole bot — one shared reasoner over one intel picture — then convert the three managers as its first adopters. Every later decision system (combat, structures, expansion) joins the same currency instead of growing new thresholds.

**Phase 1 (this plan):** scoring-table elevation → reasoner library + intel context → Macro, Reaction, and Scouting conversions → coordination wiring → fitted magnitudes.
**Phase 2-3 (Adoption Roadmap at the end):** combat engagement, chrono/recharge, expansion, defender allocation, mass recall, macro leftovers, per-unit micro.
**Out of scope for Phase 1:** opening build orders from the build runner, combat micro execution — runtime unit targeting/maneuver behavior (mechanical import/comment updates in micro files are in scope), the belief models themselves (they become inputs).

## Design language

Every decision cycle, all candidate actions get a number; the highest wins. No if-then gates, no sticky switches.

**Design authority beyond this plan:** the owner's vault architecture notes are upstream of this plan and take precedence where they speak. The counter-table lineage in particular comes from `400 Notes & Ideas/Counter Tables, Demand Systems, and Reinforcement Learning — Architecture Session.md` (MindMe PostBots session 2026-08-29, Addendum 2026-09-12). Rule for every task brief: sweep the owner's 400 Notes architecture docs for the task's domain BEFORE writing the brief.

**Fallback rule (owner directive 2026-11-10):** a fallback/last-resort path is permitted only as a last line of defense for a structural invariant (e.g., "the ballot is never empty"), and it must satisfy ALL of: (1) it never decides — candidates it produces still compete on score, no fiat wins; (2) it is OBSERVABLE — engaging it emits a telemetry event, so the primary path's failure is an alarm, not silent work the team relies on without knowing; (3) it is on the ledger — every fallback carries a comment and a plan entry naming the mechanism that will retire it (see Task 8 audit, Task 9 retirement). "If I didn't consider something this should catch it" is not a design; a fallback that is doing real work means the design above it is wrong. No duct tape as a deliverable.

- **Reasoner** = a manager's decision step. **Choices** = its candidates (compositions, responses, scout missions).
- **Consideration** = one scored aspect of the situation (economy, threat, cost, staleness). Small, parameterized, reusable across managers. Two functions: `load(params)` and `evaluate(context) → appraisal`.
- **Context** = the intel snapshot handed to considerations. Perception belongs to the intel layer; considerations never dig into raw game state.
- **Appraisal** = (base score, multiplier, veto).

Canonical combination, used by every task:

```
score(choice) = [ Σ base_scores ] × [ Π multipliers ]   (any veto fires ⇒ score 0)
```

- One currency: normalize raw inputs so mining value, threat response, and tech value stay comparable.
- Every raw input passes a response curve first: normalize to [0,1], then shape (linear / exponential / logistic). No raw values enter scores.
- Beliefs enter as multipliers, not additive terms — scores are expected utilities (value × confidence). Stale intel discounts a score instead of zeroing it.
- Marginal utility decides "how many": the 1st worker pulled costs little, the 6th costs a lot. Pull counts and stop points are curve crossings, not constants.
- Combine at most 2-3 considerations per box, then chain boxes (compartmentalized confidence). Reuse boxes across managers: threat severity feeds reaction scoring AND scouting risk.
- Hard impossibilities veto (no production facility for a candidate composition, no worker available to scout). Costs are negative bases, never vetoes.
- Exponential / logistic curves are the default for threat-like inputs: distant noise ignored, aggressive commitment near critical. **Curve steepness is the primary anti-oscillation tool.** Scored hysteresis is the fallback if thrash survives the curves — never sticky switches.
- Curve parameters live in config as personality hypotheses, seeded from the Liquipedia-annotated table and fitted from telemetry in Task 8 (2,300+ game corpus).

Update cadence:

- Re-score on belief change or every N game loops (N in config). Cache appraisals until an input changes. Never score every frame.

Selection rule: deterministic argmax for this refactor (verification needs determinism). Weighted-random top-N is a future personality knob — do not add it now.

## Global constraints (apply to every task)

Style, dependency discipline, and scope rules live in AGENTS.md — not repeated here. These are the refactor-specific constraints:

1. Do not change behavior of: opening build orders, combat micro execution, belief models — they are inputs, not targets. The scoring table module is in scope (Tasks 1-2); its rename may update import lines, comments, and structure listings in combat files — the fence is behavior, not file avoidance.
2. Every scoring change requires pure-Python unit tests pinning the scoring math BEFORE wiring it into the manager (test-first).
3. Every behavior change ships with its telemetry gate result; no regression vs master without a documented reason.
4. Preserve existing telemetry event names and fields. Add new ones freely; remove or rename nothing.
5. One task, one PR: `feature/task-N` → PR → `development`. PR description must state which acceptance criteria are met and which remain. Acceptance gates pass before merge, each on its named channel. `development` = integration line; agents never push to or merge into `master` — promotion is the owner's decision.
6. Every score follows the Design language: intel context only (no raw game-state access), every input through a response curve, re-score on cadence — never per frame.

## Task 1 — Rename + elevate scoring table

**Goal:** `target_scoring` module renamed to `scoring_table`; the `COUNTER_TABLE` dict becomes `SCORING_TABLE`. Its role documented as scoring core, not combat nudge.

**Scope:** module rename, import updates (6 sites — including the function-local import in `utilities/debug.py` that only executes in debug sessions, so a miss survives smoke runs), docstring stating the table's elevated role. `render_target_scoring_debug` keeps its name (it names the overlay). Comments, docstrings, and README/AGENTS structure listings referencing the old path update too.
**Preserve:** behavior — zero functional change.
**Acceptance:**
- [x] old module path gone — `grep -rn "target_scoring" bot/` returns zero hits except the kept overlay function `render_target_scoring_debug` — and `scoring_table` imports resolve (offline) *(2026-11-05: 3 hits, all `render_target_scoring_debug`; import verified — SCORING_TABLE 95 rows, SITUATIONAL_ROWS 6)*
- [x] bot imports cleanly (`python -c "import bot"`); pytest configured — `testpaths` pinned in pyproject so the opponent bots are never collected — and collection runs clean (the suite itself starts with Task 2; dev-group dep per the repo's Poetry workflow) (offline) *(2026-11-05: `poetry run pytest` → exit 5, collection clean, no errors. Import caveat: `import bot` needs run.py's ares-submodule paths (`ares-sc2/src/ares`, `ares-sc2/src`, `ares-sc2`) because poetry installs the path dep as `site-packages/src/ares` — the bare form needs those paths; pytest's `pythonpath` supplies them inside the suite)*
- [x] one local game with debug overlay enabled — exercises the function-local `debug.py` import that only executes in debug sessions (local game) *(2026-11-05: MagannathaAIE_v2 vs Zerg VeryHard/Rush, BotDebug=True — rush detected, cheese response, decisive win, zero tracebacks; the overlay call path is not exception-wrapped, so a failed import would have killed the step loop)*

## Task 2 — Extend SCORING_TABLE to full Protoss arsenal

**Current:** 135 pairs, Liquipedia-annotated. The 2026-11-06 coverage pass filled the mirror-target gaps: every Protoss combat unit we build now has rows on both sides of the table — as an attacker (what we field) and as a target (what the enemy fields, i.e. PvP). Dark Templar, Disruptor, High Templar, Archon and Sentry had zero target rows before; Immortal and Colossus had two each. Mothership stays excluded by design (support ult, no meaningful counter rows — see the research doc), Observer exempt (no weapon).

**Goal:** fill coverage gaps so any composition the bot builds or faces scores correctly. The scoring function can only compare compositions the table knows — blocking for the macro conversion (Task 4).

**Gaps to cover:**
- Missing enemy targets: Dark Templar, Disruptor (mirror matchups) — plus High Templar, Archon, Sentry (same gap, same pass) and the thin Immortal/Colossus target coverage
- Audit remaining attacker/target rows for completeness vs the Liquipedia research plan

**Rules:** new pairs are Liquipedia-annotated, same format as existing entries.
**Acceptance:**
- [x] every unit the bot can build has complete target rows (Observer exempt — no weapon; Mothership excluded by design per the research doc) (offline) *(2026-11-06: 40 rows added, 95 → 135. Every unit in the bot's army compositions (macro.py) has attacker AND target rows; every attacker in the table has target rows. The rule is pinned by tests/unit/test_scoring_table.py — composition roster read from macro.py with `ast`, no import. No existing pair value changed (diffed against HEAD): 0 modified, 0 removed)*
- [x] table passes a persisted format-validation test in `tests/unit/` (pytest): no duplicate pairs, all values on one scale, and every key resolves against the `UnitTypeId` enum — a typo'd key silently scores 0.0 via `.get(..., 0.0)`. The pytest runner, Poetry dev-group dep, and `[tool.pytest.ini_options]` block are configured in Task 1; this PR adds the suite's first test. `poetry run pytest` passing is a merge condition from here on (unit) *(2026-11-06: `poetry run pytest` → 10 passed. The duplicate check parses the source with `ast` — a dict literal silently keeps the last value for a repeated key, so the runtime table cannot show duplicates. Mutation-checked: duplicate pair, string key, 120.0 value, target-side hole and composition-side hole each fail the suite)*
- [x] no existing pair values modified
- [ ] full roster run matches master within the noise band — table additions legitimately change targeting; this is the roster smoke moved here from Task 1 (lab) *(2026-11-06: NOT run — lab-channel gate, no runner access in the task environment. Owner to run on the task branch before merge: master vs feature/task-2-scoring-table-gaps, ±1 game per 3-game series)*

## Task 3 — Reasoner library + intel context

**Goal:** the shared foundation every decision system scores through — one library, not three bespoke scorers. Phase 2 adoption (combat, structures, expansion) reuses this module unchanged.

**Deliverables:**
- One shared module under `bot/utilities/`: response curves (linear / exponential / logistic), the appraisal type (base score, multiplier, veto), and the canonical combination runner — `score(choices) → ranked candidates`, deterministic argmax.
- `IntelContext`: the snapshot dataclass considerations score against — belief state, threat severity, intel urgency/freshness, economy, tech progress. Built by the intel layer on the re-score cadence, never per frame. Reads from the belief layer and intel quality systems; owns no state.
- Curve parameters load from config.

**Acceptance:**
- [x] curve shapes + canonical combination pinned by unit tests (sum of bases, product of multipliers, any veto ⇒ 0) (unit) *(2026-11-07: 39 new checks in `tests/unit/test_reasoner.py`; `poetry run pytest` → 49 passed (10 pre-existing + 39 new). Mutation-checked: sum×product → sum+product, veto not zeroing, tie-break order flipped, sum → max of bases, product → sum of multipliers, flipped exponential direction, logistic ignoring the midpoint, unclamped normalization, empty choice scoring 1.0, and a manager import in the library each fail the suite)*
- [x] library imports nothing from manager code — dependency direction is managers → library (offline) *(2026-11-07: the module imports stdlib only — an `ast` parse of the source pins it, and a subprocess loads the module by path and asserts no `bot`/`sc2`/`ares` module entered `sys.modules`. Note: importing it as `bot.utilities.reasoner` runs `bot/__init__.py`, which pulls in the game stack — the library itself has no such dependency)*
- [ ] Tasks 4-6 consume it; no manager defines its own curve or scorer (offline) *(forward-looking — the adopters land in their own PRs; today the library is the only curve/scorer in the tree)*
- [x] `IntelContext` is the test seam: scenario and unit tests construct it directly, no game required — this is what makes Tier A gates legitimate for every later adopter (unit) *(2026-11-07: `IntelContext` is a frozen dataclass of plain scalars and mappings — the tests construct it directly, no bot instance and no SC2 binary involved)*

## Task 4 — Macro manager utility scoring

**Current:** `select_army_composition` — profile base → sticky econ switch → archon % threshold → nudges (scoring-table, strategy, resource-pressure, priority reorder), clamped + re-normalized.

**Goal:** composition candidates compete by score every macro cycle.

**Design (canonical form — see Design language):**
```
score(comp) = [ Σ counter_effectiveness(my_units vs believed_enemy_comp)
              + economy_term(worker count, income, saturation)
              + tech_term(upgrade/tech progress)
              − cost_term(minerals/gas/time) ] × belief_confidence
```
- The scoring table becomes the core base score (no longer a post-decision nudge).
- `belief_confidence` is a multiplier: expected utility, not an additive term.
- Candidates: current profile compositions + counter-driven variants + hold/defend candidate, capped at 8 per cycle. The winner builds.
- Remove sticky one-way switches and the threshold archon switch. Stabilizers: curve steepness first, commitment consideration second — never hard switches.
- Magnitudes: seed from Liquipedia baselines, fit later from telemetry (Task 8); document seed provenance in comments.
- **Buildability invariant (MindMe, added 2026-11-10 after the gate fail):** every candidate on the ballot must be fully buildable with current tech and production — MindMe's rule: the counter walk "stops at the first row where it can build everything. Always lands on something." Ideal counters that are not yet buildable are translated toward the nearest reachable approximation along the ARES tech tree (extraction/query job — ARES already holds prerequisites, costs, build times; no hand-authored tech map), never silently dropped alone.

**Acceptance:**
- [ ] **buildability contract test (added 2026-11-10):** every candidate on the ballot is fully buildable, and the ballot is never empty — a permanent regression pin after the 2026-11-09 gate failure, where 74% of the winning candidate's counter score came from unbuildable units (COLOSSUS/HT/DISRUPTOR with no Robotics tech)
- [x] unit tests pin curve shapes and candidate scoring before integration (unit) *(2026-11-09: `tests/unit/test_composition_scoring.py` — 33 checks written before the module landed; `poetry run pytest` → 82 passed (49 pre-existing + 33 new). Pinned: the counter sums read straight out of SCORING_TABLE (share-weighted, situational rows excluded), the signed cost base, belief confidence as a multiplier and never a term, the canonical `score = (Σ bases) × confidence`, the candidate cap, the hold candidate's survival, the deterministic tie order, the counter variant's shift, the tech-ready filter, ARES composition validity (Σ = 1.0, priority ∈ [0,11)), and the commitment hysteresis (inside the margin, past the margin, bonus fade))*
- [x] econ switch removed, no one-way state remains (offline) *(2026-11-09: `grep -rn "_economy_switch_triggered\|economy_switch_threshold\|archon_switch_threshold" bot/ tests/ config.yml` → 0 hits outside the removal-proof test itself; `bot._economy_switch_triggered` deleted from bot.py; `economy_switch_threshold`, `archon_switch_threshold` retired from `BuildProfile`; the archon gas gate survives under its own name (`archon_merge_gas_requirement`) because it gates HT merging, not a switch. The old one-way economy switch is replaced by a scored commitment bonus — the composition now switches back when the challenger's advantage grows, which the sticky flag could never do)*
- [ ] branch passes full roster regression vs master within the noise band, compared in the lab's Compare tab (lab) *(2026-11-09: NOT run — lab-channel gate, no runner access in the task environment. Owner to run master vs feature/task-4-macro-scoring before merge, ±1 game per 3-game series)*

## Task 5 — Reaction manager utility scoring

**Current:** category routing (CHEESE/ALL_IN/TIMING/MACRO) behind confidence thresholds; handlers fire or don't. Firing rate on rush tapes: 0/6 (Log).

**Goal:** replace category routing + confidence thresholds with scored response vs continue.

**Design:**
```
u(respond) = response_curve(p_cheese) × threat_severity(army_ratio, distance, time_to_contact)
u(continue) = plan_value(economy, tech progress)
```
- Response fires when `u(respond) > u(continue)` — no threshold to mis-set.
- Worker pulls: marginal cost curve vs defensive value; pull count = curve crossing point. Pull enough, not everything.
- Logistic response curve as default, personality-tunable parameter exposed.
- Handlers (`defend_worker_rush`, `defend_cannon_rush`, ...) become scored candidates, not privileged routines.
- Commitment: `u(continue)` may carry a hysteresis bonus so near-ties don't thrash between respond and continue. Scored, never sticky.

**Acceptance:**
- [ ] the `reactions` activation event fires on the decision-layer tape — the PR gate (scenario/lab). Judged on the transition event (next bullet), never on `used_cheese_response`: that match field is a game-end snapshot of live state (`is_cheese_response` = active AND not transitioned, post-1640601), so a fired-then-cleared response records False — a gate on it would false-fail a working response. Villain: the constructed cannon-rush setup (lab DB `test_group_id=8`, matches 985–988, LeyLinesAIE_v3) — fully recorded and reproducible; Chance/ProxyCannoneer is optional hardening, not a gate. Evidence basis: baseline 0/3 = banked records + code-path elimination; the re-run supplies the measured version (per-game telemetry not retained — lab-DB-confirmed; the PR must state the evidence basis and the code version the banked tapes ran — see the Verification caveat). Worker-rush firing (perception-layer: model classified MACRO 0.795; fix is model-side retraining, outside this plan) is reported but not gated. Survival-rate deltas reported for both tapes.
- [ ] `reactions` transition event emitted on every response activation and clear (category, level2 label, scores, game_time) — turns the binary end-state match field into a per-activation trace; timing/all-in responses become measurable, and "fired then cleared" is distinguishable from "never fired" (unanswerable for the banked baseline for the same retention reason). `game_time` in the payload makes response latency answerable — firing too late is a failure win rate cannot catch. Rationale (git-history): this restores the "ever fired" guarantee of the pre-1640601 latch — `_used_cheese_response` was made one-way in ee01454 (2025-10-27) precisely so game-end reports would retain it; the ReactionManager refactor (1640601, 2026-06-16) replaced the latch with live-state, and the game-report field inherited the name without the semantics. New event, emission-on-change only; handler switches emit as clear-then-activate pairs. Existing names and fields preserved (constraint 4) (scenario/lab).
- [ ] the scoring competition is subtype-agnostic by construction — the same `u(respond) vs u(continue)` decides worker rush, cannon rush, proxy, ling rush; the only subtype-specific data is each handler's own candidate cost/effect inputs. Verified by: one scoring path, N scored candidates (unit).

## Task 6 — Scouting manager utility scoring

**Current:** VOI destination ranking is already scored; launch timing and scout production are rules.

**Goal:** complete utility on launch and production decisions.

**Design:**
```
u(scout) = info_gain(staleness × decision_impact) × P_survive − cost(worker-time)
```
- Risk enters as a survival multiplier (expected value of the info), cost as a negative base — canonical form.
- Scout launches compete against mining/defending in the same score currency.

**Acceptance:**
- [ ] existing VOI destination ranking pinned by an offline equivalence test — same inputs → same ranked outputs as master (`enable_scout_voi: False`; an in-game "unchanged" gate is unobservable) (unit)
- [ ] launch/production decisions routed through the same scoring currency (unit)
- [ ] unit tests pin the risk and info-gain terms (unit)

## Task 7 — Intel coordination wiring

**Goal:** single belief state read by all three managers through the intel layer.

**Design:**
- Managers register as intel consumers.
- Stale intel discounts contribution (decay), never zeroes it.
- Same `p_cheese` input feeds reaction and macro scores.
- A rising p_cheese must raise defense utility, shift composition, and raise scout priority together — all three read the same number.

**Acceptance:**
- [ ] no manager reads a private/duplicate belief source (offline)
- [ ] the four hard-coded `_intel_urgency > 0.5` gates (scout hunt, hallucination scout, expansion block, urgent observer) route through one shared staleness consideration — the urgent-observer gate lives in structure_manager and may follow in Phase 2 (unit)
- [ ] staleness gate tested (decay curve, not binary cutoff) (unit)
- [ ] full regression vs master within the noise band (lab)

## Task 9 — Tech-proximity semi-tree (future phase, from the vault Addendum 2026-09-12)

**Goal:** the full "ideal vs what can I actually do now" machinery — the vault architecture doc §B/§C/§D, sequenced after the manual conversions (its own §E sequencing: response fixes first, vocabulary expansion second, graph/learned layers last).

- Compositions as nodes, viable next compositions as edges weighted by switching cost = forward spend + vulnerability window (sunk cost excluded) — extracted from ARES/game data, no hand-authored tech map.
- Counter selection = counter-value ÷ switching-cost, walked from the current army node. Trunk builds (stalker-robo, sky, splash) pay no switching tax; fringe counters only followed when value justifies cost.
- Three-layer scope split: fight oracle (what trades best) / tech graph (numbers only, no decisions) / policy module (reinforce vs invest, gated by belief confidence).
- **Retire the hand-tuned conservative fallback (owner directive 2026-11-10):** "cheapest buildable = safest" in `_conservative_buildable` is a seed judgment standing in for a real measure. Task 9 replaces it with the earned version: nearest-reachable selection by switching cost from the incumbent node (forward spend + vulnerability window), computed from the tech graph — not a heuristic, and no longer a fallback at all but the primary translation path. Until Task 9 lands, the fallback must remain an OBSERVABLE alarm (see the Design language fallback rule), never a silent workhorse.

**Acceptance:** defined when scheduled. Depends on Tasks 4–7 outcomes.

## Task 8 — Fitted magnitudes from telemetry

**Goal:** reseed score weights using lab outcomes instead of Liquipedia seeds alone.

**Method:** fit per-term weights against the 2,300+ game corpus (e.g., immortal-vs-roach outcomes vs win rate).
- [ ] **fallback-engagement telemetry audit (owner directive 2026-11-10):** before fitting, measure how often the conservative fallback fired in the corpus and in gate series. Frequent engagement means the primary candidate-construction path is misaligned with reachable tech (profile bases containing unbuildable mixes) — fix the construction, not the weights. If the fallback fired at all in a gate series, the PR must say so.
**Acceptance:**
- [ ] fitted weights documented with fit evidence (offline)
- [ ] regression run shows no-worse within the noise band across the roster; improvements logged per matchup; series length stated in the PR (lab)

## Task 10 — Squad attack-targeting utility scoring

**Goal:** army attack-targeting converts from the rule hierarchy to scored utility. This is the combat-targeting conversion — candidate targets compete per squad through the reasoner; the bot-global `current_attack_target` singleton goes away for behavior (a computed `army_focus` remains, reporting only).

**Current:** `attack_target` (combat.py) — 7-step rule chain: sticky-structure keep-if-valid, proximity stickiness, army targeting behind two supply thresholds (6/7), structure fallback, expansion cycling, keep-if-unscouted, final fallback. Stickiness patches hold the oscillation down manually.

**Named regression to fix:** army-level bouncing between candidate targets — the endgame mop-up bounces between the last two enemy buildings until the hour timeout records a tie.

**Design — each squad scores candidate targets, holds its own commitment (canonical form, per the Design language):**

1. **Proximity** — exponential curves (vault principle): linear closest-flip dies at the equidistance boundary; the flat tail keeps near-competitors decisively separated. No endgame mode — the curve solves the mop-up.
2. **Strategic value** — production > tech > supply > misc. **New content:** a structure-value seed table (~20-30 rows; `army_value` is 0.0 for structures and `TACTICAL_BONUS` covers units only — the table does not exist today). Same Liquipedia-provenance style as SCORING_TABLE; Task 8 refits.
3. **Defendedness** — feed defending units into the existing targeting value system (`army_value × TYPE_VALUE_SCALE + TACTICAL_BONUS`, group_snipe.py:89) — static D including Bunkers/PFs (25.0/15.0 in UNIT_DATA), plus long-range defenders (siege tanks, colossi, tempests) counted as defense at a config multiplier when within a config radius of the target. Posture detection is out of scope — cheap seed only.
4. **Reachability through danger** — path cost through `get_ground_grid` (enemy presence already painted as cost), cached per re-score cadence. No new grid, no influence map.
5. **Kill progress** — damage completion bonus via `1 − health_ratio` through an exponential curve: finishing a wounded target beats opening a fresh one. Curve weights must let a heavily-defended near-dead target lose to a fresh undefended one — no special case.
6. **Belief freshness × confidence** — multiplier, never additive (Task 4's `BeliefConfidence` template); stale intel discounts the score, it doesn't zero it.

- Per-squad commitment margin for any thrash that survives the curves. Seeds hand-set now, fitted from telemetry later (Task 8's machinery applies when scheduled).
- Reporting: the 12 `current_attack_target` consumers (telemetry `attack_target` field, debug overlay) read the computed `army_focus` — the strongest squad's target, reporting only, same field names (constraint 4). One new per-squad transition event (emission-on-change) is the gate instrument.
- **Sequencing:** plugs in where `attack_target()` is called today, and both callers are named: `handle_attack_toggles` (bot.py) receives per-squad candidates; the observer lead (scouting.py:1093) consumes the computed `army_focus` as its lead anchor — it reads reporting state, it does not drive combat behavior. Army-wide engage/retreat stays until its own task. May execute before Task 8 — its seeds are hand-set; the fit applies to it when Task 8 runs.

**Acceptance:**
- [ ] unit tests pin every consideration's curve shape and the canonical combination before integration (unit)
- [ ] the 7-step rule chain and the two supply thresholds are gone; `current_attack_target` is reporting-only (offline)
- [ ] structure-value seed table passes the format-validation pattern (valid `UnitTypeId` keys, no duplicates) (unit)
- [ ] squad-target transition events emitted emission-on-change; gate metric: target-switches-per-minute in mop-up, before vs after (scenario/lab)
- [ ] endgame regression tape: last-two-buildings scenario settles on one target and ends the game — no hour-timeout ties (scenario)
- [ ] full roster regression vs master within the noise band (lab)

## Adoption Roadmap (Phase 2-3 — not tasks in this plan)

Phase 1 proves the foundation on three managers. The remaining decision surfaces adopt the same reasoner in ROI order:

| # | System | Current mechanism | Phase | Gates |
|---|--------|-------------------|-------|-------|
| 1 | Combat engagement — `handle_attack_toggles` engage/retreat | combat-sim gate + sticky duration timers | 2 | unit + scenario |
| 2 | Chronoboost + Energy Recharge — target selection | first-match priority lists | 2 | roster |
| 3 | Expansion — `_has_map_control` / `expansion_checker` | hard gates + threshold conjunctions | 2 | roster |
| 4 | Defender allocation — threat response sizing | graduated severity table | 2 | unit + scenario |
| 5 | Mass Recall — `try_mass_recall` gates | three AND-gates | 2 | unit + scenario, no roster |
| 6 | Macro leftovers — tech-invest, gateway/forge counts, detection cannons, shield battery | booleans/thresholds | 2 | roster |
| 7 | Per-unit micro — blink, storm, disruptor nova | thresholds | 3 | unit + scenario |

Gate principle: constructible states → unit + scenario (Tier A constructs the score math; Tier B spawns the state and judges on transition events, never win rate); emergent behavior → roster. Phase 2 prerequisite: the scenario stub bot (~100 LOC, hosted in `tests/`, modeled on the existing test bots, debug-create-unit spawn at a fixed game loop) — test infrastructure in-tree, ours, subject to the same consent rules as any lab bot; no ProBots submissions. Row 1 (engage/retreat) is the army-wide decision that Task 10's per-squad targets feed; per-unit targeting stays row 7 (Phase 3).

## Verification (test channels + the lab as harness)

Every acceptance criterion names its channel. Channels, cheapest first:

- **Unit (offline):** pure-Python tests pinning scoring math — no SC2 binary, no game. Runner: pytest via the Poetry dev group (`poetry run pytest` from repo root); suite root `tests/unit/` with `testpaths` pinned in pyproject so the opponent test bots are never collected; `poetry run pytest` passing is a merge condition; CI runs this channel on PRs into `development` (`.github/workflows/ci.yml`). The persisted suite starts with Task 2's format-validation test; OpenCode writes tests first (constraint 2). `IntelContext` is the test seam — unit tests construct it directly, no game required.
- **Local game:** one-off run.py game vs in-game AI; debug overlay on for import/render surface. Minutes.
- **Scenario:** DebugSpawn-assisted local run for states that cannot be manifested by playing (recall, engagement edges, allocation). Phase 2 prerequisite: the scenario stub bot (~100 LOC, `tests/` host). Judged on transition events, never win rate.
- **Lab:** series, banked tapes, roster, Compare tab, branch worktree mounting (runner `branch` param).

**Noise band (one definition, reused by every "within noise"/"no-worse" gate):** no-worse = within ±1 game per 3-game series or better, series length stated in the PR. If 3-game series prove too noisy in practice, the fix is a stated longer series, not a vaguer criterion.

- **Branch topology:** `feature/task-N` branches from `development` and PRs back into it. Gates run on the channel named in each acceptance — unit/offline gates locally via `poetry run pytest`; lab-channel gates on the task-branch worktree (runner `branch` param) before merge. Agents never push to or merge into `master` — promotion is the owner's decision.
- **Behavior-level:** the banked before-tapes re-run on the branch, each labeled with its failure layer:
  - Cannon rush: the constructed Who CannonRush (lab DB `test_group_id=8`, matches 985–988, LeyLinesAIE_v3 — fully recorded and reproducible) — 3/3 wins, 0/3 responses, belief peak p_cheese 0.983 with zero action. **Decision-layer failure:** detection and confidence were right; routing failed to act. This is Task 5's PR gate tape — target: action taken (gate details in Task 5's acceptance). Chance/ProxyCannoneer = optional hardening, not a gate. Evidence basis: baseline = banked records + code-path elimination; the re-run supplies the measured version (986–988 per-game telemetry not retained — lab-DB-confirmed). **Caveat (semantics of the 0/3):** `used_cheese_response` has meant two things across its history — live-state until ee01454 (2025-10-27) made it a one-way latch, then live-state again after 1640601 (2026-06-16) replaced the latch with the `is_cheese_response` property. If the banked tapes ran post-1640601 code, a fired-then-cleared response records False and "never fired" is unproven from banked records alone — the re-run resolves it. The PR must state which code version the banked tapes ran.
  - Worker rush: PiG_Bot vs Who (WorkerRush) — 3/3 losses, 0/3 responses, belief peak p_macro 0.795. **Perception-layer failure:** the model classified MACRO and guards are deliberately fallback-only (model-authoritative design — guards fire only when the model is missing), so scoring never received a p_cheese to act on. Utility scoring cannot flip this tape; the fix is model-side (worker-rush features/training in the sklearn pipeline), owned by the belief layer — not guard rewiring. Report the delta; do not gate the PR on it.
- **Macro lane:** at least one non-cheese series from the roster (e.g., MicroMachine) checked for composition-scoring stability — the refactor is bot-wide, not defined by the rush demo.
- **Regression:** full roster series (MicroMachine, Clicadinha, Who-random) master vs branch, compared in the lab's Compare tab.
- **Telemetry gates:** firing rate judged on `reactions` activation events (not the `used_cheese_response` end-state field — live-state snapshot, blind to fired-then-cleared; stays reported per constraint 4), response latency via event `game_time`, win-rate deltas per matchup. No regression vs master without a documented reason.


## Log

The plan stays forward-looking; verification history and progress notes are recorded here.

- **Pre-work:** Lab runner `branch` param confirmed — branch runs mount the `development` worktree.
- **2026-10-19 — Task 2 partial:** +34 rows added (61→95), 0 pair values changed vs the pre-#5 table; 0 duplicate keys, values −10.0..12.0. 8/9 buildable units have attacker rows (Observer exempt — no weapon). Remaining gap: mirror targets (Dark Templar, Disruptor, Mothership as enemy targets — 0 rows). Format-validation test added in 77dc34e, deleted in 0698240 per the run-then-delete convention — a persisted test is still owed.
- **2026-11-02 — v0.4 cleanup:** Scope fence clarified — out of scope is combat micro execution (runtime behavior); mechanical import/comment updates in combat files are in scope. Task 1 blast radius verified: 6 import sites; `render_target_scoring_debug` keeps its name. Principles merged into the Design language (each rule now stated once); historical annotations moved here from task bodies. Task 2 gap list refreshed from the live table: mirror targets (DT/Disruptor/Mothership as enemy rows, 0) remain; version-tagged firing-rate history (2/104 v0.13.1, 0/7 v0.13.2, 0/6 banked; p_cheese 0.983 no-action) moved here from Task 4.
- **2026-11-02 — v0.5 scope correction:** Goal reframed from "three managers" to bot-wide foundation with the managers as first adopters. New Task 3 (reasoner library + IntelContext) added ahead of the manager conversions; old Tasks 3-7 renumbered to 4-8 (task numbers in pre-0.5 log entries refer to the old numbering). Adoption Roadmap added for decision surfaces the original scope missed: combat engagement, chrono/recharge, expansion, defender allocation, recall gates, macro leftovers, per-unit micro. "Combat micro already utility-shaped" corrected — true only of `score_target()`; engagement, recall, and choke decisions remain threshold-based (Phase 2 material). Macro lane added to Verification so success is not defined by the rush tapes alone.
- **2026-11-02 — Tape relabeling:** Baselines section folded into the Verification tape bullets (numbers live next to their gates). Rush tapes labeled by failure layer: worker rush = perception (model classified MACRO 0.795 under the deliberate model-authoritative design — guards are fallback-only by intent, so scoring never saw a p_cheese; fix is model-side retraining, not guard rewiring); cannon rush = decision (p_cheese 0.983, zero action — Task 5's own gate). Task 5's firing gate re-scoped to the decision-layer tape; the worker-rush delta is reported but not PR-gating.
- **2026-11-02 — Task 5 acceptance generalized:** response scoring is subtype-agnostic by construction (same `u(respond) vs u(continue)` for all cheese subtypes; handlers as scored candidates) — now pinned by acceptance instead of intent. Added the `reactions` activation/clear transition event so firing is measurable per-activation across ALL_IN/TIMING too, and "fired then cleared" is distinguishable from "never fired" (the binary `used_cheese_response` match field stays as-is per constraint 4). Cannon-rush tape remains the PR gate instance, not the definition of success.
- **2026-11-02 — v0.6 acceptance-to-channel review (Praxis-approved, lab-DB-verified):** All 8 decisions applied. (1) Task 1 acceptance re-channeled — import-clean + pytest-collection-parity + one debug-overlay local game; roster smoke dropped (a rename can't produce roster-visible signal). (2) Roster smoke moved to Task 2 + `UnitTypeId`-validity check added (typo'd keys silently score 0.0 via `.get`). (3) Task 5's gate pinned to the constructed cannon-rush villain (lab DB `test_group_id=8`, matches 985–988 — fully recorded; per-game telemetry not retained, so the re-run becomes the measured version). (4) Task 6 VOI acceptance → offline equivalence test (`enable_scout_voi: False` makes in-game unobservable). (5) Noise band adopted: ±1 game per 3-game series, series length in PR, reused across all "within noise" gates. (6) Test Channels block added to Verification — every acceptance now names its channel (offline/unit/local game/scenario/lab). (7) Scenario stub bot approved as named Phase 2 prerequisite (~100 LOC, `tests/` host, judged on transition events never win rate; in-tree infra, no ProBots). (8) Roadmap rows annotated with gate channels; principle: constructible states → unit+scenario, emergent behavior → roster. Plus Praxis's addition: Task 3's acceptance names `IntelContext` as the test seam — unit/scenario tests construct it directly, which is what makes Tier A legitimate. Mothership moved from Task 2 gap to "excluded by design" (documented in the research doc; Observer exempt — no weapon).
- **2026-11-05 — v0.7 gate-signal fix (git-archaeology, user-driven):** Task 5's firing gate re-pointed from the `used_cheese_response` match field to the `reactions` activation event. Reason: the field is a game-end snapshot of live state — `is_cheese_response` = active AND not `_transitioned` — so a fired-then-cleared response records False and a gate on the field would false-fail a working response. History verified in git: the field was live-state with an explicit reset until ee01454 (2025-10-27) made `_used_cheese_response` a one-way latch so game-end reports would retain "ever fired"; 1640601 (2026-06-16) replaced the latch with the property and the telemetry field inherited the name without the semantics — the pre-ee01454 blindness returned silently. The transition event restores the latch's "ever fired" guarantee in event form; `game_time` added to its payload so response latency becomes answerable. Verification cannon-rush bullet gained a caveat: if the banked tapes ran post-1640601 code, "never fired" is unproven from banked records alone — the re-run resolves it; the PR must state the banked tapes' code version.
- **2026-11-05 — runner pinned (Praxis question, user answer):** pytest is the unit runner — pure-Python scoring math, no SC2 binary, no game. Manifest: Poetry dev group (`poetry add --group dev pytest`, commit pyproject + lock together per AGENTS.md) — not requirements.txt, which stays ladder-era legacy, and no parallel requirements-dev.txt. Suite root `tests/unit/` with `testpaths` pinned in pyproject (structural guarantee the opponent test bots are never collected; they also match no default collection pattern today). `poetry run pytest` passing is a merge condition from Task 2 onward; no CI step for Phase 1. Task 1's acceptance reworded accordingly — it wires config only, since collection is empty until Task 2 lands the first test.
- **2026-11-05 — Branch topology stated (constraint 5 + Verification):** "One task, one PR" and "OpenCode works on the development branch" read as conflicting — they were two halves of one flow, now stated once: `feature/task-N` → PR → `development`; gates pass before merge, each on its named channel (unit/offline locally via pytest; lab-channel gates on the task-branch worktree via the runner's `branch` param). Agents never push to or merge into `master` — promotion is the owner's decision. The old "Branch runs" line's `development` default was pre-task-branch wording; same mechanism, now applied to the task branch.
- **2026-11-05 — Task 1 executed (OpenCode):** `target_scoring` → `scoring_table` (git mv; `COUNTER_TABLE` → `SCORING_TABLE`), all 6 import sites plus doc/comment references updated (macro.py, constants.py, combat files, AGENTS.md, README.md, Liquipedia research doc); `render_target_scoring_debug` keeps its name per scope. Behavior unchanged — table still 95 pairs + 6 SITUATIONAL_ROWS. pytest wired per v0.8: Poetry dev group (`poetry add --group dev pytest`; pyproject + lock committed together), `[tool.pytest.ini_options]` with `testpaths = ["tests/unit"]` and `pythonpath` mirroring run.py's ares submodule paths; suite starts with Task 2. Env findings for the record: the local poetry env carried orphaned `ares-sc2 2.52.1` metadata with missing package files (`poetry sync` repairs it), and ares is importable only via the submodule paths `ares-sc2/src/ares`, `ares-sc2/src`, `ares-sc2` — because poetry installs the path dep as `site-packages/src/ares`, the bare `python -c "import bot"` form of Task 1's acceptance needs those paths (pytest's `pythonpath` supplies them inside the suite). Windows console additionally needs `PYTHONIOENCODING=utf-8` for run.py's emoji prints. Follow-up (same PR): `pandas` was an **undeclared runtime dependency** of the strategy belief model — `sklearn_inference.py` builds a one-row DataFrame for `predict_proba`, and pandas only existed transitively via the old PyPI ares-sc2 2.52.1 stack, so `poetry sync` removed it and the model silently degraded to rules (the lab was unaffected — the banked p_cheese 0.983 proves the model loads there). Now declared in the Poetry main group. Verified: `strategy_model_sklearn.pkl` loads (18 features, 4 classes) and predicts. Note for later: the DataFrame is not required — a numpy array in `feature_cols` order gives identical classes and probabilities against the fitted pipeline (plain OneHotEncoder keys on position, not name), so the dependency can be dropped from the runtime path later if the belief layer is ever opened for changes.
- **2026-11-06 — Task 2 executed (OpenCode):** SCORING_TABLE 95 → 135 pairs (+40, all Liquipedia-annotated, no existing pair value touched — diffed against HEAD: 0 modified, 0 removed). The audit found the ATTACKER side already complete and the gap on the TARGET side: our own unit types as the enemy (PvP mirror) — Dark Templar, Disruptor, High Templar, Archon and Sentry had 0 target rows, Immortal/Colossus 2 each. Rows added per unit: Dark Templar 5 (Disruptor/Stalker/Immortal/Archon +, Zealot −), Disruptor 6 (Immortal/Stalker +, Archon/Zealot −), High Templar 5, Archon 6, Sentry 3, Immortal 2, Colossus 2, plus the Feedback line (HT → Phoenix/Oracle/Sentry), the Nova one-shot line (Disruptor → Zealot/HT/DT/Adept) and the audit holes (Colossus − Marauder/−sieged Tank, Stalker − burrowed Lurker). Sources: LotV wikitext for DT/Disruptor/HT/Archon/Sentry/Immortal/Colossus via the Liquipedia API (regular pages 403; API + descriptive User-Agent returns 200). Suite's first test landed — `tests/unit/test_scoring_table.py`, 10 checks, `poetry run pytest` → 10 passed; the duplicate check parses the source with `ast` (a dict literal silently keeps the last value for a repeated key) and the composition roster is read from macro.py with `ast` so the unit test never imports a manager. Mutation-checked: duplicate pair, string key, 120.0 value, target-side hole and composition-side hole each fail the suite. Corrected one stale claim in the table's Sentry header ("no direct attack in LotV" — the LotV Sentry has a weak Disruption Beam; the row values are untouched and still encode GS/FF support). NOT DONE: the roster/noise-band gate (lab channel) — no runner access in the task environment; owner to run master vs this branch before merge.
- **2026-11-07 — Task 3 executed (OpenCode):** the shared reasoner library landed in `bot/utilities/reasoner.py` — response curves (linear / exponential / logistic), the `Appraisal` type (base score, multiplier, veto), the canonical runner (`score(choices) -> ranked candidates`, deterministic argmax), the frozen `IntelContext` snapshot, and the `Consideration` protocol (`load(params)` + `evaluate(context) -> appraisal`). No manager code touched. Canonical math as implemented: `score(choice) = (Σ base_scores) × (Π multipliers)`, any veto ⇒ 0; ties keep the candidate's input order (stable sort, so the argmax is reproducible); a candidate with no considerations scores **0** — the empty sum is 0, not 1, so an unscored candidate never wins by accident. `combine()` folds one box's appraisals (sum of bases, product of multipliers) and reports the veto instead of applying it, so a vetoed box stays visible in the trail; a compartmentalized box is expressed by folding it into a single appraisal (base = the box's score, multiplier = 1.0), which keeps its confidence local. Curve shapes: `linear` = normalized identity; `exponential` = `(e^(k·x) − 1)/(e^k − 1)` on the normalized input, sign of the steepness picking convex (positive — the threat-like "distant noise ignored, commitment near critical" shape) or concave (negative — saturating value shape); `logistic` = sigmoid, exactly 0.5 at the midpoint. Both curved shapes are written with only non-positive exponents, so a large steepness cannot overflow. Curve parameters live in `config.yml` → `UtilityAI.Curves.<name>` (each entry: `kind` + `minimum`/`maximum`/`midpoint`/`steepness`) and load through `load_curve_params(config, name)`; curves are named by *purpose*, not by shape, because the normalization bounds belong to the scored quantity (a worker count runs 0–70, a severity 0–1). Seeded entries: `threat` (exponential, steepness 4.0), `response` (logistic, midpoint 0.5, steepness 8.0), `value` (linear) — Task 8 refits these from telemetry. The library imports **stdlib only** (the dependency direction stays managers → library); two tests pin that — an `ast` parse of the source, and a subprocess that loads the module by path and asserts no `bot`/`sc2`/`ares` module entered `sys.modules` (importing it as `bot.utilities.reasoner` runs `bot/__init__.py`, which pulls in the game stack, so the by-path load is what proves the library itself is clean). `IntelContext` is a frozen dataclass of plain scalars and mappings (belief snapshot, threat severity, intel urgency/freshness, economy, tech progress) — the test seam: scenario and unit tests construct it directly, no game. 39 new tests (49 total, `poetry run pytest` → 49 passed), mutation-checked as listed in the Task 3 acceptance. NOT DONE: the "Tasks 4-6 consume it" criterion is forward-looking — the adopters land in their own PRs.
- **2026-11-09 — Task 4 executed (OpenCode):** the macro manager's composition decision became a scored competition. The new subsystem `bot/managers/composition_scoring.py` owns the ballot, the `IntelContext` and the incumbent; `select_army_composition` now asks it for a winner, re-orders the priorities for production mechanics and reports the choice. **Pipeline replaced:** profile base → sticky one-way economy switch → Archon-percentage threshold → counter nudge (±10%) → strategy nudge (±10%) → resource-pressure nudge (±15%) → priority reorder → clamp/re-normalize. **Pipeline now:** the profile's compositions, the counter-driven variants (0.15 of proportion shifted from the least effective buildable unit to the best counter) and the hold candidate (the composition we already decided to build, restricted to the build's unit types) compete on the canonical score, capped at 8, deterministic argmax. **Considerations** (each `load(params)` + `evaluate(context) → appraisal`, every raw input through a response curve): `CounterEffectiveness` — the scoring table's own magnitude, now the *core base score* instead of a post-decision nudge (share-weighted over the believed enemy army, `SITUATIONAL_ROWS` excluded); `EconomyFit` — affordability headroom (what the economy can spend per minute, scaled by saturation and worker coverage, minus the mix's price); `TechSynergy` — the share of researched upgrades the mix exploits (Warp Gate/Charge/Exteneded Thermal Lance/Psi Storm); `CostBurden` — resources plus supply (production time) of one army member, scaled by income tightness; `BeliefConfidence` — a multiplier, never an additive term; `Commitment` — the scored hysteresis that replaced the sticky switch. **Removed:** `_economy_switch_triggered` (bot state), `economy_switch_threshold`, `archon_switch_threshold`, `nudge_proportions`, `strategy_nudge_proportions`, `resource_pressure_nudge`, `_FakeEnemy`, `_compute_effectiveness`, `PRODUCTION_MAX_NUDGE`, `PRODUCTION_MIN_ENEMY_UNITS`, `STRATEGY_NUDGE_MAX`, `STRATEGY_NUDGE_THRESHOLD`, `RESOURCE_PRESSURE_MAX_NUDGE`. The Archon gas gate survives as `archon_merge_gas_requirement` — it gates HT *merging*, not a composition switch, so it keeps its meaning under a truthful name. `army_composition_2` stays a plain scored candidate. **Seed magnitudes + provenance:** the counter term carries the table's own scale (Liquipedia-annotated pairs, −10..+12) over a symmetric −8..+8 linear band (a neutral matchup scores exactly zero, and a value scale keeps its resolution where a sigmoid would saturate); the economy band is ±600 resources/min of headroom around break-even (a rich economy's headroom runs past the band, so affordability stops separating the candidates); the cost band is 0..560 (the priciest single-unit mix: Colossus 300+200 plus 6 supply priced at 10 resources a point); income is a 0..2000 sigmoid feeding the cost term's tightness (`get_economy_state` calls 1600 minerals/min "full"). Weights: counter 2.0 (core), economy 1.0, tech 0.5, cost 0.6 (guards), commitment 0.03. The commitment margin (0.03) is seeded from the two gaps the scorer actually produces: a 15-point counter shift against a matching enemy is worth 0.05–0.09 of base score, a 5% wobble in the believed shares is worth ~0.01 — so a real counter advantage takes the crown and belief noise does not. **Behavior delta, deliberate:** the old Archon switch's intent (stop heavy HT production once HTs have merged) is *not* reimplemented as an HT-specific rule — the fence forbids scenario special cases — so `army_composition_2` now wins only when it outscores the alternatives; the post-merge mix may persist a little longer. **Telemetry:** new `composition_choice` transition event under `subsystem="macro"` (winner, winner score, runner-up score, ballot size) — fires only when the winner changes, so existing event names and fields are untouched. **Verification:** `poetry run pytest` → 82 passed (49 pre-existing + 33 new); the removal proof is grepped in the PR body; offline calibration scripts confirmed that a ±4% belief wobble keeps the incumbent while a real enemy change (Roaches → Mutalisks) takes the crown and the composition switches back when the enemy returns — the one-way behavior the sticky flag baked in is gone. **NOT DONE:** the full roster regression vs master (lab channel) — no runner access in the task environment; the owner runs it before merge, ±1 game per 3-game series.
- **2026-11-11 — v1.0, Task 10 added (squad attack-targeting, user-driven):** the attack-targeting conversion woven in after Task 4 (Task 9 was already taken by the semi-tree). Objectives reviewed against source before weaving: the 7-step rule chain and both supply thresholds confirmed at `attack_target` (combat.py:2101-2177); the endgame tie attributed to the linear equidistant flip at the 25-unit² keep-valid boundary. Six considerations, each reusing verified machinery except one new deliverable: proximity (exponential, vault principle — solves the mop-up, no endgame mode); strategic value (**new structure-value seed table ~20-30 rows — `army_value` is 0.0 for all structures, `TACTICAL_BONUS` covers units only; production > tech > supply > misc, Liquipedia provenance, Task 8 refits**); defendedness (existing `army_value × TYPE_VALUE_SCALE + TACTICAL_BONUS` system from group_snipe.py:89; PF 25.0/Bunker 15.0 confirmed in UNIT_DATA; long-range defenders as defense at a config multiplier within a config radius — posture detection out); reachability (path cost through `get_ground_grid` — enemy presence already painted as cost; no new grid, no influence map, cached per re-score cadence); kill progress (`1 − health_ratio` through an exponential curve — finishing beats opening fresh; curve weights must let a defended near-dead target lose to a fresh undefended one); belief freshness (multiplier, `BeliefConfidence` template). Per-squad decisions with per-squad commitment margin; the bot-global `current_attack_target` becomes reporting-only via a computed `army_focus` (strongest squad's target; the 12 consumers — telemetry `attack_target` field, debug overlay — keep their field names, constraint 4); one new per-squad transition event (emission-on-change) is the gate instrument — target-switches-per-minute in mop-up, before vs after. Sequencing: plugs in where `attack_target()` is called today, feeding `handle_attack_toggles` per-squad candidates; army-wide engage/retreat stays until its own task; may execute before Task 8 (hand-set seeds; the fit applies when Task 8 runs). Roadmap row 1 annotated: Task 10's per-squad targets feed it; per-unit targeting stays row 7.
