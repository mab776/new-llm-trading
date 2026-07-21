# GOOD_IDEAS.md — set aside, but positive for growth

Ideas that showed **real positive evidence** but are not deployed — parked for a reason
(blocker, timing, or discipline), not because they failed. The complement of the research
graveyard: everything here *worked* in some validated sense and is worth revisiting when its
blocker clears. Keep entries brief; the linked artifacts hold the full story. Each entry
carries an *Origin* note (how the idea came up) when the story is known — keep doing that
for new entries; provenance is half the value.

> ⚠️ Discipline reminder: nothing here is pre-approved. Revival = fresh pre-committed
> protocol (select-TRAIN / report-TEST; the worn holdout is invariance-only; the LIVE track
> record is the real OOS) + Marc's explicit go + supervised deploy.

---

## 1. Walk-forward adaptive retuning — the biggest shelved edge (~2× vs static)
*Origin: grew out of the optimization rounds themselves — every era's search kept picking a
different winner, raising the obvious question: if no single config rules all regimes, what
does periodically re-searching on a rolling window buy?*

Periodically re-searching config parameters on a rolling window ~**doubled** median
continuous growth vs the static config (60/300/1,000-trial studies: median ratio 1.64 → 1.89
→ **2.02**; wins 80–100% of trials, including the previously weak 2025H1).
**Why parked:** parameter selection never converges (every seed picks a different winner) and
operational retuning would confound live execution validation.
**Unblocks when:** the live/maker/slippage calibration questions are settled and there's an
appetite for an auto-retune pipeline. Explicitly retained as the **leading post-paper research
item**. Artifacts: `opt/walk_forward_robustness_results.json`, `opt/README.md` (walk-forward
section).

## 2. Min-size rescue / conditional cap-overshoot — ALL GATES PASSED, awaiting go
*Origin: Marc, 2026-07-20 ~01:30, watching the 8pm bar live: ETH fired STRONG +31.2 and got
MIN_SIZE_SKIPped at $0.10 free margin while BTC held the whole cap — "what if we could go
above the max margin, just if squeezed to min-size cancel and the signal is strong enough?"
Third idea of a late-night chain (reserved slices → rotation → this); the only gate-passer.*

Rescue MIN_SIZE_SKIPped entries by flooring to the exchange minimum when the signal is
strong, within caps ×(1+O). Probe (2026-07-20, `opt/probe_overshoot.py`, commit `b5fe4de`):
rescue-vs-skip is **split-consistent** — TRAIN +577→+624 geo, TEST +302→+325..339, holdout
5.65×→6.01× (ratio 1.063), worst-folds improve everywhere. O is inert (rescued lots are
slivers; 1.25× stretch never binds); the S=30 selectivity edge over plain floor is TRAIN-only
(noise) — the solid claim is just "rescue > skip at small balance."
**Why parked:** needs scheduler-side code (the live `REFUSED` path) + supervised deploy;
deliberately violates sized-risk discipline (min lot ≈ 2.4× the 2% size at $193).
**Self-retiring:** skips fade ~20% tax @$100 → 0 @$2500 — worthless once the account grows.
**Unblocks when:** Marc says go (soonest-value item on this list — it fires today).

## 3. Unconditional "floor" sizing (subsumed by #2)
*Origin: Marc, 2026-07-16, during the $100 go-live sizing sims — floor sub-minimum sizes
instead of skipping; validated slightly better the same day but shelved for fail-closed live.*

Marc's original idea: bump sub-minimum sizes to the exchange minimum, always.
`opt/sizing_scenarios.py`: 4.27× vs 4.00× (skip) @$100; in the overshoot probe the floor
control was the **best TEST arm** (+339.0). Same blocker/self-retirement as #2 — if #2 is
ever implemented, choosing "floor everything" vs "S=30 gate" is a coin-flip the data can't
settle (splits disagree); the conservative pick is the S-gate.

## 4. Cross-asset rotation as a ROBUSTNESS knob (growth version rejected)
*Origin: Marc, 2026-07-20 ~00:45 — "signal_flip gives up a position when its own symbol
turns; what if cross-asset: when one weakens a lot and another rises a lot, give up to
switch?" Searched the same night; growth verdict negative, robustness signal unexpected.*

Evicting the weakest other-symbol position for a cap-squeezed STRONG entry failed the growth
gates (TRAIN winner failed TEST — noise). **But** (post-hoc, unselected): several rotation
cells improved **worst-fold** returns on BOTH splits (TRAIN worst +181→+213, TEST +221→+274).
**Why parked:** the observed effect is not what the protocol selected for.
**Unblocks when:** someone wants a robustness/DD-targeted pre-committed protocol (select on
worst-fold, not geo). `opt/probe_rotation.py`, commit `d8b4a24`.

## 5. Continuous (tanh) alignment — reproducibility, not growth
*Origin: postmortem of the first live loss (04:00 UTC Jul-17 bar) — a near-zero 1h trend
flipped its flat ±5 vote between data vintages, teleporting the score across the −20 exit
cliff. "Votes should scale with conviction" fell straight out of the incident.*

Replaces the discrete per-TF alignment vote with `scale·tanh(trend/k)`: return-neutral on
TRAIN/TEST but kills the "±5 teleport" tail (tiny data wobble near zero can no longer flip a
full vote across the −20 exit cliff — the mechanism behind the first live loss). Largely
superseded by `1h: 0`, but the 1d×3 vote still teleports.
**Why parked:** zero return upside; the reproducibility win shrank after `{"1h":0}` shipped.
**Unblocks when:** another live/backtest divergence traces to a teleporting 1d vote.
Staged knob `alignment_mode` + `opt/probe_alignment.py`.

## 6. Williams %R / stochastic band loosening — audited watchlist
*Origin: side-product of the 2026-07-19 hat-number fragility audit (AST-perturbing every
scorer trigger constant ±15%): nothing hit the flag threshold, but these two bands were the
only constants with a consistent positive direction.*

The hat-number fragility audit (2026-07-19) found the scorer's trigger constants robust
overall, but two bands sat just under the flag threshold with a consistent direction:
**williams −20/−80** (loosening improved FULL 568→786×) and **stoch 80/20** (same pattern).
**Why parked:** below the pre-committed flag threshold; searching them now = data-mining the
same worn folds.
**Unblocks when:** a few clean live weeks exist; then full protocol (these are the only two
scorer constants with a positive direction on file).

## 7. 15m Donchian + vol-expansion scalper — a working second product, outclassed
*Origin: Marc's overnight research ask, 2026-07-19 — "can a scalper make money on 5–15m as a
second product?" ~13k backtests later: mostly no; this was the one survivor.*

Sole survivor of ~13k scalper backtests: 15m Donchian-96 breakout gated by ATR-expansion
≥1.3, ~**10%/yr @ ~8% maxDD** (TEST ≈ HOLDOUT — it generalizes). Parked by Marc 2026-07-19:
not worth a live path vs the 4h product.
**Why parked:** opportunity cost, and ⚠️ its holdout is **SPENT**.
**Unblocks when:** wanted as a diversifier on new (live/paper) data only — no more backtests.
`opt/scalp/SCALPER_RESEARCH.md`.

## 8. Balance growth is free alpha — the deposit lever
*Origin: Marc's +$100 deposit (2026-07-19) prompted the balance sweep that quantified the
granularity-tax curve — turning "more capital helps" from a hunch into a schedule.*

Not a strategy idea, but the cheapest validated growth on file: the small-account granularity
tax falls ~20.5% @$100 → 10.7% @$193 → ~6% @$250 → 2.2% @$1000 → ~0 @$2500 (balance sweep,
2026-07-19). Every deposit buys back skipped/squeezed trades with zero model risk (bot sizes
off realized balance each decision — no restart needed). Also the natural retirement path for
ideas #2/#3.

## 9. Aggressive profile — validated headroom, deliberately capped
*Origin: designed alongside standard from the start as the two-profile deliverable of the
optimization campaign; capped by the go-live decision, not by evidence against it.*

The aggressive configs hold up on the clean OOS holdout (**37.3× / 32.7% MTM DD** vs standard
5.90× / 14.2%) and inherit every strategy fix via `_extends`.
**Why parked:** live validation runs on the standard profile only, tiny capital, by explicit
decision — execution realism (maker fills, slippage, TP fees) must be measured before 2×-ing
the risk.
**Unblocks when:** the live track record earns it (post maker-vs-taker decision, clean weeks,
larger balance).

---

**Already-scheduled decision (not an idea):** maker-vs-taker entry, ~**2026-07-30** — the 86%
rule on the live fill funnel decides; every maker placement is evidence. Don't preempt it.

**Not on this list (graveyard — don't re-pitch without a new mechanism):** LLM gate/consensus,
decay exits & slope gating, marginal half-size, 1d adx_di overlay, regime-switching overlays,
anti-martingale sizing, 1h/5m static transplants, 5m scalping & 15m mean-reversion, reserved
per-asset capital, rotation-for-growth, consecutive-loss penalty, NeuTTS-style CPU ideas.
See `opt/README.md` round history and the probe results files.
