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
**Re-validated on the 1w-vote base 2026-07-23** (`opt/walk_forward_robustness_results_2026-07-23.json`,
Marc's rerun ask): **18/18 runs still beat static** (static itself rose 10.91×→16.75× — the deployed
1w vote banked part of the old gap), median ratios 1.36 / 1.76 / 1.76 at 60/300/1,000 trials (was
1.64 / 1.89 / 2.02), **2025H1 now wins 13/13** (was the weak window), tuned windows don't worsen DD
(worst 14.3–17.6% vs static 16.3%). Winner non-convergence unchanged: 13 seeds-×-windows, all unique,
static never chosen — the blocker stands. NEW unselected observation: all nine 1,000-trial winners
picked trailing **callback 0.25–0.27 (vs static 0.33, bottom of the sampled range) + activation
0.70–0.99 (8/9 below static 0.94)** — tighter/earlier trailing, same family as GOOD_IDEAS #10;
one more DD-robustness data point, not a selection.

## 2. Min-size rescue / conditional cap-overshoot — ✅ DEPLOYED LIVE 2026-07-23
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

## 9. Aggressive profile — ✅ DEPLOYED LIVE 2026-07-22 (sub-account)
*Origin: designed alongside standard from the start as the two-profile deliverable of the
optimization campaign; capped by the go-live decision, not by evidence against it.*

The aggressive configs hold up on the clean OOS holdout (**37.3× / 32.7% MTM DD** vs standard
5.90× / 14.2%) and inherit every strategy fix via `_extends`.
**Unblocked early by Marc's explicit decision 2026-07-22:** running live on a dedicated Bitget
**virtual sub-account** (~$169, funded via XRP) in parallel with the standard bot — a separate
account is mandatory (one-way mode = one position book per symbol; two bots on one account
collide and fail reconcile). Pre-go gate (`opt/aggressive_live_gate.py` + results): chain
reproduces the 37.30× anchor exactly; per-asset BTC 2.00× (alignment fix cured the old 0.71×
loss) / ETH 12.67× / SOL 8.49×; **⚠️ $100 sits on a quantization cliff** (2.82×, 37% of trades
lost) — the working band starts ≥$115. Ops layout: PAPER_LIVE_READINESS_REVIEW.md
§ "Aggressive sub-account bot". Maiden bar (20:00 UTC Jul 22): BTC+ETH maker entries filled
2/2 zero-retry while the capped standard bot MIN_SIZE_SKIPped the identical signals.

## 10. Trendline-tightened stops as a ROBUSTNESS knob (growth version rejected)
*Origin: Marc, 2026-07-23 — the live whipsaw day put every ATR stop just above the rising
trendline BTC tagged and bounced from; "the bot doesn't see geometric supports?". Searched
the same day; growth verdict negative, DD signal unexpected.*

Causal swing-pivot/trendline features (`opt/fastbt.py` structure knobs, `opt/probe_geometry.py`).
Growth gates: entry-proximity gate **catastrophic** (0/9, best −66 geo — extended entries ARE
the edge), structure-break exit 0/6, and the structural-stop TRAIN winner
(trendline/**tighten** w5 b0.25: pull the SL up to just below the rising trendline when that
is closer than the ATR stop) passed TRAIN (+171.7 vs +166.4) but **failed TEST** (+63.2 vs
+66.8) = split-disagreement noise. **But** (post-hoc, unselected): that cell is return-neutral
on FULL (300.8 vs 301.6 geo/f) while cutting **max DD on every split** — TRAIN 14.5↓15.6,
TEST 18.6↓23.9, FULL 18.1↓23.5. Same pattern class as #4. Note: on the motivating live day
the ATR stops were already tighter than the trendline — even this variant would have traded
that day identically; the stop-outs were the strategy's real cost.
**Why parked:** the DD effect is not what the protocol selected for.
**Unblocks when:** a robustness/DD-targeted pre-committed protocol (select on DD/worst-fold,
not geo) — natural to run jointly with #4. Full grids: `opt/probe_geometry_results.txt`.

---

## Priority queue (Marc + review, 2026-07-23) — what to harvest next, in order

**Low-hanging fruit (evidence already paid for):**
1. ~~Min-size rescue (#2)~~ — **DEPLOYED 2026-07-23** (commit `fc171d6`; release gates
   re-passed on the 1w base with true splittable floors: TRAIN +687 vs +621, TEST +304
   vs +270, holdout 6.96/6.57 = 1.060; watch live MIN_SIZE_RESCUE records).
2. **Walk-forward retuning (#1)** — the biggest known shelved edge (~2× vs static);
   unblocks after enough clean live weeks to trust the pipeline end-to-end. NOW THE TOP ITEM.
3. **DD-robustness protocol (#4 + #10 jointly)** — two INDEPENDENT probes (rotation,
   trendline-tightened stops) each showed drawdown-cutting at ~zero return cost as
   unselected observations. A pre-committed DD/worst-fold-targeted search is the most
   evidence-backed NEW experiment on the shelf. Run when a lower-DD variant matters
   (e.g. before any size-up).

**Textbook gaps never searched (mechanism-ranked, none validated yet):**
- ~~**Cross-market context votes**~~ — **SEARCHED 2026-07-23, REJECTED at gate 1**
  (`opt/probe_context_votes.py` + results): daily BTC→ETH/SOL, DXY-inverted and SPX
  trend votes through the alignment machinery ALL hurt TRAIN geo (+656.8 baseline vs
  +587.7/+589.6/+599.0 best cells), monotonically worse with weight — external daily
  context injects noise into the self-referential momentum machine; the 1w vote's win
  was same-symbol, not "external context". TEST/holdout never consulted. Unselected
  observation (same class as #4/#10): btc-5 improved TRAIN worst-fold (+224.8 vs
  +209.9) at heavy geo cost — variance compression, not adoptable, but one more data
  point for the DD-robustness protocol. Engine knob `context_votes` + pinned
  history/external/{dxy,spx}_1d.csv retained for any future DD-targeted re-search.
- **Derivatives positioning data** — zero perp-native signals beyond funding-as-cost: no
  open interest, no long/short ratio, no liquidation clusters. OI divergence is THE
  textbook crypto-perp signal. Blocker: historical data acquisition (Coinglass paid,
  exchange OI history patchy) — a data project before it's a probe. NOW THE TOP
  new-research candidate (context votes fell 2026-07-23), tempered by that rejection:
  external daily series through the score already failed once; OI at least is
  same-symbol + perp-native.
- **Event-calendar risk pause** — no new entries in the 4h bar containing FOMC/CPI.
  Real mechanism, public calendar, cheap probe. Medium prior (this system LIKES vol).
- **Correlation-aware exposure cap** — 3 same-direction crypto positions ≈ one big
  position; caps treat them independently. Risk-control research only — every
  portfolio-structure probe so far says the shared pot's cross-subsidy is the edge.

**Textbook ideas this system has FALSIFIED for itself (don't relearn them):** buy-near-
support entry gating (worst probe result ever: −66 geo/fold), stop-below-structure
widening, entry confirmation/delays, cutting decaying momentum early, marginal
half-sizing. The strategy is a momentum machine — mean-reversion instincts keep
failing its gates.

---

**Already-scheduled decision (not an idea):** maker-vs-taker entry, ~**2026-07-30** — the 86%
rule on the live fill funnel decides; every maker placement is evidence. Don't preempt it.

**Not on this list (graveyard — don't re-pitch without a new mechanism):** LLM gate/consensus,
decay exits & slope gating, marginal half-size, 1d adx_di overlay, regime-switching overlays,
anti-martingale sizing, 1h/5m static transplants, 5m scalping & 15m mean-reversion, reserved
per-asset capital, rotation-for-growth, consecutive-loss penalty, NeuTTS-style CPU ideas,
geometric entry-proximity gating & structure-break exits (probe_geometry 2026-07-23),
daily external context votes — BTC→alts / DXY / SPX (probe_context_votes 2026-07-23).
See `opt/README.md` round history and the probe results files.
