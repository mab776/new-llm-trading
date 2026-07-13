# opt/ — fast backtest harness & config optimization (2026-07-11/12)

## What this is

A ~4000× faster evaluation harness (`fastbt.py`) plus search/validation drivers used to
optimize `config.json` for profit across market regimes. The engine's per-bar
recompute-on-slice is O(n²); since every indicator is **causal** (ewm/rolling), computing
them once on the full series and reading row *i* is numerically identical. `validate.py`
and the engine-vs-fast checks confirmed **exact** agreement with `BacktestEngine` (to the
last digit) on both non-trailing and trailing configs.

| File | Purpose |
|---|---|
| `fastbt.py` | Vectorised indicator precompute + trade simulator mirroring `BacktestEngine.run` (reuses the real `compute_composite_score`/`calculate_targets`/`apply_pre_trade_filters`/`Portfolio`). Adds optional **slippage** and **isolated-margin liquidation** modeling the engine doesn't have. |
| `driver.py` | Loads 2020-08→2025-06 Bitget data once, evaluates a config over yearly / half-year folds; geo-mean objective. |
| `search.py` / `search_wf.py` | Random search; `search_wf` = walk-forward (train on odd half-years, validate on even). |
| `refine.py` | Focused search in the robust (trailing-on) region; ranks by `min(trainGeo, testGeo)`. |
| `ablate.py`, `finalize.py` | One-change-at-a-time ablation; slippage/leverage sensitivity + chronological OOS split. |
| `run_once.py`, `validate.py` | Engine baseline runner; fast-vs-engine exactness check. |

## Critical bug found & fixed on the way (commit 504638d)

The engine ratcheted the **trailing stop up using the current bar's high, then checked the
bar's low** against the raised stop — implicitly assuming high-before-low. Worst-case
intrabar path (low first) must be assumed instead. This was inflating trailing-stop
results **4–10×**. Fixed in `backtesting.py` (exits checked against start-of-bar stop,
ratchet applied after, effective on subsequent bars only); guarded by
`tests/test_intrabar_conservatism.py` (also proves the SL/TP path takes the SL when one
bar spans both). All numbers below are post-fix, and all search results are
**out-of-sample validated** (train folds ≠ test folds) with slippage included.

## Result (fast harness, liquidation modeled, 2021-01 → 2025-06, compounding)

| Config | slip/side | Compound | Worst year | Max DD | Trades |
|---|---|---|---|---|---|
| old baseline (lev 20, no trailing) | 2 bps | 1.30× | −14.1% (2022) | 15.9% | 280 |
| **new aggressive (lev 25)** | 2 bps | **22.5×** | **+10.1%** | 12.4% | 1223 |
| new aggressive | 5 bps | 15.0× | +6.6% | 13.0% | 1209 |
| new aggressive | 10 bps | 7.8× | +0.6% | 14.3% | 1186 |
| new conservative (lev 12) | 5 bps | 4.2× | +8.7% | 7.7% | 1158 |

Per-year (lev 25 @5bps): 2021 +204%, 2022 +82%, 2023 +45%, 2024 +76%, 2025H1 +7% —
**every regime green, including the 2022 bear** (old config lost money there).
Chronological OOS: trained on 2021-23 only, 2024-25 still made 2.47× (10% DD).

## What actually carried the edge (ablation)

1. **Trailing stops ON** (activation ~0.94%, callback ~0.33%) — the one structural change:
   alone it turns every fold green and cuts maxDD ~3×. Let winners run, exit on reversal.
2. **Lower entry thresholds + lighter filters** (strong 21.3 / marginal 12.6, min_adx ~20,
   3-of-5 category agreement, no trend-momentum-agree) — more trades, edge× frequency.
3. **Wider ATR stop (2.26×) with modest R:R (TP1 2.02 / TP2 3.34, 70% out at TP1)**.
4. **Leverage is a clean risk dial** on top: 10→3.3×(DD 7%) … 25→15×(DD 13%) @5bps.
   Sweet spot 25; 30 adds little return for more DD.

## Round 2 — strategy-level changes (same day)

Beyond config knobs, structural variants were implemented behind flags in `fastbt.py`
(`strat=` dict; defaults reproduce the engine) and ablated (`strat_ablate.py`). Three
winners were ported into the real engine/config (all engine==fastbt exact-validated):

| Feature | Config key | Effect (alone) |
|---|---|---|
| **Pyramiding** — up to N concurrent same-direction positions | `position_sizing.max_positions: 3` | 22.5× → 140× (DD 12→23%) |
| **Conviction sizing** — risk × clamp((\|score\|/strong)^1, 0.5..1.5) | `position_sizing.conviction_exponent: 1.0` | +~10% |
| **Opposite-signal exit** — close when composite flips ≥ threshold | `risk_management.opposite_exit_threshold: 20` | better worst-fold/test |

Rejected: ATR-based trailing (worse than pct), long/short threshold asymmetry (worse),
vol-targeted leverage (cuts DD ~40% but halves return — keep in back pocket as a risk knob),
marginal-half-sizing (−60% return).

**Combined result (2021-01 → 2025-06, liquidation modeled):**

| slip/side | Compound | Worst year | Max DD | Trades |
|---|---|---|---|---|
| 2 bps | **312×** | +23% | 21.4% | 2427 |
| 5 bps | 127× | +10% | 25.7% | 2386 |
| 10 bps | 30× | +2% | 28.3% | 2331 |

Walk-forward: train geo +106%/half-year → held-out test +70%. Chronological: trained on
2021-23 only, unseen 2024-25 made 4.36× (2bps). Live parity: scheduler implements all
three features (entry slots, conviction margin, `_maybe_opposite_exit`).

## Round 3 — DD throttle → shipped as a wide circuit-breaker only

Swept a drawdown-aware pyramiding throttle (pause slots / cut risk while balance DD ≥
threshold). **As a profit lever it fails**: at tight thresholds (8–15%) it cuts exposure
exactly when this system recovers (trailing stops exit losers fast, so DDs are shallow
and V-shaped) — return halves, maxDD barely improves, worst-year sometimes flips
negative. Gentle (3→2 slots) variants were strictly worse at 5 bps.

**Shipped instead as tail insurance** (`risk_management.dd_throttle_threshold: 0.25`,
`dd_throttle_slots: 1`, `dd_throttle_risk: 0.5`): at 25% it never triggers in the whole
2021-2025 backtest @2bps (312× unchanged) and costs ~9% @5bps (127→116×), but caps the
bleeding in the one scenario the backtest cannot show — the edge breaking in live.
Implemented in engine + scheduler (in-session peak; resets on restart — see scheduler
note). Tightening the threshold is *expected* to cost return; don't "optimize" it below
~0.20 based on in-sample data.

## Round 4 — funding-rate realism

Perps settle funding every 8h on NOTIONAL (longs pay positive rates). Implemented in
`llm_trading_bot/funding.py`: fetch + incremental disk cache (`history/funding/`,
gitignored — refetches in seconds) + pure per-bar aggregation, applied in both the
engine (`run(..., funding=)`, step 1.5: survivors of the bar's exits settle before new
entries) and `fastbt` (`funding_by_pos=`). **Source is Binance** (full history since
2019; Bitget only serves ~3 months) — rates are arbitraged across venues, documented
approximation. `backtesting.include_funding: true` wires it in `main.py`; live trading
ignores it (the exchange settles funding itself).

Impact on the final config (2021-01 → 2025-06): **~27% of total compound** —
312× → 228× @2bps, 116× → 84× @5bps — concentrated in the 2021 bull (longs paying
peak funding), **every year still green**. Re-tune probes with funding on (easier
shorts, stricter longs, oppexit/tp2/maxpos tweaks) found nothing that didn't flip a
year negative → config unchanged. Engine==fastbt digit-equal with funding
(2024: +368.01%, 608 trades, $19 net funding on $100 start).

## Round 5 — ETH transfer test

The BTC-tuned config, **byte-for-byte unchanged**, evaluated on ETH/USDT:USDT
(same pipeline: Bitget candles → `history/bitget/ETHUSDT_USDT/`, Binance ETH funding):

| ETH (2021-01 → 2025-06) | Compound | Worst year | Max DD |
|---|---|---|---|
| @2bps + funding | **1015×** | +48% | 25.9% |
| @5bps + funding | 296× | +21% | 30.1% |

Every year green (2021 +2175%, 2022 +404%, 2023 +83%, 2024 +227%, 2025H1 +48%);
both interleaved half-year sets strongly positive (+117%/+66% geo). A config tuned
purely on BTC transferring to another asset with *better* results is strong evidence
the edge is structural (trend + trailing + pyramiding), not BTC curve-fit — ETH's
higher volatility simply gives it more room. Engine==fastbt digit-equal on ETH too.

**Running it:** `config-eth.json` is the same strategy pointed at ETH. For both assets
simultaneously, run two bot instances with split capital (e.g. two containers / a
second Portainer service) — the bot is single-symbol by design. Expect BTC/ETH
drawdowns to partially overlap (high correlation), so don't double leverage just
because there are two instances.

## Round 6 — trailing-ratchet CADENCE is the strategy (critical live fix)

Added 1h sub-bar exit replay to `fastbt` (`exit_granularity="sub"`: each 4h bar's four
1h bars replayed in order — real intrabar sequencing instead of the worst-case single-
bar assumption; corrupt 1h stretches auto-masked, e.g. Bitget's 1h perp history is
placeholder junk before 2021-01-02).

**Finding:** with clean data, trailing OFF gives identical results at both granularities
(sanity ✓), but trailing ON collapses under hourly ratcheting: **84× → 5× @5bps**, and
NO wider activation/callback recovers it (all 1h-cadence params sweep to 0.4–5×, most
losing). Ratcheting the stop hourly chokes winners on intrabar noise; ratcheting once
per COMPLETED 4h bar (stop fixed intrabar, exchange triggers on touch) is what the
84×/228× backtests model — and it is implementable live exactly.

**Live fix:** `scheduler._maybe_trail_stop` previously ratcheted every 15-min position
check using the current price (~16× tighter than even the 1h replay — live would have
performed like ~5×, not 228×). Now it fetches the last COMPLETED primary bar and
ratchets once per bar on its favorable extreme (`last_trail_bar` gate). Guarded by
`tests/test_trailing_cadence.py` + the updated scheduler test — do not "improve" this
back to continuous trailing.

## Round 7 — funding as a SIGNAL: measured, REJECTED (no robust edge)

Backlog #1 ("extreme funding = crowded positioning → fade it") — measured, and it does
**not** produce a robust edge for this strategy. Kept as opt-in machinery in `fastbt`
(`fund_metric=` + `strat` keys `funding_block_long/short`, `funding_trend_gate`,
`funding_short_boost`/`funding_long_boost` + thresholds; all default None ⇒ engine
behavior unchanged, regression-verified: gate-on with no thresholds == baseline 227.6×
to the digit). Repro: `opt/eda_funding*.py` (raw conditional forward returns) and
`opt/probe_funding.py` (walk-forward TRAIN/TEST/chrono).

**EDA (causal EWM-30 of per-bar funding, all events ≤ bar):** the raw funding→forward-
return effect is real but **trend-confounded**. High funding is *fine* in an uptrend
(+0.4…+2.0%/30bar) and only bearish in a downtrend (−0.3…−1.5%/30bar); the strongest
cell is the bottom tail (very low/negative funding = crowded shorts/capitulation,
+2.95%/30bar, ~57-59% up, robust in both trends). A naive "fade high funding" rule would
wrongly kill 2021-bull longs.

**Strategy integration (2 bps + funding, select on TRAIN half-years, report held-out
TEST + yearly chrono):** four integrations, none survives the discipline:
- **Block LONG (high funding + downtrend)** — a **no-op**: trades unchanged (2427) at every
  threshold. The trend-following strategy essentially never takes crowded-knife longs in a
  downtrend, so the flagged-bad longs aren't trades it makes anyway.
- **Block LONG (no trend gate)** — *hurts* (227×→126×): removes profitable bull-market longs.
- **Block SHORT (low funding)** — ~no-op (≤5 trades).
- **SHORT-boost (ease short thresholds, high funding + downtrend)** — TRAIN/FULL rise
  (30.6→38, 227×→286×) but **held-out TEST is flat** (7.5→7.6) and the *entire* gain is
  ~17 lucky shorts in **2021's** sharp pullbacks (766%→982%); **2022, the real bear, is
  unchanged** (funding went negative there — "crowded longs" rarely triggered). In-sample
  artifact.
- **LONG-boost (ease long thresholds, very low funding)** — marginal and inconsistent:
  gentle settings nudge TEST +5-8% (within noise), aggressive ones degrade TRAIN and flip
  a fold negative (worst-fold −15%). No setting improves TRAIN and TEST together with all
  folds green.

**Conclusion:** funding's predictive signal barely intersects the strategy's actual
entries, and every apparent win is in-sample-concentrated → **config unchanged**. Funding
stays modeled as a realistic **cost** (Round 4), not a signal. Don't re-pitch funding-as-
signal without a materially different mechanism (e.g. a positioning feature inside the
composite score searched under strict held-out discipline — high overfit surface).

## Round 8 — single local LLM gate: measured, REJECTED

Added an optional `marginal_gate` callback to `fastbt.simulate` and a resumable runner in
`opt/llm_gate_pilot.py`. Default/accept-all behavior is digit-identical to the existing
auto-trade backtest. Unlike the old three-model consensus plan, the test queries exactly
one local model: **`qwen3.6:35b-a3b-q8_0`** through Ollama. Prompts use the canonical
scoring/indicator report frozen at the bar, but omit symbol/date and independently rebase
each timeframe to close=100. This reduces the risk that a post-period model recognizes an
exact historical BTC price and recalls the future. Responses are cached per case under
`reports/` so slow batches resume safely.

Method: 2 bps slippage + funding + liquidation, interleaved TRAIN/TEST half-years. The
runner first queried all 967 actual baseline marginal-entry opportunities. Because rejected
entries change slot/cooldown state and expose opportunities absent from the baseline path,
it then replayed and queried to **fixed-point closure** (77 new cases on pass 1, 5 on pass 2,
none on pass 3): **1,049 total responses, 0 failures**, mean latency 1.8s. The model never
flipped direction: it echoed the deterministic side 748 times (71.3%) and returned WAIT 301
times (28.7%). The converged path actually encountered 1,008 of those queried setups: 719
accepted and 289 rejected (the other 41 were superseded as earlier decisions changed state).

| Split | Auto-trade baseline | Full LLM gate | Growth ratio | MaxDD baseline→gate |
|---|---:|---:|---:|---:|
| TRAIN | 30.57× | 21.48× | 0.7029 | 21.4%→20.9% |
| held-out TEST | 7.51× | 6.72× | 0.8957 | 21.7%→23.6% |
| ALL half-years | **229.51×** | **144.50×** | **0.6296** | **21.7%→23.6%** |

The gate hurts both TRAIN and held-out TEST, loses 37% of full-period compound growth,
worsens overall drawdown, and cuts the worst fold from +17.4% to +7.2%. A few folds improve
(notably 2022H1), but the effect is inconsistent and overwhelmed elsewhere. **Reject the LLM
gate; keep auto-trading MARGINAL signals.** No engine, scheduler, live config, or strategy
defaults changed. Reproduce entirely from the response cache (or resume an interrupted run)
with:

```bash
PYTHONPATH=. /tmp/tmlvenv/bin/python opt/llm_gate_pilot.py --sample-size 967 \
  --model qwen3.6:35b-a3b-q8_0
```

## Repro

```bash
PYTHONPATH=. python opt/driver.py            # baseline eval over folds
PYTHONPATH=. python opt/eda_funding.py       # funding predictive-edge EDA (Round 7)
PYTHONPATH=. python opt/probe_funding.py     # funding-signal walk-forward probe (Round 7)
PYTHONPATH=. python opt/search_wf.py 5000 7  # walk-forward search
PYTHONPATH=. python opt/finalize.py 0        # validation battery on a candidate
```

Caveats: single asset (BTC-perp), 4.4y of data, backtest treats MARGINAL signals as
trades (no LLM in the loop). Funding IS modeled since Round 4 (Binance series as a
Bitget proxy). Paper-trade before real money.
