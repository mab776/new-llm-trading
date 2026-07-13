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

The non-thinking gate hurts both TRAIN and held-out TEST, loses 37% of full-period compound
growth, worsens overall drawdown, and cuts the worst fold from +17.4% to +7.2%. A few folds
improve (notably 2022H1), but the effect is inconsistent and overwhelmed elsewhere. This
**rejects non-thinking mode only**; see the correction below. No engine, scheduler, live
config, or strategy defaults changed. Reproduce entirely from the response cache (or resume
an interrupted run) with:

```bash
PYTHONPATH=. /tmp/tmlvenv/bin/python opt/llm_gate_pilot.py --sample-size 967 \
  --model qwen3.6:35b-a3b-q8_0
```

### Round 8b correction — thinking-enabled pilot is mixed (continue before verdict)

The initial full run used Ollama `think: false`, which materially handicaps this reasoning
model. A fresh leakage-blinded pilot enabled native thinking with `num_predict: 8192`, a
separate settings-keyed cache, temperature 0, and the same deterministic seed. It sampled
36 baseline entries—four per half-year fold. All 36 returned valid JSON and non-empty
thinking traces (median ~984 words); mean latency rose from 1.8s to 14.2s. Thinking changed
12/36 decisions versus non-thinking on the identical cases, mostly becoming more selective
(9 LONG→WAIT, 2 SHORT→WAIT, 1 WAIT→LONG). It accepted 17/36 versus 27/36 non-thinking.

| Split | Auto-trade baseline | Sparse thinking gate | Growth ratio | MaxDD baseline→gate |
|---|---:|---:|---:|---:|
| TRAIN | 30.57× | 28.21× | 0.9228 | 21.4%→21.4% |
| held-out TEST | 7.51× | 7.64× | **1.0181** | **21.7%→20.1%** |
| ALL half-years | 229.51× | 215.63× | 0.9395 | 21.7%→21.4% |

This is genuinely mixed: the small held-out slice improves slightly, while TRAIN and total
compound decline. With only four interventions sampled per fold, neither the +1.8% TEST
gain nor the −6.1% all-fold loss is decisive. The earlier blanket rejection therefore does
not apply to thinking mode. Do not change production behavior yet; expand the thinking run
under the same fixed prompt/settings before reaching a verdict. Reproduce/resume with:

```bash
PYTHONPATH=. /tmp/tmlvenv/bin/python opt/llm_gate_pilot.py --sample-size 36 \
  --model qwen3.6:35b-a3b-q8_0 --think --num-predict 8192 \
  --cache reports/llm_gate_qwen36_35b_q8_think8k.jsonl
```

### Round 8c — thinking gate expanded: REJECTED, item CLOSED

The Round 8b thinking pilot was expanded (operator-run) to settle the mixed n=36 result under
the same frozen prompt/settings (`think:true`, `num_predict 8192`, temp 0, blinded rebased
prompts). **Outcome: the LLM gate was worse than the deterministic auto-trade baseline across
all splits** — the small held-out TEST bump from the sparse pilot did not survive a larger
sample. Signal-only trading wins outright: adding the model as a MARGINAL-entry gate throws
away edge (it mostly turns LONG/SHORT into WAIT, skipping profitable entries) and buys nothing
robust in return, in either thinking or non-thinking mode.

**Verdict: backlog item #2 (LLM gate) is DONE / REJECTED — do not retry** without a materially
different mechanism (e.g. not a per-entry accept/reject gate). Production config is unchanged;
no engine/scheduler/strategy defaults touched. The opt-in `marginal_gate` machinery and the
`opt/llm_gate_pilot.py` runner + response caches (`reports/llm_gate_qwen36_35b_q8*.jsonl`) are
kept for reference only. This supersedes the "continue before verdict" note in Round 8b.

## Round 9 — maker-entry modeling: EV-positive SCREEN (fastbt only, not yet shipped)

Backlog #4. Entries are currently market/taker (0.06% + slip). Alternative: rest a **limit at
the decision bar's close** and fill it only if the **next bar trades back to it** (LONG:
next-bar low ≤ limit; SHORT: next-bar high ≥ limit), paying **maker 0.02% with no entry slip**,
cancelling unfilled orders (good-for-one-bar → missed trades when price runs). Implemented as an
opt-in `strat["entry_mode"]` in `fastbt.simulate` (default `"taker"` reproduces the engine
bit-for-bit; the taker baseline still prints BTC 227.6× / ETH 1014.6×). Exit fees/slip unchanged
(SL=taker+slip, TP=maker). Screen: `opt/maker_entry.py`.

Apples-to-apples (2bps slip + funding + liquidation, only entry model differs):

| exit mode | asset | taker FULL / TEST | maker FULL / TEST | worst · DD · missed fills |
|---|---|---|---|---|
| primary | BTC | 227.6× / +65.5%/f | **336.0× / +73.9%/f** | +32.2% · 24.0% · 7.9% |
| primary | ETH | 1014.6× / +81.4%/f | **2954.9× / +113.5%/f** | +17.9% · 22.4% · 7.3% |
| sub (honest) | BTC | 12.25× / +16.4%/f | **32.10× / +29.4%/f** | +6.7% · 28.5% |
| sub (honest) | ETH | 22.46× / +13.4%/f | **101.08× / +39.8%/f** | +1.9% · 24.1% |

Maker wins on **both assets, held-out TEST, and both exit granularities** — the better entry
price + fee/slip saving more than pays for the ~7–8% of taker fills it misses. Rare unambiguous
signal. **NOT SHIPPED:** the fastbt fill is booked once per 4h bar *after* that bar's exit step,
so a freshly filled trade gets a one-bar exit delay (sub mode mitigates but doesn't remove it —
the fill isn't placed at its precise 1h sub-bar). Per methodology rule #4 the honest intrabar
sequencing must be validated in the **engine port** (real pending-order lifecycle, same-bar exit
after fill) before the magnitude is trusted or anything goes live — where maker also carries
non-fill / queue-position risk not modelled here. Config/engine/scheduler untouched; 264 tests pass.
Repro: `PYTHONPATH=. /tmp/tmlvenv/bin/python -m opt.maker_entry`.

## Round 10 — third-asset transfer: SOL (unchanged BTC config stays green everywhere)

Backlog #6. Ran the **unchanged BTC-tuned `config.json`** on SOL via `driver.setup(symbol=
"SOL/USDT:USDT")` (fetch is automatic; `config-sol.json` added for engine parity, symbol-only
diff like `config-eth.json`). Data is genuine — SOL 4h begins 2021-07-23 (when SOL perp actually
launched, so the 2021 fold is H2-only), 100% sub-bar coverage, no Bitget placeholder junk.

2bps slip + funding + liquidation, every fold green on BOTH entry models and BOTH exit modes:

| exit mode | entry | TRAIN | TEST | FULL | worst · DD | per-year (ret, dd) |
|---|---|---|---|---|---|---|
| primary | taker | +114%/f | +335%/f | 15100× | +173% · 18.5% | 21:+173 22:+936 23:+3027 24:+445 25:+213 |
| primary | maker | +130%/f | +391%/f | 33551× | +217% · 20.5% | 21:+216 22:+937 23:+5029 24:+412 25:+289 |
| sub (honest) | taker | — | +94%/f | 56.8× | +45% · 24.9% | 21:+48 22:+237 23:+354 24:+45 25:+74 |
| sub (honest) | maker | — | +145%/f | 244× | +59% · 29.0% | 21:+78 22:+260 23:+994 24:+59 25:+119 |

**Robustness signal, not a return forecast.** The config is now green on every yearly fold of
**three** assets (BTC, ETH, SOL) with no per-asset retuning — the "near-unfalsifiable" bar from
the backlog. The absolute multiples are inflated by SOL's cycle-defining volatility × 25× leverage
and a flat-2bps slip that ignores book size; treat the *green-everywhere* fact, not the number, as
the finding. Maker beats taker on SOL too, consistent with Round 9. Config/engine untouched (SOL
config is additive). Repro: `PYTHONPATH=. /tmp/tmlvenv/bin/python -m opt.driver` after
`setup(symbol="SOL/USDT:USDT")`, or point any run script at `config-sol.json`.

## Round 11 — maker entry shipped with honest fill-bar exits

Finished Round 9's release gate. `trading.entry_mode` now drives the same good-for-one-primary-
bar lifecycle in fastbt, `BacktestEngine`, and live scheduling. A pending limit counts as a slot;
on the next bar it fills only if touched, and the new position is checked for SL before TP on the
**same fill bar**. Live uses a post-only order with preset SL+TP, persists pending context, queries
order state, handles fill/cancel races, and cancels at the next UTC-aligned primary close. The
exchange client gained order-detail/cancel operations. `config.json`, `config-eth.json`, and
`config-sol.json` now select `entry_mode: "maker"`.

Strict 1h sub-bar replay, 2bps slip + funding + liquidation:

| asset | taker FULL / TEST geo | honest maker FULL / TEST geo | maker worst fold · maxDD |
|---|---:|---:|---:|
| BTC | 12.25× / +16.4% | **26.13× / +27.8%** | +27.3% · 25.6% |
| ETH | 22.46× / +13.4% | **56.35× / +27.3%** | +11.5% · 29.9% |
| SOL | 56.79× / +94.3% | **138.76× / +119.1%** | +68.6% · 22.4% |

Maker still wins on all three assets, TRAIN, held-out TEST, and every chronological fold after
removing the delay optimism. Full-engine↔fastbt 2024 maker parity matched return (438.48%), final
balance, 607 trades, win rate, PF, DD, and Sharpe. Queue-position/non-fill realism remains a paper-
trading concern; a touched OHLC limit is not proof of a real exchange fill.

## Round 12 — shared BTC+ETH+SOL portfolio harness

Added `opt/multi_asset.py` and `opt/multi_portfolio.py`. Primary streams are timestamp-interleaved
into one `Portfolio`; balance, peak equity, and DD throttle are shared, while entry slots, pending
orders, cooldowns, loss penalties, targets, funding, and trailing state remain per symbol. Trades
carry a symbol and snapshots mark each open position using its own current price. Same-timestamp
capital allocation is deterministic (sorted symbol order).

Honest sub-bar yearly folds, 2bps + funding + liquidation:

With the Round 14 scoring points, the current maker configuration compounds **235,389×** across
the independently-reset yearly folds, with worst fold +218%, maxDD 37.8%, and 8,380 trades.
(Before Round 14, the same shared maker harness produced 20,879×, worst +158%, maxDD 35.9%.)

The multiple is not a forecast: three symbols × three slots × 25× creates much more aggregate
exposure than one instance. The useful result is green-every-year shared compounding **and** the
warning that shared maxDD (~38%) exceeds the single-asset runs. Add a global exposure/risk cap
before considering this layout for live trading.

## Round 13 — annual walk-forward retuning: promising, not shipped

Added `opt/walk_forward_retune.py`: for target year N, search only N-2..N-1 and trade N. A first
60-candidate/window run used maker entry, honest sub exits, funding, and 2bps slip. After applying
Round 14's static scoring points, chained unseen 2023→2025H1 growth was **13.08× retuned vs 9.63×
static** (1.36 ratio): 2023 +381% vs +220%, 2024 +151% vs +118%, but 2025H1 +8% vs +38%.
This is encouraging but unstable and only three deployment windows;
no production parameters changed. Expand seeds/search counts and include parameter-turnover costs
before adopting a cadence.

## Round 14 — scoring internals parameterized; constrained winner SHIPPED

All canonical hand-tuned awards/penalties now live in
`openwebui_filter.DEFAULT_SCORING_POINTS`; partial `scoring.points` overrides flow through the
typed scorer, routing/live analysis, full engine, fastbt, and shared portfolio without duplicating
logic. Added `opt/search_scoring_points.py`, constrained to nine interpretable values and selecting
on TRAIN only. The 120-candidate pilot overfit, so the search was expanded to 500 before a verdict.

The 500-candidate TRAIN winner improved BTC TRAIN geo +57.6%→+89.8%, held-out TEST
**+27.8%→+29.5%**, and chronological 2024-25 **2.80×→3.01×**. More importantly, without any
asset-specific selection it transferred to ETH (TEST +27.3%→+31.1%, FULL 56.4×→157.5×, worst
+11.5%→+19.1%) and SOL (TEST +119.1%→+184.6%, FULL 138.8×→777.0×, worst +68.6%→+103.5%).
That cross-asset falsification pass clears the bar; the nine overrides are now in all three configs.
Current honest maker/sub BTC is **70.28×**, worst +38.0%, maxDD 25.1% (vs 26.13× before points).
Final 2024 full-engine↔fastbt parity with maker + point overrides is digit-equal: +532.52%,
$632.52 final balance, 660 trades, 80.4545% win rate, PF 1.69, maxDD 24.3%, Sharpe 2.77.

## Round 15 — anti-martingale sizing: rejected as portfolio-DD control

Added an experimental causal, per-asset signed outcome streak to `fastbt` and the shared-portfolio
harness. Completed wins raise the next trade's risk and completed losses lower it, with configurable
step/min/max bounds; a zero step is exactly the previous behavior. Searched 96 bounded variants on
the interleaved shared BTC+ETH+SOL TRAIN half-years only, using maker entry, honest 1h sub exits,
funding, liquidation, and 2bps market slippage.

No candidate met the required 25% TRAIN maxDD. The minimum-DD TRAIN candidate (step 0.05, bounds
0.70-1.10) improved TRAIN geo return +394.6%→+467.0% and maxDD 37.8%→31.8%, but held-out TEST
maxDD worsened 35.3%→36.0% (despite geo return improving +251.6%→+278.8%). Continuous full-period
maxDD improved only 37.8%→36.0%; chronological 2024-25H1 improved 36.3%→34.6%. It therefore fails
both the hard ≤25% acceptance rule and the cross-split drawdown robustness test. Nothing was ported
to production config, the full engine, or scheduler. The harness and
`opt/anti_martingale_results.json` are retained for audit; do not retry simple closed-trade streak
sizing as the solution to shared exposure without a materially different mechanism.

## Round 16 — portfolio-wide exposure controls + anti-martingale: SHIPPED

Added shared causal sizing math in `llm_trading_bot/exposure.py`, exchange-wide live queries for
account equity, open positions, resting entry orders, and closed-position net profit, plus matching
full-engine/fastbt/shared-portfolio enforcement. Caps are ex ante only: a new order is reduced to
remaining capacity or skipped; normal exits are untouched and no drawdown kill switch or synthetic
threshold fill is used.

The first 330-candidate TRAIN-only grid winner (four global slots, 7% margin / 1.75× notional) had
21.4% TRAIN maxDD but failed held-out TEST at 30.5%. Conservative fallbacks at two slots and 5–6%
margin also failed TEST. A narrow sensitivity sweep found a stable boundary: 4.4% margin / 1.10×
notional with Round 15's anti-martingale overlay (step 0.05, bounds 0.70–1.10). The next cap was
materially above the target; 4.4% realizes **25.03%**, accepted under the user-approved
"approximately 25%" criterion rather than treating rounding noise as a cliff.

Honest maker + 1h sub exits + funding + liquidation + 2bps market slip:

- TRAIN: 308.61× compound, worst +100.7%, maxDD 17.41%.
- Held-out TEST: 6.48× compound, worst +31.0%, maxDD 25.03%.
- Chronological 2024–25H1: 5.98×, maxDD 16.79%.
- Annual-reset folds: 1,932.51× compound, every year green, maxDD 25.03%.
- Continuous 2021–25H1 shared portfolio: **1,905.59×**, maxDD **25.03%**.
- Standalone continuous: BTC 30.08×/21.30% DD; ETH 92.46×/23.58%; SOL 492.23×/20.14%.

The full 2024 engine↔fastbt parity check is digit-equal after the port: +226.20%, 562 trades,
79.3594% win rate, PF 1.57, maxDD 22.03%, Sharpe 2.43. Independent live schedulers query
exchange-wide exposure, but their check/place sequence is not atomic; shared deployment should run
through one orchestrator or equivalent cross-stack serialization to eliminate simultaneous-order
races.

## Round 17 — uncapped anti-martingale aggressive profile: SHIPPED SEPARATELY

At the user's explicit risk/return preference, the highest-return Round 15 policy is now available
as a separate aggressive profile; the Round 16 capped configs remain the defaults. The aggressive
BTC/ETH/SOL files use config inheritance, retain maker entry, mandatory SL+TP, the bounded
0.70×–1.10× anti-martingale, the 25% DD throttle, fees/funding/liquidation modeling, and disable
only the portfolio margin/notional ceilings. Their high `max_position_usd` ceiling prevents the
live sizing guard from silently turning off equity compounding; all three still inherit
`bitget.testnet: true`.

Honest maker + 1h sub exits + funding + liquidation + 2bps market slip:

- TRAIN: 5,859.08× compound, worst +182.0%, maxDD 31.79%.
- Held-out TEST: 205.97× compound, worst +117.8%, maxDD 35.95%.
- Chronological 2024–25H1: 14.53×, maxDD 34.65%.
- Annual-reset folds: 901,910.30×, every year green, maxDD 35.95%.
- Continuous 2021–25H1: **920,165.82×**, maxDD **35.95%**, 8,388 trades.

A new non-mutating 4h-close equity sampler confirmed 36.15% mark-to-market maxDD, close to the
engine headline rather than hiding a much deeper collapse. It is not a weekly 36% event, but it is
not one-off: the top three peak-to-trough episodes were 36.15%, 35.02%, and 33.72%. Drawdown was at
least 33% for 2.09% of samples (11/231 weeks) and at least 30% for 9.79% (47/231 weeks). The top
three complete peak-to-recovery episodes lasted about 112, 79, and 175 days. Full machine-readable
results are in `opt/aggressive_profile_results.json`.

These multiples are path-dependent backtest compounding, not forecasts. Touched maker limits do
not model queue priority, and live slippage, outages, correlation, execution races, or a new regime
can produce drawdown materially above the corrected ~34% history. Use the explicit aggressive filenames so the uncapped
policy cannot be mistaken for the standard profile.

## Round 18 — sub-bar cadence correction + maker queue sensitivity

An audit of `exit_granularity="sub"` found that its 1h replay was incorrectly ratcheting the
trailing stop after every sub-bar. That contradicted the strategy's non-negotiable cadence: replay
1h bars for exit ordering, keep the stop fixed intrabar, then ratchet once using the completed 4h
bar's favorable extreme. `fastbt` and the shared harness now do exactly that, guarded by focused
single/shared cadence tests. Round 16/17's sub-bar numbers are therefore superseded (their configs
are unchanged):

- Standard continuous: **292,212.44×**, 19.95% reported maxDD, 20.67% independent 4h MTM maxDD;
  held-out TEST 104.99×. Standalone BTC/ETH/SOL: 301.18× / 2,436.13× / 66,125.23×.
- Aggressive continuous: **5,748,971,553,896.69×**, 34.28% reported maxDD, 34.11% 4h MTM maxDD;
  held-out TEST 686,340.87×. This enormous path-dependent multiple is emphatically not a forecast.
- A fresh TRAIN-only exposure search chose 12% margin/3.0× notional/3 slots, but it failed held-out
  maxDD at 28.6%. The shipped standard 4.4%/1.10× caps remain unchanged and validate at 19–21% DD.

Maker queue stress is deterministic and reproducible: require 0–10bps penetration beyond the
limit and/or accept only 70–95% of eligible touched orders using an order-identity hash. Across five
seeds, 70% fills alone retained a median 79.3% of baseline log growth. The harsh combined 5bps +
70% case retained 65.6%, produced 231.82 million× median continuous growth, kept every annual fold
green, and reached 38.15% worst 4h MTM DD. This establishes broad historical execution tolerance;
it does not replace paper measurement of actual fill rate. Full artifacts:
`opt/cadence_correction_results.json`, `opt/queue_fill_sensitivity_results.json`, and
`opt/portfolio_exposure_cadence_results.json`.

## Round 19 — shared live orchestration/state parity: SHIPPED

Added one-process `SharedTradingOrchestrator` plus atomic `SharedLiveState`. BTC/ETH/SOL cycles run
serially and every account-wide exposure check → size → place sequence shares one re-entrant lock,
so simultaneous symbol signals cannot independently consume the same capacity. A non-blocking
process lock rejects a second orchestrator in the same deployment state directory.

The realized account-balance peak, maker pending orders, and trailing context/current stop are
atomically persisted and restored. Exchange equity minus open unrealized PnL reconstructs the
realized balance used by the backtest DD throttle. Shared pending reconciliation and opposite-order
cancellation are symbol-local. The legacy pending file migrates once without resurrecting stale
orders. Shipped configs now set marginal execution to deterministic, matching both fast/full
backtests and Round 8c's signal-only winner; three-model consensus remains an explicit opt-in mode.
Start later (only with explicit authorization) via `main --mode live --shared-configs ...`; no
live/testnet process was started in this round.

## Round 20 — expanded walk-forward retuning: RETURN-ROBUST, NOT SHIPPED

Expanded the Round 13 pilot with corrected 4h-close trailing cadence: five seeds at 60 and 300
candidates/window, three seeds at 1,000, and explicit normalized parameter-turnover penalties.
Every unpenalized run beat static across the chained unseen 2023/2024/2025H1 windows:

- 60 trials: median 17.93× vs static 10.91×, median ratio 1.644 (range 1.608–1.966).
- 300 trials: median 20.60×, ratio 1.889 (range 1.311–2.287).
- 1,000 trials: median 22.03×, ratio 2.020 (range 1.950–2.358).

The return effect is robust, including the previously weak 2025H1 window (wins in 80% / 60% /
100% of the 60/300/1,000-trial runs). Parameter selection is not stable: every seed chose a unique
winner in every deployment window. A 15-point turnover penalty changed nothing. Penalties of 200
and 500 reduced median turnover only modestly (about 0.71→0.58 at 500); the 500-point study made
one seed revert to static while the other four still chose different parameter sets. Automatic
annual search/reload and a complete pre-deployment training window are also absent.

Verdict: the adaptive *process* has convincing historical return evidence, but specific deployable
parameters do not converge and operational retuning would confound the first execution-validation
paper run. Keep static production configs for paper trading; retain this as the leading post-paper
research item. Artifacts: `opt/walk_forward_robustness_results.json` and
`opt/walk_forward_turnover_results.json`.

## Round 21 — regime-switching parameters: REJECTED

Added research-only causal overlays for regime-specific entry thresholds, leverage, and trailing
activation/callback distances. Five independent 60-candidate searches selected only on shared
aggressive TRAIN half-years tested bounded trending/weak/ranging/volatile variants. The unchanged
static strategy ranked first in **all five searches**; consequently held-out TEST, chronological,
continuous-improvement, and seed-stability acceptance checks all failed. Nothing was ported to
engine/config/scheduler. The empty-overlay path remains exact and the research result is retained
in `opt/regime_search_results.json`; do not retry this parameterization without a materially new
regime mechanism.

## Repro

```bash
PYTHONPATH=. python opt/driver.py            # baseline eval over folds
PYTHONPATH=. python opt/eda_funding.py       # funding predictive-edge EDA (Round 7)
PYTHONPATH=. python opt/probe_funding.py     # funding-signal walk-forward probe (Round 7)
PYTHONPATH=. python opt/search_wf.py 5000 7  # walk-forward search
PYTHONPATH=. python -m opt.maker_entry --exit-granularity sub --include-sol
PYTHONPATH=. python -m opt.multi_portfolio --exit-granularity sub
PYTHONPATH=. python -m opt.walk_forward_retune --trials 300
PYTHONPATH=. python -m opt.search_scoring_points --trials 500
PYTHONPATH=. python -m opt.anti_martingale
PYTHONPATH=. python -m opt.portfolio_exposure
PYTHONPATH=. python -m opt.multi_portfolio --profile aggressive --entry-mode maker --exit-granularity sub
PYTHONPATH=. python -m opt.validate_parity --entry-mode maker
PYTHONPATH=. python opt/finalize.py 0        # validation battery on a candidate
```

Caveats: single asset (BTC-perp), 4.4y of data, backtest treats MARGINAL signals as
trades (no LLM in the loop). Funding IS modeled since Round 4 (Binance series as a
Bitget proxy). Paper-trade before real money.
