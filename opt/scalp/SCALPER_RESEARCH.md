# Scalper Research — 15m/5m second product (2026-07-19 overnight session)

**Goal (Marc):** an honest, from-scratch attempt at a 5–15 min "scraper" product inside this
project — not a knob-swap of the 4h strategy. Maximize profit at decent maxDD, do not tune a
lucky one-shot, do not overfit. The live 4h bot is untouched.

**Verdict up front:** a real, robust, fee-surviving edge exists at **15m** — but it is a
**volatility-burst breakout** edge (Donchian break + ATR-expansion filter), *not* mean
reversion, and *not* the house composite scorer. It holds for hours (median ~8–10h), trades
~3×/week/asset, and earns a modest **~9–11% CAGR at ~8% maxDD** (portfolio, 0.5%/trade loss
budget; scales ~linearly with the loss budget). **Pure 5m scalping is structurally dead at
retail fees** — the same signal is profitable gross and fees eat 72% of it. **BTC's edge is
gone in 2024+** (negative in TEST *and* holdout); ETH/SOL carry the product.

---

## 1. Why the earlier 5m attempt failed (fee reality)

`fee_reality.py` — the natural TP/SL scale is the ATR; fees are fixed per notional:

| cadence | median ATR% | maker+taker+slip vs 1.5×ATR TP |
|---|---|---|
| 5m | 0.189% | **35% of the move** |
| 15m | 0.353% | 19% |
| 1h | 0.747% | 9% |
| 4h | 1.551% | 4% |

At 5m, realistic execution burns a third to half of every average-sized move before edge is
counted. No entry signal survives that at scale. This is structural, not a tuning failure.
(Confirmed empirically: the full 5m grid's best config is 1.016×/fold with fees/gross = 72%,
and its best windows are n=288 bars — the strategy trying to become a 15m/1h system.)

## 2. Method (pre-committed protocol)

- Data: Binance USDT-perp 5m/1h (BTC, ETH, SOL) 2020-10 → 2026-07-17, downloaded to
  `history/futures/`; 15m aggregated from 5m. Funding settlements included (Binance history).
- Engine: `opt/scalp/engine.py` — dedicated vectorized backtester (~2-7ms/run):
  decision at bar close acts next bar; maker post-only with **open-cancel semantics**
  (matches the live bot's observed MAKER_POST_ONLY_CANCELLED); **5m sub-bar exit replay**
  inside 15m bars (Marc's suggestion — same pattern as 4h/1h in the main product);
  adverse-first (SL-before-TP) within each sub-bar; SL exits taker+slip; isolated-margin
  liquidation cap; loss-targeted sizing (notional = balance × loss_budget / SL_distance).
  Smoke-tested: no-lookahead guard, fee-drag calibration on random entries, sub-bar vs
  adverse-first ordering, no-signal identity (`smoke.py`).
- Folds: TRAIN 2021→2023 (6 half-year folds, ALL selection), TEST 2024→2025-05 (report
  only), HOLDOUT 2025-06→2026-07-17 (opened once, at the end, on the single chosen config).
- Gates (fixed before results): ≥200 trades, ≥5/6 TRAIN folds positive, worst-fold DD ≤25%.
  Finalists ranked by **worst-asset** TRAIN geo (kills single-asset path fits).

## 3. What was searched (≈13,000 backtests)

| family | space | result |
|---|---|---|
| BB z-score reversion | n, z_in, chop-gate (ER), trend gate, passive-fade offsets, TTL, mean-touch exits | **dead** — best ≈ breakeven, fees 75–80% of gross |
| RSI(2–4) reversion | thresholds, gates | worst family, dropped round 2 |
| VWAP-deviation reversion | n, dev_in (ATR units), gates | dead (only degenerate ~5-trade configs ≥1.0) |
| **Donchian breakout + vol-expansion** | n 8–96, ATR-ratio filter 1.2–2.2, SL 1.5–2.5×ATR, TP 3–6×ATR, trailing, maker/taker | **only survivor — entire top list** |
| House scorer @15m retuned (`control_fastbt15m.py`) | threshold ×1–3, ATR-SL ×0.75–1 | best 1.023×/fold, 3/6 folds, 28–36% DD — **not competitive**; different cadence needs a different core strategy |

Dose-response of the vol filter (BTC 15m, mean geo/fold): none → 0.76–0.84 (loses),
1.2 → ~0.97, 1.5 → 1.00–1.06 (works). The filter is causal to the edge, and the edge exists
in *expansion* regimes — the opposite of the MR intuition. Trend gates hurt (breakouts
self-select direction). Cross-asset: the SAME region survives on ETH and SOL with zero
per-asset re-tuning.

## 4. The chosen config

```
signal : 15m Donchian break of prior-96-bar extreme (24h channel),
         gated by ATR14 / SMA32(ATR14) >= 1.3   (vol expansion)
entry  : post-only maker limit at decision close, good for 1 bar,
         open-cancel semantics (≈27% of orders cancel/miss)
exits  : SL 2.0×ATR (taker), TP 6.0×ATR (maker), cooldown 2 bars after stop
sizing : loss budget 0.5%/trade → notional = bal×0.005/SL_dist, cap 3× bal
```

| window | BTC | ETH | SOL | portfolio (⅓ each) |
|---|---|---|---|---|
| TRAIN 2021-23 (selected) | 1.74× | 1.35× | 1.49× | 1.53×, **15.2% CAGR, 7.7% DD** |
| TEST 2024-25H1 | 0.95× | 1.24× | 1.20× | **9.1% CAGR, 7.5% DD** |
| HOLDOUT 2025-06→2026-07 | 0.81× | 1.29× | 1.26× | **10.6% CAGR, 8.0% DD** |

TEST ≈ HOLDOUT at the portfolio level — the number to believe is **~10%/yr at ~8% maxDD per
0.5% loss budget**. Holdout monthly breakdown: 8/14 months positive, best month +7.8%
(Oct-2025 vol spike), no lucky-month dominance — but the FOUR most recent months
(2026-04→07) are all mildly negative (≈−6% cumulative): the edge's latest stretch is soft,
inside the expected DD envelope but worth watching before committing capital. Ladder (TRAIN): 1% budget ≈ 2× CAGR at ~2× DD (~15–23%/asset); 2% ≈
28–38% DD. Trades ≈ 3/week/asset (~9/week portfolio), median hold ~8–10h — a burst-swing
scalper, not a minute-scale scraper; that is what survives fees honestly.

## 5. Robustness (all PASS)

- **Jitter ("value shaking"):** 40 draws, every knob ±15% — 92% keep a positive worst-asset
  TRAIN edge (gate 80%); BTC/ETH never negative; plateau, not a spike (`jitter.py`).
- **Queue penetration:** 1/2/5bps → BTC TRAIN 1.097 → 1.072/1.066/1.054. Still profitable
  under harsh queue assumptions. **Taker-entry flip: 1.054×** — the edge does NOT depend on
  the maker fill fantasy (unlike the 4h product, where maker-vs-taker is still being measured
  live). fees×1.5 → 1.080; slip×2 → 1.090.
- **Cross-asset invariance:** one config, zero per-asset tuning, survives on all 3 TRAIN and
  on ETH/SOL in TEST + holdout.

## 6. The BTC problem (honest reading)

BTC: TRAIN 1.74× → TEST 0.95× → holdout 0.81× (win rate 24%). The burst-momentum edge on BTC
died in the 2024+ regime. Independently, the 4h product found BTC its weakest OOS asset
(1.65× vs ETH 6.52×/SOL 6.21×). Two unrelated searches, same conclusion.

Per the pre-committed protocol the 3-asset portfolio is the reported product (it still makes
~10%/yr because ETH/SOL dominate). **Dropping BTC would be a post-hoc, TEST-informed decision**
— defensible given two independent evidence lines, but it must be owned as such and validated
by the live track record, not by more backtests.

## 7. Known model optimism / caveats

- Maker fills modeled fill-on-touch after open-cancel (~73% fill rate observed in sim); real
  queues are worse — but the taker stress (still profitable) bounds this risk.
- TP fills at exact price with maker fee; live TP may execute as taker (same known divergence
  as the 4h product). SL slippage 2bps may be thin in fast tapes.
- 2 liquidation events on SOL holdout at the default 25× margin-efficiency cap — a live
  config should cap effective leverage ~10× (SL 2×ATR never needs more).
- Funding modeled from Binance history; Bitget funding differs slightly.
- Returns are small enough that live frictions matter proportionally more than for the 4h
  product. A $193 account earning 10%/yr is ~$20 — this product only makes sense at larger
  balance or as a diversifier alongside the 4h bot (their daily correlations were 0.17-0.45).

## 8. If Marc wants to go further (next steps, in order)

1. Decide BTC in/out (see §6) and the loss budget (0.5% vs 1%).
2. Live-ify: the scalp engine is research-only; a live scheduler needs a 15m analysis loop
   (the orchestrator already polls minutely), Bitget 15m candles (available), and the same
   preset-bracket order flow the 4h bot uses. Non-trivial but all building blocks exist.
3. Paper/micro-live it in parallel with the 4h bot (separate sub-account or strict
   portfolio-margin partition) and let the live record decide, like the 4h product.
4. Do NOT re-grid on TEST/holdout. The next honest data is live data.

## Files

- `opt/scalp/engine.py` — vectorized scalp backtester (sub-bars, post-only, funding, liq)
- `opt/scalp/strategies.py` — signal families; `opt/scalp/grid.py` — grid runner + protocol
- `opt/scalp/finalists.py` — worst-asset selection, TEST, stress ladders
- `opt/scalp/jitter.py`, `portfolio_view.py`, `control_fastbt15m.py`, `fee_reality.py`,
  `smoke.py`, `download_data.py`
- `opt/scalp/results/*.json|csv` — all artifacts (grids, finalists, jitter, portfolio,
  holdout, control)
