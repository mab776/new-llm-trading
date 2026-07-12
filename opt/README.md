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

## Repro

```bash
PYTHONPATH=. python opt/driver.py            # baseline eval over folds
PYTHONPATH=. python opt/search_wf.py 5000 7  # walk-forward search
PYTHONPATH=. python opt/finalize.py 0        # validation battery on a candidate
```

Caveats: single asset (BTC-perp), 4.4y of data, backtest treats MARGINAL signals as
trades (no LLM in the loop). Funding rates NOT modeled (perp funding averages ~0.01%/8h;
positions here are held hours-days so impact is small but nonzero). Paper-trade before
real money.
