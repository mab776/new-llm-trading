# Signal Trading Bot

> **🔬 2026-07-20 RESEARCH NIGHT (no strategy change; live config untouched).** Three
> portfolio-architecture probes (all opt-in fastbt/simulate_multi knobs, default-off =
> engine-identical, 398 tests) + new tooling:
> 1. **Reserved per-asset capital — REJECTED** (`opt/probe_reserved.py`, `37b898a`): equal
>    thirds trade ~+70% more but grow less (OOS 4.79× vs 5.88×; 3.89× vs 5.65× @$193+mins);
>    cross-subsidy concentration IS the edge — margin-cap crowding is its price.
> 2. **Cross-asset rotation — REJECTED for growth** (`opt/probe_rotation.py`, `d8b4a24`):
>    TRAIN winner failed the TEST gate (split-disagreement noise). Post-hoc: improves
>    worst-folds — parked as a robustness idea.
> 3. **Conditional cap-overshoot (min-size rescue) — ALL GATES PASSED** (`opt/probe_overshoot.py`,
>    `b5fe4de`): flooring MIN_SIZE_SKIPped strong entries to the exchange minimum beats
>    fail-closed skip on TRAIN+TEST+holdout (5.65→6.01×). Small-account provision
>    (self-retires by ~$2500). NOT deployed — needs scheduler-side code + Marc's go.
> Also: **`opt/live_reconcile.py`** promoted to an official sim-vs-live reconciliation tool
> (`1aebd52`; live −$4.74 ≈ sim −$4.15/−$4.48 over the first 4 live days — no execution
> drift), and **`GOOD_IDEAS.md`** now indexes every shelved-but-positive finding (walk-forward
> retuning ~2×, min-size rescue, scalper, watchlist bands, …).

> **⚡ 2026-07-19 STRATEGY UPDATE (supersedes the numbers below where they differ).**
> Two config changes deployed LIVE (bot restarted, 398 tests, engine==fastbt parity exact):
> 1. **Per-TF alignment weights `{"1h": 0, "1d": 3}`** (commit `49f236e`) — the hardcoded ±5
>    secondary-TF alignment vote was never searched; live/backtest reconciliation of the first
>    paper days exposed it (one near-zero 1h vote flipping ±5 across the −20 exit cliff = the
>    whole early live loss). Independent sweeps: 1h vote = noise, 1d wants 3.
> 2. **Consecutive-loss penalty removed (5→0)** (commit `79a914e`, `opt/probe_penalty.py`) —
>    penalty stacked on thresholds tuned for the old ±10 alignment range; on the new base it is
>    monotone-harmful on TRAIN+TEST, gave no measurable 2022-bear insurance, won 8/9 folds at 0.
>
> All profile configs (aggressive, 1x) inherit both via `_extends`. **New reference numbers**
> (same methods as below): standard continuous **6,050,413×** / 20.7% MTM DD, TEST 320.34×;
> aggressive continuous **~2.48 quadrillion×** / 36.4% MTM DD, TEST 7.52M×. Clean OOS holdout
> (canonical `opt.holdout_oos`): standard **5.90×/14.2% MTM DD** (was 4.88×/17.0%), aggressive
> **37.3×/32.7%** (was 16.8×/35.0%); standalone BTC 1.65× / ETH 6.52× / SOL 6.21× (all improved).
> With real Bitget mins @ $100: **5.15×/12.9% DD**.
> ⚠️ 1× contingency validation numbers (3.69×) predate this — re-run `opt/validate_1x` before use.
> Hat-number audit: all scorer trigger levels proven robust (±15% AST perturbation); regime
> detection = dead code (skip flags off); watchlist williams/stoch bands (below flag threshold).
> Research parked: 1d adx_di overlay (holdout advantage inverted on new base — don't re-pitch).
> Also 2026-07-19: **+$100 deposit** (futures ~$193; granularity tax 20.5%→10.7%).

An automated cryptocurrency trading bot driven purely by deterministic
multi-timeframe technical-analysis scoring (no LLM in the decision loop). An LLM
marginal-gate was tested and rejected — it lost to signal-only execution; the last
commit containing that code is tagged `last-llm-consensus`.

## Quick Start

```bash
pip install -r requirements.txt
cp config.json config.local.json  # Edit with your credentials
python -m llm_trading_bot.main --config config.json
```

## Architecture

```
Market Data (Bitget futures — windowed history + disk cache; also binance/yfinance)
        ↓
  Scoring Engine (indicators + weighted score)
        ↓
  Signal Router
   ├── STRONG  → Execute (deterministic template)
   ├── MARGINAL → Execute (deterministic — counted as a trade in the backtest)
   └── WAIT    → Skip trade
        ↓
  Bitget Execution (futures + mandatory TP/SL + risk-based sizing + trailing stops)
```

### Market data

Default source is **Bitget USDT-perpetual futures** (`data_source.source: "bitget"`,
`exchange_symbol: "BTC/USDT:USDT"`, `market: "futures"`). Bitget's history endpoint is
**200-cap and END-anchored** (it returns the last N candles before `until`), so a naive
`since` fetch silently drops everything but the tail of the range. `bitget_csv.py` handles
this with **explicit windowed pagination** (`until` per ≤200-candle window) and caches
**complete months** to `history/bitget/…` — repeated deep backtests are then instant.
(Note: Bitget *spot* lacks 2h/1s candles; futures has full coverage. `binance` CSV archive
and `yfinance` remain available.)

## Modules

| Module | Purpose |
|--------|---------|
| `scoring` | Typed scoring API, targets, configurable canonical point values |
| `entry` | Shared maker pending-entry fill rule and lifecycle data |
| `data` | OHLCV fetching, caching, 4H aggregation, source routing |
| `bitget_csv` | Bitget windowed history getter + monthly disk cache |
| `binance_csv` | Binance public CSV archive downloader |
| `routing` | Signal classification and routing |
| `openwebui_filter` | Signal library — indicator math + category scoring (source of truth) |
| `exchange` | Bitget futures integration (orders, balance, stop updates) |
| `trailing` | Shared trailing-stop math (backtest + live) |
| `live_state` | Atomic persisted portfolio peak, pending orders, and trailing context |
| `orchestrator` | Serialized multi-symbol live scheduling against one account |
| `backtesting` | Historical replay engine |
| `portfolio` | Fee-aware portfolio simulation |
| `reporting` | Reports and charts |
| `scheduler` | Cron-like scheduling + risk-based sizing + live trailing |

## Safety Rules

- **Never** place an order without stop loss + take profit
- **Never** hardcode API credentials
- Confidence bounded to [5%, 95%]
- Backtesting never peeks at future data
- All PnL accounts for fees on leveraged notional

## Running Tests

```bash
python run_tests.py
# or
pytest tests/ -v
```

## Configuration

All settings live in `config.json`. See `AGENTS.md` for full documentation.

Two shared-portfolio risk profiles are available:

- **Standard (default):** `config.json`, `config-eth.json`, and `config-sol.json`; portfolio-wide
  caps target approximately 25% historical shared max drawdown. With the live execution model
  enforced in the simulators (scale-invariant 66% per-trade margin rail, 2bps market-exit
  slippage, isolated-margin liquidation), completed-candle validation produces **842,919×**
  shared continuous growth at 18.77% reported / 18.85% 4h mark-to-market maxDD. Held-out TEST
  (216.76×) and every annual fold stay green; the shipped 4.4% margin and 1.10× notional caps
  remain unchanged. (The per-trade rail is `position_sizing.max_position_pct: 0.66` — a
  fraction of the account, so it scales with equity and never freezes compounding; normal
  ~2-3% sizing sits far below it and it only stops a runaway size computation.)
- **Aggressive:** `config-aggressive.json`, `config-eth-aggressive.json`, and
  `config-sol-aggressive.json`; these small profiles inherit their standard asset config and
  disable the portfolio margin/notional caps. They remain on testnet by inheritance.
  Completed-candle validation produces **72.9 trillion×** with 36.17% reported / 35.32% 4h
  mark-to-market maxDD. This extreme path-dependent compounding is a robustness result, not a
  live-return forecast (no market-impact modeling); live drawdown can be materially worse than
  the approximately 35% history.

> **The in-sample multiples are yardsticks, not forecasts.** The honest expectation comes from
> the **clean out-of-sample holdout** — the frozen configs replayed on **2025-06 → 2026-04**
> (~11 months the strategy was never tuned on; `python -m opt.holdout_oos`): **standard 4.88×,
> aggressive 16.8×**. ⚠️ Out of sample the edge is carried by **ETH/SOL, not BTC** — BTC standalone
> is 1.30× (standard) and **0.71×, a loss, on aggressive**. It's still an upper bound (optimistic
> fills/fees/slippage). Start paper on the standard profile and watch BTC's live P&L.

### Week-by-week progression (Excel)

To see the *compounding path* rather than only the final multiple, export a weekly equity
progression to Excel. Each run produces one workbook with a Summary sheet, a combined "Weekly
Multiples" sheet (all curves side by side), and a per-study detail sheet (week #, week-ending
date, equity, multiple, weekly return %, drawdown %). Six studies per workbook:
`{standard, aggressive} × {BTC+ETH+SOL (default), BTC only, ETH only}`. Single-asset studies
reuse the identical shared-portfolio simulator with a one-symbol universe (so the standard
profile's portfolio caps still apply — a single-asset multiple here differs from a fully
standalone run).

```bash
# In-sample 2021-01 → 2025-06 → reports/weekly_progression.xlsx
PYTHONPATH=. python -m opt.weekly_progression

# Out-of-sample 2025-06 → 2026-04 → reports/weekly_progression_oos.xlsx
PYTHONPATH=. python -m opt.weekly_progression_oos
```

The OOS window ends **2026-04-30** (not 2026-06): Bitget futures history has genuine candle holes
in May 2026 for ETH/SOL and the data pipeline fails closed on gaps, so the window is truncated to
the last gap-free month boundary common to all three assets. OOS finals match the holdout above
(standard 3-asset 4.85×, aggressive 3-asset 16.87×, aggressive BTC-only 0.71× — a loss).

Reproduce the shared aggressive study with:

```bash
python -m opt.multi_portfolio --profile aggressive --entry-mode maker --exit-granularity sub
```

Corrected queue sensitivity is recorded in `opt/completed_candle_queue_results.json` (regenerated
2026-07-15 with the full execution model). The combined 5bps-penetration/70%-fill scenario retains
64.6% of baseline log growth across five deterministic seeds and keeps every annual fold green
(worst fold +398.6%).

### Lower-timeframe research and completed-candle audit

The research branch `experiment/lower-timeframes` transplants the shipped numeric strategy to 1h
and 5m without tuning. On common Binance USDT-perpetual BTC data (2021-01→2025-06), with maker
entry, funding, liquidation, and 2bps market-exit slippage, the causal 4h control returned 225.91×
at 15.80% maxDD. The 1h transplant returned 76.94× at 28.06% maxDD but lost 6.76% in 2025H1; the
5m transplant fell to 0.237× at 79.26% maxDD and lost in every annual fold. The existing 12× tier
reduced 1h risk but did not fix its losing 2025H1. The unchanged strategy therefore remains 4h.

The experiment also found that cached Bitget OHLCV is bar-open stamped while historical secondary
selection had used only that open timestamp. Round 23 fixed this throughout the full engine,
fastbt, Binance normalization, one-shot analysis, and live scheduling. Live now drops forming
candles, freezes all inputs at the completed primary close, persists an at-most-once decision key
across restarts, and polls once per minute so a new 4h bar is handled promptly. Corrected full/fast
2024 parity is exact. No paper/testnet process has been started.

Reproduce the audit with:

```bash
PYTHONPATH=. /tmp/tmlvenv/bin/python -m opt.lower_timeframes
```

Machine-readable results are in `opt/lower_timeframe_results.json`.

**Scalper second-product research (2026-07-19, PARKED):** a full from-scratch 5m/15m scalper
search lives in `opt/scalp/` (dedicated vectorized engine with 5m sub-bar replay, ~13k
backtests, pre-committed TRAIN/TEST/HOLDOUT protocol). One survivor: 15m Donchian-96 breakout
gated by ATR-expansion ≥ 1.3 — ~10%/yr at ~8% maxDD (equal-weight portfolio, 0.5%/trade loss
budget; TEST ≈ HOLDOUT). Pure 5m is structurally fee-dead; mean reversion and the retuned
house scorer are not competitive at 15m; BTC's short-horizon edge is gone post-2024. Marc
reviewed and parked it (not worth building a live path vs the 4h product's returns). Full
story + protocol + caveats: `opt/scalp/SCALPER_RESEARCH.md`. ⚠️ Its holdout is SPENT — any
revival must validate on new (live/paper) data, not more backtests.

Reproduce the corrected standard/aggressive validation with:

```bash
PYTHONPATH=. /tmp/tmlvenv/bin/python -m opt.completed_candle_validation
```

For eventual multi-symbol paper/live execution, use one serialized account orchestrator (do not
run independent symbol stacks against the same exposure budget):

```bash
python -m llm_trading_bot.main --mode live --shared-configs \
  config-aggressive.json config-eth-aggressive.json config-sol-aggressive.json
```

This command starts trading immediately. The repository prepares it but does not run it
automatically.

The current `config.json` is the output of an out-of-sample-validated optimization pass
(2026-07) — see `opt/README.md` for the methodology, the intrabar trailing-stop bug it
uncovered (backtests must assume the adverse extreme hits first), and full results.
The current configuration uses a post-only maker limit at the completed decision bar's
close, good for the following primary bar. Honest 1h sub-bar replay (2021-01→2025-06,
2bps market-exit slippage, liquidation, perp funding, and the per-trade margin cap modeled)
remains profitable in every yearly fold on standalone
**BTC (199.54×), ETH (1,789.35×), and SOL (126,351.99×)** with the shipped
standard portfolio exposure controls after a constrained
scoring-point search selected on BTC TRAIN and validated on BTC TEST plus untouched ETH/SOL.
These multiples are robustness signals, not forecasts. Queue/penetration stress tests are
available, but OHLC still cannot reproduce real queue priority, latency, outages, or partial fills.
Structural changes vs the original design:
**trailing stops ON** (activation 0.94%, callback 0.33%), **pyramiding** (up to 3
same-direction positions), **conviction sizing** (risk scales with |score|), bounded
**anti-martingale sizing**, portfolio-wide **margin/notional caps**, and an
**opposite-signal exit** (close on a hard composite flip, threshold 20). Marginal signals are
traded deterministically (the LLM-gate experiment lost to signal-only execution and was removed). A shared
BTC+ETH+SOL portfolio harness and leakage-free annual retuning/scoring-point experiments
live under `opt/`; see `opt/README.md` for validation results and caveats.

## License

Private — All rights reserved.
