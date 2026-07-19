# AGENTS.md — Architecture & Development Guide

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

This document is for AI agents and developers working on the LLM Trading Bot project.

## Project Overview

An automated cryptocurrency trading bot using deterministic technical analysis scoring — a pure
technical-signal bot. The former optional LLM consensus / marginal-gate was tested, found
net-negative, and removed (last commit with it: tag `last-llm-consensus`).

## Architecture

### 3-Tier Signal Routing Pipeline

```
Market Data → Scoring Engine → Signal Router
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼             ▼
                STRONG       MARGINAL         WAIT
            (Template)   (Deterministic;    (Skip)
                           LLM opt-in)
                    │            │
                    ▼            ▼
               Bitget Execution (TP/SL mandatory)
```

### Module Map

| Module | File | Purpose |
|--------|------|---------|
| **Config** | `llm_trading_bot/config.py` | Pydantic models, config loading |
| **OpenWebUI Filter** | `openwebui_filter.py` | **SOURCE OF TRUTH** — indicator computations + scoring logic |
| **Scoring** | `llm_trading_bot/scoring.py` | Typed API layer (IndicatorSet, CategoryScore, etc.) — imports from filter |
| **Entry lifecycle** | `llm_trading_bot/entry.py` | Shared maker-limit fill rule + pending-entry structure |
| **Data** | `llm_trading_bot/data.py` | OHLCV fetching, caching, 4H aggregation, source routing |
| **Bitget history** | `llm_trading_bot/bitget_csv.py` | Windowed (END-anchored) Bitget fetch + monthly disk cache |
| **Binance history** | `llm_trading_bot/binance_csv.py` | Binance public CSV archive downloader |
| **Funding** | `llm_trading_bot/funding.py` | Perp funding-rate history (Binance proxy) + per-bar settlement math for backtests |
| **Timeframes** | `llm_trading_bot/timeframes.py` | Bar durations, completed-candle slicing, frozen live snapshots |
| **Routing** | `llm_trading_bot/routing.py` | Signal classification and routing decisions |
| **Signal library** | `openwebui_filter.py` | Indicator math + category scoring (source of truth) |
| **Exchange** | `llm_trading_bot/exchange.py` | Bitget API — orders, balance, stop updates, safety checks |
| **Trailing** | `llm_trading_bot/trailing.py` | Shared trailing-stop math (backtest + live, no drift) |
| **Live state** | `llm_trading_bot/live_state.py` | Atomic persisted account peak + pending/trailing lifecycle state |
| **Live orchestration** | `llm_trading_bot/orchestrator.py` | Serialized multi-symbol scheduling and process lock |
| **Portfolio** | `llm_trading_bot/portfolio.py` | Fee-aware portfolio simulation |
| **Backtesting** | `llm_trading_bot/backtesting.py` | Historical replay engine |
| **Reporting** | `llm_trading_bot/reporting.py` | Charts, text reports, CSV export |
| **Scheduler** | `llm_trading_bot/scheduler.py` | Scheduling + risk-based sizing + live trailing stops |
| **Main** | `llm_trading_bot/main.py` | Entry point (analyze, backtest, live modes) |

### Risk Management (imported from predecessor project)

The `RiskManagementConfig` in `config.py` controls four features ported from the old project:

| Feature | Config Key | Default | Effect |
|---------|-----------|---------|--------|
| **Max holding time** | `max_holding_hours` | 168 (7 days) | Force-close after N hours |
| **Post-SL cooldown** | `cooldown_candles_after_sl` | 3 | Skip N candles after SL hit |
| **Consecutive loss penalty** | `consecutive_loss_penalty` | 0.0 (removed 2026-07-19, was 5.0) | Raise entry threshold per loss |
| **Maker/taker fees** | `use_maker_fee_for_tp` | true | TP→maker fee, SL→taker fee |

Entries default to `trading.entry_mode: "maker"`: place a post-only limit at the
completed decision bar's close, keep it for the following primary bar, fill only if that
bar trades back to the limit, otherwise cancel. A fill is immediately exposed to the
fill bar's adverse-first SL/TP checks. The pending order counts as a position slot and is
placed with mandatory preset SL+TP. `"taker"` remains available for comparison/fallback.

Marginal signals are traded deterministically, matching the backtest's auto-trade behavior. The
per-entry LLM gate (Round 8/8c, and re-tested 2026-07-15 with vLLM `qwen3.6-27b` in both
thinking and no-thinking modes) was net-negative every time and has been removed entirely.

`scoring.points` contains the nine OOS/cross-asset-validated overrides selected in Round 14.
All other point awards come from `openwebui_filter.DEFAULT_SCORING_POINTS`. Never copy these
values into `scoring.py`; the filter remains the source of truth.

Round 15 found anti-martingale sizing was not sufficient as a standalone DD control. Round 16 then
validated it as a return overlay under portfolio-wide ex-ante exposure caps. Round 17 preserved
that capped policy as the default and added separate `*-aggressive.json` profiles which inherit the
base configs but disable the shared margin/notional caps. The shared math lives in
`llm_trading_bot/exposure.py` and is used by fastbt, the full engine, and live scheduling.

Round 18 corrected a later-discovered sub-bar harness cadence violation: 1h exit replay must keep
the trailing stop fixed through all sub-bars and ratchet exactly once after the completed 4h bar.
The corrected shared continuous results are 292,212.44× at 19.95% reported / 20.67% mark-to-market
maxDD for the standard profile and 5.749 trillion× at 34.28% reported / 34.11% mark-to-market maxDD
for aggressive. These path-dependent multiples are robustness results, never forecasts. The old
Round 16/17 sub-bar headline results are superseded by `opt/cadence_correction_results.json`.

Round 22's lower-timeframe research found a separate completed-candle alignment issue that must be
resolved before paper trading. Round 23 resolved it across Binance timestamp normalization, the
full engine, fastbt, one-shot analysis, and live scheduling. All indexes now mean bar open and a
row is visible only after `bar_open + duration <= decision_time`. Live snapshots exclude forming
rows, are frozen at the completed primary close, and use a persisted at-most-once bar key across
restarts. A one-minute poll detects UTC 4h closes promptly without repeated fetching/scoring.
Corrected full/fast 2024 parity is exact (+177.54%, 590 trades, 16.74% maxDD; rolling VWAP).

Round 23 corrected shared continuous results (superseded 2026-07-15, kept for history): standard
445,508.49× at 17.94%/18.03% maxDD; aggressive 4.976 trillion× at 38.47%/38.67% maxDD.

**2026-07-15 paper-readiness execution parity (current headline).** The simulators now enforce
the full live execution model in the full engine, fastbt, and the shared multi-asset harness;
live sizing switched to realized balance (equity − open PnL, backtest parity); live gained the
backtest's post-SL cooldown, consecutive-loss entry penalty, and `max_holding_hours` (persisted
per symbol in live state v4); `--mode backtest` gained config-driven
`slippage_pct`/`model_liquidation` matching fastbt semantics; and the shipped configs switched
to **isolated margin** to match the harness's liquidation model. Full/fast 2024 maker parity
remains exact at the new settings (+150.42%, 571 trades, 17.56% maxDD, zero mismatches — the
drop from +223.04% is the now-modeled slippage/liquidation).

The per-trade margin rail is **`position_sizing.max_position_pct: 0.66`** — a FRACTION of the
sizing balance (`margin = balance × min(risk_pct, 0.66)`), enforced identically in live and all
simulators. It replaced the absolute `max_position_usd` (Marc: a fixed USD cap silently freezes
compounding once the account grows; a fraction scales with equity). Normal sizing (~2-3%) never
reaches 66% — the rail exists purely to stop a runaway size computation from betting most of
the account on one trade.

Regenerated gap-free completed-candle results (`opt/completed_candle_results.json`, rolling
reproducible VWAP): standard **842,919.58×** continuous at 18.77% reported / 18.85%
mark-to-market maxDD; aggressive **72.9 trillion×** at 36.17% reported / 35.32% mark-to-market
maxDD. Standalone standard BTC/ETH/SOL: 303.97× / 2,755.59× / 430,108.74× (the huge SOL
dispersion underlines that these are path-dependent, not a portable per-asset edge). Held-out
TEST: standard 216.76×, aggressive 3,348,599.63×; every annual fold green (standard worst
+162%). Queue stress (`opt/queue_fill_sensitivity_results.json`) predates the VWAP fix and
should be rerun. These path-dependent multiples are robustness results, never forecasts (no
market-impact modeling); the maker fill model fills ~99.9% of touched limits, which live will
not.

**The honest number to plan against — clean out-of-sample holdout** (`opt/holdout_oos.py`, frozen
configs replayed on **2025-06 → 2026-04**, ~11 months never tuned on; rolling VWAP, full execution
model): standard **4.88×** (17.0% MTM DD, 1,044 trades), aggressive **16.8×** (35.0% DD, 1,638
trades). ⚠️ Per-asset the edge is carried by **ETH/SOL, not BTC** — standalone OOS standard
BTC 1.30× / ETH 4.87× / SOL 3.47×; aggressive **BTC 0.71× (a loss)** / ETH 10.14× / SOL 5.15×.
BTC (the tuned asset) is the weakest OOS and goes negative under 25×; the portfolio masks it.
Still an upper bound (optimistic fills/fees/slippage), single ~11-month regime.

**Week-by-week progression export** — `opt/weekly_progression.py` (in-sample 2021-01→2025-06 →
`reports/weekly_progression.xlsx`) and `opt/weekly_progression_oos.py` (OOS 2025-06→2026-04 →
`reports/weekly_progression_oos.xlsx`) render the *compounding path*, not just the final multiple,
to Excel. Each workbook = Summary + combined "Weekly Multiples" + six per-study detail sheets
(`{standard, aggressive} × {BTC+ETH+SOL, BTC only, ETH only}`); single-asset studies drive the
same shared-portfolio simulator with a one-symbol universe (portfolio caps still apply, so these
differ from a fully standalone run). Both reuse `collect_studies`/`build_workbook` and replay via
`simulate_multi` (maker entry, 1h sub-bar, funding, liquidation, 2bps slippage) — the OOS finals
reproduce the holdout above. The OOS window stops at 2026-04-30 because Bitget has genuine May-2026
candle holes for ETH/SOL and the data layer fails closed on gaps (`load_context`/`load_assets` now
take optional `data_end`/`funding_end` to replay past the in-sample cutoff).

### Key Design Principle: Single Source of Truth

**`openwebui_filter.py` is THE source of truth** for all indicator computations (`compute_ema`,
`compute_rsi`, `compute_adx`, etc.) and scoring logic (`calc_trend_score`, `calc_momentum_score`,
etc.). `scoring.py` imports these functions and provides a typed dataclass API
(`IndicatorSet`, `CategoryScore`, `ScoringResult`) that the rest of the package uses.
This ensures backtesting, live trading, and the OpenWebUI filter all exercise the
**same** calculation code — zero duplication.

When the filter is copy-pasted into OpenWebUI, it works standalone because it contains
all the canonical functions. When used in the project, `scoring.py` imports from it.

## Safety Rules (Non-Negotiable)

These rules must NEVER be violated. Any PR that breaks these must be rejected:

1. **Never place an order without a stop loss AND take profit** — enforced in `exchange.py` via `SafetyViolation`
2. **Never hardcode API credentials** — all creds come from `config.json`
3. **Confidence bounded to [5%, 95%]** — never 0% (false certainty of neutral) or 100% (false certainty of direction)
4. **Backtesting never peeks at future data** — each bar only sees data up to and including itself
5. **All PnL accounts for fees on leveraged notional** — `fee = size × price × fee_rate`, not on margin
6. **Never duplicate core logic** — one source of truth in `openwebui_filter.py`, imported by `scoring.py`

## Configuration

All settings in `config.json`. Key sections:

- `trading` — Symbol, timeframes, leverage tiers, SL strategy
- `scoring` — Category weights, ATR multipliers, confidence bounds
- `filters` — Pre-trade filter thresholds
- `fees` — Maker/taker rates (default: Bitget rates)
- `bitget` — Exchange credentials (NEVER in code)
- `backtesting` — Date range, initial balance, warmup
- `scheduling` — Interval timings

### Leverage Tiers

Two tiers with different risk profiles (optimized + OOS-validated 2026-07, see
`opt/README.md`; both share the same signal params — leverage is the risk dial):

| Setting | Conservative | Aggressive |
|---------|-------------|------------|
| Leverage | 12x | 25x |
| Strong threshold | 21.3 | 21.3 |
| Marginal low | 12.6 | 12.6 |
| TP1 R:R | 2.02 | 2.02 |
| TP2 R:R | 3.34 | 3.34 |
| TP1 exit | 70% | 70% |

Trailing stops are **enabled** (activation 0.94%, callback 0.33% of entry) — the single
biggest contributor to the edge; stop strategy is `atr` with `atr_sl_multiplier: 2.26`.

⚠️ **Trailing-ratchet cadence IS the strategy.** The stop ratchets once per COMPLETED
primary (4h) bar using that bar's favorable extreme, and stays FIXED intrabar (the
exchange triggers it if touched). An honest 1h sub-bar backtest showed hourly ratcheting
collapses the edge 84×→5× and no wider callback recovers it. The live scheduler
(`_maybe_trail_stop`) implements bar-close cadence with a `last_trail_bar` gate — never
revert it to per-tick/current-price trailing (guarded by `tests/test_trailing_cadence.py`).

Strategy features (2026-07, implemented in BOTH `backtesting.py` and `scheduler.py`
— keep them in sync):

- **Pyramiding** — `position_sizing.max_positions` (3): concurrent SAME-direction
  positions; never stacks against an open opposite position.
- **Conviction sizing** — `position_sizing.conviction_exponent` (1.0): per-trade risk
  scaled by `clamp((|score|/strong_threshold)^k, 0.5, 1.5)`; 0 disables.
- **Anti-martingale sizing** — a causal per-asset closed-trade streak changes risk by 0.05 per
  outcome, bounded to 0.70×–1.10×. Live derives it from Bitget net position history.
- **Portfolio exposure caps** — the default profile caps total open + resting-entry exposure at
  4.4% account-equity margin and 1.10× account-equity entry notional. Existing positions are never
  force-closed to comply; the new order is reduced or skipped. The separately named aggressive
  profiles disable these two caps and must never be confused with the default policy.
- **Opposite-signal exit** — `risk_management.opposite_exit_threshold` (20): when the
  composite score flips ≥ threshold against open positions they are closed at market
  (`signal_flip`; does NOT trigger the SL cooldown); 0 disables.
- **DD circuit-breaker** — `risk_management.dd_throttle_threshold` (0.25): while balance
  drawdown from its peak ≥ threshold, entry slots cap at `dd_throttle_slots` (1) and risk
  is multiplied by `dd_throttle_risk` (0.5) until equity recovers. This is **tail
  insurance against a regime break**, deliberately wide — tight thresholds cost return
  (they cut exposure right before the V-shaped recovery); don't tune it below ~0.20 from
  in-sample data. Live: peak is in-session (resets on restart).

### Backtest intrabar conservatism (do not regress)

With only OHLC per bar, the intrabar path is unknown, so the engine assumes the
**adverse extreme is reached first**: SL is checked before TP (a bar spanning both is a
loss), and the trailing stop is ratcheted only AFTER exit checks, using the bar's
favorable extreme, taking effect on subsequent bars. Guarded by
`tests/test_intrabar_conservatism.py` — any change that makes these fail is inflating
backtest results.

Historical exchange candles may be stamped at bar open. "Index <= decision index" is not enough
to establish availability across timeframes: compare candle close times. A secondary candle is
causal only when `secondary_open + secondary_duration <= primary_open + primary_duration`.

## Development Guidelines

### Running

```bash
# Single analysis (no trading)
python -m llm_trading_bot.main --mode analyze

# Backtest
python -m llm_trading_bot.main --mode backtest

# Live scheduled trading
python -m llm_trading_bot.main --mode live
```

### Testing

```bash
python run_tests.py            # Convenience script
pytest tests/ -v               # Full verbose
pytest tests/test_scoring.py   # Specific module
pytest -x                      # Stop on first failure
```

### Adding Indicators

1. Add the calculation function in `openwebui_filter.py` (e.g., `compute_new_indicator()`)
2. Add the field to `IndicatorSet` dataclass in `scoring.py`
3. Call it in `calculate_indicators()` in `scoring.py` and store the result
4. Also call it in `_compute_analysis()` in `openwebui_filter.py` and store in the dict
5. Use it in the appropriate scoring category function
6. Add it to `format_indicator_report()` for LLM context
7. Add tests in `tests/test_scoring.py`

### Adding Scoring Categories

1. Create `calc_new_category_score()` in `openwebui_filter.py`
2. Create `score_new_category()` wrapper in `scoring.py`
2. Add weight to config schema and `config.json`
3. Register in `compute_composite_score()` → `cat_funcs` dict
4. Add tests

### Modifying Safety Rules

**DON'T.** If you think a safety rule needs changing, document why thoroughly
and get explicit approval. The `SafetyViolation` exception in `exchange.py`
exists for a reason — it's the last line of defense.

### Data Flow for Backtesting

```
1. Fetch full date range + warmup from the configured source (Bitget by default)
2. For each bar in test period:
   a. Slice data UP TO current bar (no future data)
   b. Calculate indicators on the slice
   c. Check exits on open positions FIRST
   d. Score and possibly open new position
   e. Record snapshot
3. Force-close remaining positions at end
4. Compute stats and generate reports
```

### Market Data Sources

`data_source.source` in `config.json` selects the OHLCV source (routing lives in
`data.py::fetch_ohlcv`):

- **`bitget`** (default) → `bitget_csv.py`. USDT-perpetual futures (`BTC/USDT:USDT`,
  `market: "futures"`). ⚠️ **Bitget's history endpoint is 200-cap and END-anchored** — it
  returns the last N candles *before* `until`, so a naive `since` request silently returns
  only the tail. We page in **explicit ≤200-candle windows** passing `params={"until":
  window_end}` and filter each window to `[since, end)` (`fetch_ohlcv_range`). Complete
  months are cached to `history/bitget/{SYMBOL}/{tf}/` (the current partial month is always
  re-fetched). The live ccxt fallback (`_fetch_ccxt`) uses the same windowing for Bitget.
  Bitget **spot** lacks 2h/1s candles; futures has full coverage.
- **`binance`** → `binance_csv.py` (public CSV archive), with a plain forward-paginated
  ccxt fallback.
- **`yfinance`** → capped at ~730 days hourly; 4H aggregated from 1H in `data.py`.

### Position Sizing & Trailing Stops

- **Sizing** (`position_sizing` config): both live and backtest commit
  `realized_balance × min(risk_pct_per_trade, max_position_pct)` as margin, leveraged to the
  notional, converted to base size at entry. `max_position_pct` (0.66) is a scale-invariant
  per-trade rail enforced in the full engine, fastbt, AND the shared harness — normal ~2-3%
  sizing never reaches it; it only stops a runaway size computation. Live
  realized balance = exchange equity − open unrealized PnL (backtest parity), additionally
  bounded by `get_available_balance()` because reserved maker margin can't be committed twice.
- **Shared risk profiles**: the default capped profile targets natural realized shared-portfolio
  maxDD of approximately 25%. Completed-candle validation realizes 18.77% reported / 18.85% 4h
  mark-to-market maxDD, so the existing caps remain. The explicit aggressive profile realizes
  approximately 35% corrected historical maxDD in exchange for uncapped compounding; never present
  that backtest as a forecast or assume live DD cannot be materially worse.
- **Trailing stops**: `trailing.py::compute_trailing_stop` is the single source of truth,
  used by `backtesting.py::_update_trailing_stop` AND `scheduler.py::_maybe_trail_stop`
  (which calls `exchange.modify_stop_loss`). A stop only ever moves in the trade's favour.

### Important Technical Notes

- **yfinance doesn't support 4H candles** — we fetch 1H and aggregate in `data.py`
- **yfinance caps hourly data at ~730 days** — prefer `bitget` for deep backtests
- **Fees compound significantly at high leverage** — a 0.06% fee at 20x = 2.4% per round trip
- **Maker entry is not guaranteed** — post-only limits can miss or lose queue priority; live
  reconciliation cancels unfilled orders after one completed primary bar
- **One account, one orchestrator** — multi-symbol live/paper execution must use
  `--shared-configs` so exposure check/size/place is serialized and the account peak plus pending
  and trailing state survive restarts. Do not run independent symbol processes against one budget.
- **Live decisions are completed-bar gated** — the scheduler polls once per minute so UTC 4h
  closes are detected promptly, but persisted `last_analysis_bars` permits at most one claimed
  decision/execution attempt per symbol/completed primary bar, including across restarts.
- **ATR adapts to volatility** — all targets (SL, TP1, TP2) scale with market conditions
- **Partial exits** — TP1 closes a fraction (default 50%), TP2 closes the rest
- **The OpenWebUI filter file is self-contained** — it contains the canonical indicator and scoring functions that `scoring.py` imports

## File Organization

```
new-llm-trading/
├── config.json              # All configuration
├── requirements.txt         # Python dependencies
├── run_tests.py            # Test runner convenience script
├── openwebui_filter.py     # STANDALONE OpenWebUI filter (copy to OpenWebUI)
├── AGENTS.md               # This file
├── README.md               # User-facing docs
├── RECREATE_PROMPT.md      # Original project prompt
├── llm_trading_bot/        # Main package
│   ├── __init__.py
│   ├── config.py           # Configuration models
│   ├── scoring.py          # CORE: indicators + scoring + targets
│   ├── data.py             # OHLCV fetching + caching + source routing
│   ├── bitget_csv.py       # Bitget windowed history getter + disk cache
│   ├── binance_csv.py      # Binance CSV archive downloader
│   ├── routing.py          # Signal routing logic
│   ├── exchange.py         # Bitget API + safety + balance + stop updates
│   ├── trailing.py         # Shared trailing-stop math
│   ├── portfolio.py        # Portfolio simulation
│   ├── backtesting.py      # Backtest engine
│   ├── reporting.py        # Charts + reports
│   ├── scheduler.py        # Scheduled trading + sizing + trailing
│   └── main.py             # Entry point
├── tests/                  # Pytest test suite
│   ├── __init__.py
│   ├── test_scoring.py
│   ├── test_routing.py
│   ├── test_portfolio.py
│   ├── test_backtesting.py
│   └── test_exchange.py
├── reports/                # Generated reports (gitignored)
└── logs/                   # Trading logs (gitignored)
```

## Tech Stack

- **Python 3.13+**
- `yfinance` — Market data
- `pandas` / `numpy` — Data processing
- `pydantic` — Config validation
- `matplotlib` — Charts
- `requests` — HTTP (OpenWebUI, Bitget)
- `schedule` — Cron-like scheduling
- `pytest` — Testing
