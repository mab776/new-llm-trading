# AGENTS.md — Architecture & Development Guide

This document is for AI agents and developers working on the LLM Trading Bot project.

## Project Overview

An automated cryptocurrency trading bot using a **hybrid intelligence approach**:
deterministic technical analysis scoring combined with LLM reasoning for marginal signals.

## Architecture

### 3-Tier Signal Routing Pipeline

```
Market Data → Scoring Engine → Signal Router
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼             ▼
                STRONG       MARGINAL         WAIT
            (Template)    (LLM Consensus)   (Skip)
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
| **Data** | `llm_trading_bot/data.py` | OHLCV fetching, caching, 4H aggregation, source routing |
| **Bitget history** | `llm_trading_bot/bitget_csv.py` | Windowed (END-anchored) Bitget fetch + monthly disk cache |
| **Binance history** | `llm_trading_bot/binance_csv.py` | Binance public CSV archive downloader |
| **Routing** | `llm_trading_bot/routing.py` | Signal classification and routing decisions |
| **OpenWebUI Client** | `llm_trading_bot/openwebui_client.py` | API client + robust JSON parsing + consensus |
| **Exchange** | `llm_trading_bot/exchange.py` | Bitget API — orders, balance, stop updates, safety checks |
| **Trailing** | `llm_trading_bot/trailing.py` | Shared trailing-stop math (backtest + live, no drift) |
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
| **Consecutive loss penalty** | `consecutive_loss_penalty` | 5.0 | Raise entry threshold per loss |
| **Maker/taker fees** | `use_maker_fee_for_tp` | true | TP→maker fee, SL→taker fee |

These are implemented in both `backtesting.py` (full engine) and `grid_search.py` (fast backtest).

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

- `openwebui` — Connection to OpenWebUI for LLM consensus
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

### Backtest intrabar conservatism (do not regress)

With only OHLC per bar, the intrabar path is unknown, so the engine assumes the
**adverse extreme is reached first**: SL is checked before TP (a bar spanning both is a
loss), and the trailing stop is ratcheted only AFTER exit checks, using the bar's
favorable extreme, taking effect on subsequent bars. Guarded by
`tests/test_intrabar_conservatism.py` — any change that makes these fail is inflating
backtest results.

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
  `min(balance × risk_pct_per_trade, max_position_usd)` as margin, leveraged to the notional,
  converted to base size at entry. Live reads the balance via
  `BitgetClient.get_available_balance()` (dry-run returns a default so sizing still works).
- **Trailing stops**: `trailing.py::compute_trailing_stop` is the single source of truth,
  used by `backtesting.py::_update_trailing_stop` AND `scheduler.py::_maybe_trail_stop`
  (which calls `exchange.modify_stop_loss`). A stop only ever moves in the trade's favour.

### Important Technical Notes

- **yfinance doesn't support 4H candles** — we fetch 1H and aggregate in `data.py`
- **yfinance caps hourly data at ~730 days** — prefer `bitget` for deep backtests
- **Fees compound significantly at high leverage** — a 0.06% fee at 20x = 2.4% per round trip
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
│   ├── openwebui_client.py # LLM consensus client (robust JSON parsing)
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
│   ├── test_consensus.py
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
