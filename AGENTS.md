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
| **Scoring** | `llm_trading_bot/scoring.py` | **CORE** — all indicators, scoring, targets, filters |
| **Data** | `llm_trading_bot/data.py` | OHLCV fetching, caching, 4H aggregation |
| **Routing** | `llm_trading_bot/routing.py` | Signal classification and routing decisions |
| **OpenWebUI Filter** | `openwebui_filter.py` | **STANDALONE** single-file filter for OpenWebUI |
| **OpenWebUI Client** | `llm_trading_bot/openwebui_client.py` | API client + consensus mechanism |
| **Exchange** | `llm_trading_bot/exchange.py` | Bitget API with mandatory safety checks |
| **Portfolio** | `llm_trading_bot/portfolio.py` | Fee-aware portfolio simulation |
| **Backtesting** | `llm_trading_bot/backtesting.py` | Historical replay engine |
| **Reporting** | `llm_trading_bot/reporting.py` | Charts, text reports, CSV export |
| **Scheduler** | `llm_trading_bot/scheduler.py` | Cron-like scheduling + position management |
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

**`scoring.py` is THE source of truth** for all technical calculations. Every module that needs
indicators, scores, targets, or filters imports from `scoring.py`. Never duplicate calculation
logic. The OpenWebUI filter (`openwebui_filter.py`) is the ONE exception — it must be
self-contained because it's copy-pasted into OpenWebUI.

## Safety Rules (Non-Negotiable)

These rules must NEVER be violated. Any PR that breaks these must be rejected:

1. **Never place an order without a stop loss AND take profit** — enforced in `exchange.py` via `SafetyViolation`
2. **Never hardcode API credentials** — all creds come from `config.json`
3. **Confidence bounded to [5%, 95%]** — never 0% (false certainty of neutral) or 100% (false certainty of direction)
4. **Backtesting never peeks at future data** — each bar only sees data up to and including itself
5. **All PnL accounts for fees on leveraged notional** — `fee = size × price × fee_rate`, not on margin
6. **Never duplicate core logic** — one source of truth in `scoring.py`

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

Two tiers with different risk profiles:

| Setting | Conservative | Aggressive |
|---------|-------------|------------|
| Leverage | 5x | 15x |
| Strong threshold | 70 | 80 |
| Marginal range | 45-70 | 55-80 |
| TP1 R:R | 2.0 | 1.5 |
| TP2 R:R | 3.5 | 2.5 |

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

1. Add the calculation function in `scoring.py` (e.g., `compute_new_indicator()`)
2. Add the field to `IndicatorSet` dataclass
3. Call it in `calculate_indicators()` and store the result
4. Use it in the appropriate scoring category function
5. Add it to `format_indicator_report()` for LLM context
6. Add tests in `tests/test_scoring.py`
7. If needed in the OpenWebUI filter, also add to `openwebui_filter.py` (keep it self-contained)

### Adding Scoring Categories

1. Create `score_new_category()` in `scoring.py`
2. Add weight to config schema and `config.json`
3. Register in `compute_composite_score()` → `cat_funcs` dict
4. Add tests

### Modifying Safety Rules

**DON'T.** If you think a safety rule needs changing, document why thoroughly
and get explicit approval. The `SafetyViolation` exception in `exchange.py`
exists for a reason — it's the last line of defense.

### Data Flow for Backtesting

```
1. Fetch full date range + warmup from yfinance
2. For each bar in test period:
   a. Slice data UP TO current bar (no future data)
   b. Calculate indicators on the slice
   c. Check exits on open positions FIRST
   d. Score and possibly open new position
   e. Record snapshot
3. Force-close remaining positions at end
4. Compute stats and generate reports
```

### Important Technical Notes

- **yfinance doesn't support 4H candles** — we fetch 1H and aggregate in `data.py`
- **yfinance caps hourly data at ~730 days** — plan date ranges accordingly
- **Fees compound significantly at high leverage** — a 0.06% fee at 20x = 2.4% per round trip
- **ATR adapts to volatility** — all targets (SL, TP1, TP2) scale with market conditions
- **Partial exits** — TP1 closes a fraction (default 50%), TP2 closes the rest
- **The OpenWebUI filter file is self-contained** — it duplicates indicator code intentionally

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
│   ├── data.py             # OHLCV fetching + caching
│   ├── routing.py          # Signal routing logic
│   ├── openwebui_client.py # LLM consensus client
│   ├── exchange.py         # Bitget API + safety
│   ├── portfolio.py        # Portfolio simulation
│   ├── backtesting.py      # Backtest engine
│   ├── reporting.py        # Charts + reports
│   ├── scheduler.py        # Scheduled trading
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
