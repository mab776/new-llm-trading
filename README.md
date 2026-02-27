# LLM Trading Bot

An automated cryptocurrency trading bot using a hybrid intelligence approach:
deterministic technical analysis scoring combined with LLM reasoning for marginal signals.

## Quick Start

```bash
pip install -r requirements.txt
cp config.json config.local.json  # Edit with your credentials
python -m llm_trading_bot.main --config config.json
```

## Architecture

```
Market Data (yfinance)
        ↓
  Scoring Engine (indicators + weighted score)
        ↓
  Signal Router
   ├── STRONG  → Template response (instant, free)
   ├── MARGINAL → LLM consensus via OpenWebUI
   └── WAIT    → Skip trade
        ↓
  Bitget Execution (futures + mandatory TP/SL)
```

## Modules

| Module | Purpose |
|--------|---------|
| `scoring` | Core scoring engine, indicators, targets |
| `data` | OHLCV fetching, caching, 4H aggregation |
| `routing` | Signal classification and routing |
| `openwebui` | Filter file + automation client |
| `exchange` | Bitget futures integration |
| `backtesting` | Historical replay engine |
| `portfolio` | Fee-aware portfolio simulation |
| `reporting` | Reports and charts |
| `scheduler` | Cron-like scheduling |

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

## License

Private — All rights reserved.
