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
Market Data (Bitget futures — windowed history + disk cache; also binance/yfinance)
        ↓
  Scoring Engine (indicators + weighted score)
        ↓
  Signal Router
   ├── STRONG  → Template response (instant, free)
   ├── MARGINAL → LLM consensus via OpenWebUI
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
| `scoring` | Core scoring engine, indicators, targets |
| `data` | OHLCV fetching, caching, 4H aggregation, source routing |
| `bitget_csv` | Bitget windowed history getter + monthly disk cache |
| `binance_csv` | Binance public CSV archive downloader |
| `routing` | Signal classification and routing |
| `openwebui` | Filter file + automation client |
| `exchange` | Bitget futures integration (orders, balance, stop updates) |
| `trailing` | Shared trailing-stop math (backtest + live) |
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

## License

Private — All rights reserved.
