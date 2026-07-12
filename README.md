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

The current `config.json` is the output of an out-of-sample-validated optimization pass
(2026-07) — see `opt/README.md` for the methodology, the intrabar trailing-stop bug it
uncovered (backtests must assume the adverse extreme hits first), and full results.
Headline (2021-01→2025-06, compounding, 2 bps slippage/side, liquidation AND perp
funding modeled): **BTC ≈228× with every year profitable (incl. the 2022 bear),
max DD ~22%** (84× at 5 bps); the same unchanged config on **ETH: ≈1015×**
(`config-eth.json`) — strong evidence the edge is structural, not curve-fit.
Structural changes vs the original design:
**trailing stops ON** (activation 0.94%, callback 0.33%), **pyramiding** (up to 3
same-direction positions), **conviction sizing** (risk scales with |score|), and an
**opposite-signal exit** (close on a hard composite flip, threshold 20).

## License

Private — All rights reserved.
