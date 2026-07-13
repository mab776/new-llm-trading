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
   ├── MARGINAL → Deterministic execution by default (optional LLM consensus)
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
| `openwebui` | Filter file + automation client |
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
  caps target approximately 25% historical shared max drawdown. Corrected 4h-close-cadence
  validation produced 292,212.44× shared continuous growth at 19.95% reported / 20.67% 4h
  mark-to-market maxDD. A looser TRAIN-selected cap failed held-out validation at 28.6% DD, so the
  shipped 4.4% margin and 1.10× notional caps remain unchanged.
- **Aggressive:** `config-aggressive.json`, `config-eth-aggressive.json`, and
  `config-sol-aggressive.json`; these small profiles inherit their standard asset config and
  disable the portfolio margin/notional caps. They remain on testnet by inheritance. Correcting a
  sub-bar harness cadence bug (the stop had been ratcheting hourly instead of once per completed
  4h bar) produced 5.749 trillion× with 34.28% reported / 34.11% 4h mark-to-market maxDD. This
  extreme path-dependent compounding is a robustness result, not a live-return forecast.

Reproduce the shared aggressive study with:

```bash
python -m opt.multi_portfolio --profile aggressive --entry-mode maker --exit-granularity sub
```

Queue sensitivity is recorded in `opt/queue_fill_sensitivity_results.json`. Even the combined
5bps-penetration/70%-fill scenario retained about 65.6% of baseline log growth across five
deterministic seeds, kept every annual fold green, and had 38.15% worst 4h mark-to-market DD.

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
2bps market-exit slippage, liquidation and perp funding modeled) remains profitable in
every yearly fold on standalone **BTC (301.18×), ETH (2,436.13×), and SOL (66,125.23×)** with the shipped
standard portfolio exposure controls after a constrained
scoring-point search selected on BTC TRAIN and validated on BTC TEST plus untouched ETH/SOL.
These multiples are robustness signals, not forecasts. Queue/penetration stress tests are
available, but OHLC still cannot reproduce real queue priority, latency, outages, or partial fills.
Structural changes vs the original design:
**trailing stops ON** (activation 0.94%, callback 0.33%), **pyramiding** (up to 3
same-direction positions), **conviction sizing** (risk scales with |score|), bounded
**anti-martingale sizing**, portfolio-wide **margin/notional caps**, and an
**opposite-signal exit** (close on a hard composite flip, threshold 20). The shipped configs use
`openwebui.marginal_execution: "deterministic"` because the completed LLM-gate experiment lost to
signal-only execution; `"consensus"` remains opt-in. A shared
BTC+ETH+SOL portfolio harness and leakage-free annual retuning/scoring-point experiments
live under `opt/`; see `opt/README.md` for validation results and caveats.

## License

Private — All rights reserved.
