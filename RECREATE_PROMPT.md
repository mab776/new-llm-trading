# Prompt to Recreate: LLM Hybrid Trading Bot

## PROMPT START

I want you to build an **automated cryptocurrency trading bot** in Python that uses a **hybrid intelligence approach**: deterministic technical analysis scoring combined with LLM reasoning for marginal signals. The project name is "LLM Trading Bot".

---

### 1. HIGH-LEVEL ARCHITECTURE

The system has a **3-tier signal routing pipeline**:

```
high score  → STRONG signal  → Deterministic template response (instant, free, no LLM)
moderate score  → MARGINAL signal → Send to LLM (via OpenWebUI) for multi-bot consensus
low score  → WAIT signal    → Skip trade
```

The key innovation is **"Financial Data Injection"**: all technical indicators are pre-calculated in Python and injected into the LLM prompt context. This reduces hallucination, cuts API costs, and ensures the LLM reasons over accurate data instead of inventing numbers. This is done by using the filters of openwebui to inject the data into the "user message" before the actual prompt.

It will also be possible to use this filter directly in OpenWebUI for interactive analysis, where the user can ask questions about the market and the LLM will have access to the same pre-calculated indicators.

---

### 2. DATA FLOW

1. **Fetch** OHLCV data from Yahoo Finance (`yfinance`) for multi-timeframe analysis
2. **Calculate** useful technical indicators
3. **Score** the market with a weighted scoring algorithm that produces a directional signal and confidence percentage
4. **Route** based on score: STRONG → template, MARGINAL → LLM consensus, WAIT → skip
5. **Execute** via Bitget API (futures) with mandatory TP/SL orders

---

### 3. WHAT THE BOT NEEDS TO DO

**Scoring & Decisions**: The bot needs a scoring system that looks at trend, momentum, volume, support/resistance levels, and risk factors to produce a directional signal (bullish/bearish/neutral) with a confidence level. It should support two leverage tiers — a conservative one and an aggressive one — with different thresholds and risk/reward ratios appropriate for each.

**Targets**: Entry, stop loss, and take profit levels should all be ATR-based so they adapt to volatility. I'd like support for different stop-loss strategies (pure ATR, structure-based using S/R levels, or a hybrid). TP1 and TP2 with configurable R:R ratios.

**Pre-trade filters**: Before taking any trade, the system should automatically skip if the market is ranging (low ADX), volatility is too low for the trade to be profitable after fees, or if the expected profit at TP1 wouldn't even cover trading fees at the given leverage. Don't waste money on bad setups.

**The hybrid routing** as described in section 1: strong signals get a template response, marginal signals go to the LLM for consensus, weak signals are skipped.

**OpenWebUI integration**: The scoring engine should also work as an OpenWebUI Filter so I can use it interactively — it injects the pre-calculated data into my message, the LLM sees accurate numbers, and I get a formatted response. For automation, a separate module calls OpenWebUI's API, sends the same pre-calculated data, and parses the LLM's structured response.

**Bitget integration**: Trade execution through Bitget's futures API. The system must **absolutely refuse** to place any order without a stop loss and take profit attached — this is the most important safety feature. Support testnet for testing and trailing stops as an alternative exit strategy.

**Backtesting**: A backtesting engine that replays historical candles without lookahead bias. It should support multi-timeframe analysis, partial exits (e.g., take half off at TP1, let the rest run to TP2), and trailing stops. Download enough extra data before the test period so indicators are warmed up. WARNING : don't forget to include fees in the backtest — many backtests look profitable until you add realistic fee accounting, especially on leveraged notional.

**Portfolio simulation**: Realistic simulation with proper fee accounting (maker vs taker fees on leveraged notional), drawdown tracking, and balance history. Fees should default to Bitget rates but be configurable.

**Reporting**: Detailed text reports for the LLM context, backtest stats with charts, and a decision log for audit.

**Scheduling**: A way to run the bot on a schedule and manage existing positions between checks (like updating trailing stops).

---

### 4. CONFIGURATION & SAFETY

Everything configurable goes in a single `config.json` — OpenWebUI connection, trading parameters, strategy settings, backtesting options, risk limits, fee rates, and exchange credentials. **Never hardcode API keys.**

**Safety rules (non-negotiable):**
- Never place an order without a stop loss
- Never hardcode credentials
- Never duplicate core logic across files — one source of truth
- Confidence should be bounded to a reasonable range (never 0% or 100%)
- Backtesting must never peek at future data
- All PnL must account for fees on the leveraged size

---

### 5. TECH & QUALITY

**Stack**: Python 3.13+. Key libraries: yfinance, pandas, numpy, pydantic, matplotlib, requests, schedule.

**Architecture**: Keep a clean separation of concerns — scoring engine, data fetching, automation controller, exchange integration, backtesting, and reporting should be separate modules. All core calculations live in one place, everything else imports from it. Cache OHLCV data briefly to avoid redundant API calls. Note that yfinance doesn't support 4H candles natively — you'll need to fetch 1H and aggregate to 4H. Also note that openwebui filters needs to be in a single separate file to be easily copy pasted into the OpenWebUI environment.

**Testing**: Write pytest tests covering the key behaviors — scoring, targets, filters, consensus, backtesting, portfolio simulation. Include a convenience script to run them.

**Docs**: Create an `AGENTS.md` that documents the architecture and development guidelines for future AI agents working on the project.

---

### 6. FREEDOM TO IMPROVE

This is a fresh start — you're not recreating the old project, you're building a better one. You have full freedom to:
- Design the scoring algorithm however you think works best
- Choose which indicators to include
- Rethink the consensus mechanism
- Organize the code however makes sense
- Add features you think would help

The only hard constraints are the safety rules and the functional requirements above. Everything else is your call — make it good.

---

## PROMPT END
