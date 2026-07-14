# Paper/Live Trading Readiness Review

**Review date:** 2026-07-13  
**Reviewed commit:** `8cc6912` (`main`)  
**Scope:** Architecture, Bitget integration, order lifecycle, risk controls, historical data,
backtest/live parity, testing, configuration, and operational readiness.

## Executive verdict

**Live trading: hard NO-GO.**  
**Bitget demo/paper trading: NO-GO until the critical execution blockers below are fixed.**

Credential-free dry-run and `analyze` mode are safe for observation, but dry-run does not
realistically simulate maker fills, partial exits, exchange stops, or restart recovery. Production
credentials should not be added yet.

The strategy research remains promising. The central problem is that the validated strategy is not
what the current live exchange path executes.

## What passed

- The tracked worktree was clean at the time of review.
- `339` tests passed.
- Python compilation and dependency consistency checks passed.
- The project has strong completed-candle causality tests, conservative SL-before-TP handling,
  shared trailing-stop math, and atomic state persistence.
- The Bitget demo header implementation is correct: `paptrading: 1`.
- No hardcoded or historically committed API credentials were found.
- `openwebui_filter.py` remains the scoring and indicator source of truth, imported by the typed
  project layer.

Test coverage was approximately 66%. More importantly, the real Bitget HTTP signing and payload
path is mocked rather than contract-tested, so the passing unit suite does not establish exchange
compatibility.

## Critical blockers

### 1. Private GET authentication is incorrectly signed

`llm_trading_bot/exchange.py::_request` signs only the request path and then sends GET parameters
separately. Bitget requires the exact `?queryString` in the signature. Balance, equity, positions,
pending orders, history, and order-detail calls should therefore fail documented authentication.

- Local code: `llm_trading_bot/exchange.py:141`
- Specification: <https://www.bitget.com/api-doc/common/signature>

### 2. Configured REST symbols do not match Bitget V2

The configs use `BTC-USDT`, `ETH-USDT`, and `SOL-USDT` for private REST requests. Bitget V2
documents `BTCUSDT`, `ETHUSDT`, and `SOLUSDT`. The separate CCXT market-data symbols are correct,
but are not used for private orders.

- Local config: `config.json:10`
- Specification: <https://www.bitget.com/api-doc/contract/trade/Place-Order>

### 3. No contract precision or minimum-size handling

Order size and target prices are submitted as raw floating-point strings. Bitget requires
symbol-specific size multiples, minimum quantity/notional, and price precision. BTC, ETH, and SOL
have different rules, so otherwise valid strategy orders can be rejected.

Contract metadata must be loaded dynamically and applied with `Decimal`, including conservative
rounding that never weakens the stop.

- Local code: `llm_trading_bot/exchange.py:228`
- Contract metadata: <https://www.bitget.com/api-doc/contract/market/Get-All-Symbols-Contracts>

### 4. Trailing stops are not actually modified

`modify_stop_loss` calls `place-tpsl-order` rather than `modify-tpsl-order`, stores no TPSL plan ID,
and supplies size for a position-level `pos_loss`. This cannot reliably move the existing attached
stop and may be rejected because a position stop already exists.

This is especially serious because the project identifies trailing as the dominant source of edge.

- Local code: `llm_trading_bot/exchange.py:306`
- Place TPSL: <https://www.bitget.com/api-doc/contract/plan/Place-Tpsl-Order>
- Modify TPSL: <https://www.bitget.com/api-doc/contract/plan/Modify-Tpsl-Order>

### 5. Live TP behavior does not reproduce the backtest

The backtest closes 70% at TP1, moves the stop to break-even, then manages the remainder toward TP2
or the trailing stop. Live placement sends only `presetStopSurplusPrice=TP1`. TP2 is logged and
stored but never submitted or executed, and live code never uses `tp1_exit_pct`.

No preset TP execution price is sent, which Bitget documents as market execution. The backtest
charges maker fees for TP. Live therefore has different exit size, lifecycle, and fee assumptions.

- Order placement: `llm_trading_bot/exchange.py:202`
- Backtest partial exits: `llm_trading_bot/backtesting.py:149`
- Bitget pending-order execution semantics:
  <https://www.bitget.com/api-doc/contract/trade/Get-Orders-Pending>

### 6. Position-mode close behavior is unsafe

`close_position` sends `sell/close` for a long and `buy/close` for a short. Bitget documents the
opposite convention in hedge mode: close-long is `buy/close` and close-short is `sell/close`.

In one-way mode, `tradeSide` is ignored, but the code does not set `reduceOnly=YES`; a stale or
oversized close can reverse the position. The bot neither sets nor verifies the account position
mode.

Pending-order parsing also prefers `posSide`; in one-way mode this is `net`, causing a pending buy
to be classified as short.

- Local close code: `llm_trading_bot/exchange.py:404`
- Pending parsing: `llm_trading_bot/exchange.py:362`
- Position-mode semantics: <https://www.bitget.com/api-doc/contract/trade/Place-Order>

### 7. Restart and timeout recovery are not idempotent

- New orders omit Bitget's `clientOid`.
- A timeout can leave an accepted but locally unrecorded order.
- The analysis bar is persisted as claimed before execution, so a transient failure consumes the
  opportunity.
- Startup reconciles only locally known pending orders; it does not adopt or cancel unknown exchange
  orders.
- Corrupt state silently resets all pending, analysis, peak, and trailing state.
- One tracked context per symbol cannot represent three independently pyramided trades.

Relevant code:

- `llm_trading_bot/scheduler.py:113`
- `llm_trading_bot/scheduler.py:128`
- `llm_trading_bot/scheduler.py:259`
- `llm_trading_bot/live_state.py:24`

### 8. Trailing can use pre-entry market extremes

New tracked trades have no initial `last_trail_bar`. A position check can immediately ratchet using
the decision bar that completed before entry. A maker fill can also use the entire fill bar's
high/low, including movement before the actual fill.

There is also an exact-key mismatch: tracked trades use config symbols such as `BTC-USDT`, while
Bitget positions normally return `BTCUSDT`. `_maybe_trail_stop` does not use the existing symbol
normalization helper.

- Local code: `llm_trading_bot/scheduler.py:608`

## Historical-data pagination defect

`llm_trading_bot/bitget_csv.py::fetch_ohlcv_range` assumes `[since, until)`. Bitget documents
candles as strictly *after* `startTime` and *before* `endTime`, with a maximum 90-day range. The
helper neither overlaps boundaries nor respects the 90-day limit for daily 200-candle windows.

- Local code: `llm_trading_bot/bitget_csv.py:64`
- API contract: <https://www.bitget.com/api-doc/contract/market/Get-History-Candle-Data>

The review reproduced:

- One missing daily candle at monthly boundaries from September 2022 onward.
- Thirty-four missing daily candles per asset inside the published 2021-June 2025 study.
- Seven missing 4-hour candles in the newly fetched period.
- Additional hourly gaps.
- Long daily fallback requests rejected for exceeding 90 days.

### Gap-free replay results

The following in-memory replay used overlapping, sub-90-day windows and the existing strategy
harness:

| Replay | Standard profile | Aggressive profile |
|---|---:|---:|
| Published 2021-June 2025 | 445,508x; 18.03% MTM DD | 4.976T x; 38.67% MTM DD |
| Gap-free 2021-June 2025 | 369,856x; 18.90% MTM DD | 4.829T x; 34.82% MTM DD |
| New June 2025-July 2026 holdout | 6.25x; 19.35% MTM DD; 1,216 trades | 29.78x; 32.51% MTM DD; 1,806 trades |

The post-June-2025 result is encouraging, but these path-dependent multiples are not forecasts and
still rely on the backtest execution model. The existing headline must be regenerated after
repairing pagination and invalidating the affected caches.

The historical `held_out_test` has also been consulted repeatedly while accepting and rejecting
later research rounds. It remains useful validation, but is no longer a pristine final holdout. The
new post-June-2025 period should now be frozen and not tuned against further.

## Backtest/live parity defects

### Missing live risk controls

Live scheduling lacks the configured post-SL cooldown and consecutive-loss threshold penalty used
by the full and fast backtests. `max_holding_hours` is also absent from live, although it is disabled
in the shipped profile.

### `max_position_usd` is live-only

`max_position_usd` is enforced at `llm_trading_bot/scheduler.py:504`, but neither the full nor fast
backtests use it. The standard `$100` margin cap makes the enormous long-run compounding headline
impossible to reproduce live once equity exceeds approximately `$5,000`.

### Margin and liquidation mismatch

Live orders use cross margin. The fast harness's liquidation model assumes isolated margin. Normal
`--mode backtest` has neither liquidation nor slippage modeling, even though the headline research
uses both.

### Risk capital mismatch

Backtests size and cap exposure using realized portfolio balance. Live combines equity and available
balance. Open PnL and reserved maker margin therefore change entries differently.

### Entry fees are excluded from trade outcome

The portfolio deducts the entry fee from account balance but does not include it in `Trade.net_pnl`.
The review reproduced a trade marked `+$0.009994` profitable while account equity actually lost
`$0.010006`.

This affects:

- Win rate and profit factor.
- Consecutive-loss penalties and cooldown decisions.
- Anti-martingale outcome streaks.
- Per-symbol net PnL.
- Reported total net PnL.

- Local code: `llm_trading_bot/portfolio.py:172`

### Drawdown reporting can understate MTM risk

Portfolio snapshots calculate drawdown but do not update `max_drawdown_pct`. The normal backtest
report therefore does not necessarily include the worst open-position mark-to-market drawdown. The
research harness calculates a separate MTM series, but normal CLI reports do not.

### Profile naming is misleading

The standard profile still selects the 25x `aggressive` leverage tier at `config.json:34`.
`standard` currently means portfolio-exposure capped, not conservative leverage. The separately
named aggressive profiles are 25x and uncapped.

## Data and configuration fail-open behavior

- `fetch_multi_timeframe` catches errors independently and returns any surviving timeframes.
- Scheduler indicator calculation also catches errors per timeframe.
- Routing falls back to the first available indicator set if the primary timeframe is missing.
- Missing ADX or ATR values bypass their corresponding pre-trade filters.
- The data cache key omits warmup length, market, and exchange.
- The Binance route does not consistently propagate the configured futures market.

This allows a degraded or incorrectly sourced strategy to trade rather than failing closed.

Relevant code:

- `llm_trading_bot/data.py:370`
- `llm_trading_bot/data.py:427`
- `llm_trading_bot/scheduler.py:191`
- `llm_trading_bot/routing.py:127`
- `llm_trading_bot/scoring.py:547`

## Configuration safety gaps

Pydantic validation does not enforce adequate bounds for:

- Leverage and risk percentages.
- Confidence `[5, 95]`.
- ATR and TP multipliers.
- TP1 exit fraction.
- Trailing activation/callback values.
- Fees.
- DD throttle configuration.
- Scheduling intervals.
- A valid `active_tier` key.

The exchange safety validator checks only that TP and SL are present and positive. It accepts NaN,
infinity, targets on the wrong side of entry, non-finite size, and non-finite entry price.

- Config models: `llm_trading_bot/config.py:16`
- Safety validation: `llm_trading_bot/exchange.py:168`

## Optional LLM consensus defects

Consensus is disabled in the shipped profiles, so these are not immediate production blockers:

- If consensus flips direction, scheduler mutates `scoring_result.direction` but keeps the original
  `targets.direction`; execution still follows the old targets.
- A two-model 1-1 tie can choose LONG at exactly 50%, despite the documented clear-majority rule.
- LLM confidences are clamped to `[0, 100]`, violating the global `[5, 95]` confidence invariant.

- Local code: `llm_trading_bot/openwebui_client.py:86`
- Consensus aggregation: `llm_trading_bot/openwebui_client.py:180`
- Scheduler integration: `llm_trading_bot/scheduler.py:339`

## Operational risks

- No centralized exchange rate limiter or account-snapshot cache.
- No retry/backoff policy, request session, or clock-drift preflight.
- Balance/equity calls occur outside the order-execution exception boundary.
- A scheduled job exception can escape the scheduler loop.
- No health heartbeat, alerting, log rotation, disk-space protection, or service definition.
- The process lock is tied to a log directory rather than an exchange account identity; another log
  directory bypasses it.
- Standalone single-symbol live mode has no process lock.
- There is no explicit production confirmation or startup preflight.
- Missing credentials silently activate dry-run rather than failing a requested paper/live startup.
- Dry-run always returns `dry_run_id` and does not simulate maker fills or exits.
- Dependencies have minimum versions but no lockfile or CI environment.
- `.gitignore` excludes only `config.local.json`, not per-symbol local secret configurations.

## Required remediation sequence

### Remediation status (updated 2026-07-13)

Current verification: **380 tests pass** after the fixes below. This status section supersedes the
original findings only where it explicitly says fixed; it does not change the NO-GO verdict.

- **In progress — market data:** Bitget requests now overlap strict page/month boundaries,
  constrain every request to 90 days or less, deduplicate and validate exact cadence/OHLCV values,
  and automatically discard incomplete legacy monthly caches. Regression tests cover strict
  boundaries, the daily 90-day constraint, and fail-closed gaps. Full cache rebuilding and result
  regeneration remain outstanding.
- **Partially fixed — Bitget adapter:** authenticated GETs now sign the exact encoded query string
  that is sent; all private REST methods canonicalize symbols; one-way closes are explicitly
  reduce-only; hedge closes use Bitget's documented close pairing; one-way pending orders no longer
  interpret `posSide=net` as short. Order safety now rejects non-finite values, invalid sizes, and
  targets on the wrong side. Contract metadata is now loaded and cached per symbol; order sizes,
  entries, stops, and targets use `Decimal` step rounding, with stops rounded only toward stronger
  protection, and quantity/notional/leverage limits fail closed before placement. Account and
  contract preflight are now implemented; exchange-observed demo validation remains open.
- **Partially fixed — idempotency:** entries now carry a deterministic account/symbol/decision-bar/
  action `clientOid`. An ambiguous transport failure is never retried as a POST; the adapter first
  queries the order by `clientOid` and adopts it if Bitget accepted it. Startup now adopts accepted
  bot orders and rejects unexplained exchange state; real demo crash fault-injection remains open.
- **Fixed in code — fail-closed data routing:** multi-timeframe fetches now fail the whole decision
  if any configured timeframe fails; completed live snapshots require every configured timeframe;
  indicator calculation failures abort the decision; primary-timeframe scoring no longer falls back;
  and missing ADX/ATR explicitly fails its risk filter. Cache identity now includes warmup, market,
  and exchange, and Binance consistently receives the configured market. End-to-end shadow testing
  remains an acceptance gate.
- **Fixed in code — accounting/reporting:** each trade's `net_pnl` and outcome classification now
  include its entry fee from inception, so streak logic and summary statistics use the true account
  outcome. Portfolio snapshots now update the reported maximum drawdown from mark-to-market equity.
  Corrected historical reports still need regeneration.
- **Partially fixed — configuration bounds:** Pydantic now rejects unsafe leverage, signal-threshold
  ordering, confidence outside `[5, 95]`, non-positive ATR/R:R values, invalid TP1 fractions,
  trailing percentages, fees, position risk/size, scheduling, cache TTL, market type, and invalid
  active-tier/timeframe references. Remaining risk-management bounds are still open.
- **Fixed in code — per-lot durable state:** live state version 3 keys independently pyramided lots
  by their deterministic entry identity and persists exact original/remaining quantity, aggregate
  fill price/fee/time plus available individual fills, targets, lifecycle phase, plan IDs, and the
  causal trailing boundary. Version-2 symbol state is rebuilt from exchange history. Malformed,
  unsupported, or unreadable state now blocks startup instead of silently resetting safety state.
- **Fixed in code — validated exit lifecycle:** entry orders retain mandatory immediate preset SL
  plus a full-position TP2 safety net. After a confirmed fill, deterministic client IDs establish
  explicitly sized per-lot `loss_plan`, 70% TP1 `profit_plan`, and TP2-remainder `profit_plan`
  orders. The preset plans are removed only after all replacements are observed active. An
  exchange-observed TP1 execution resizes the remainder plans and modifies the existing stop to
  break-even. SL/TP2 execution and signal-flip closes cancel sibling plans before lot retirement.
- **Fixed in code — TPSL discovery and trailing:** current/history plan queries adopt TPSL IDs by
  deterministic client ID after fills, ambiguous responses, and restarts. Trailing modifies each
  lot's existing sized loss plan and still ratchets once per eligible completed primary bar.
- **Fixed in code — startup reconciliation and account preflight:** paper/live now requires explicit
  credentials, authenticated account access, ≤30-second clock drift, a tradable contract, and exact
  configured position/margin modes. Before analysis, exchange normal orders, positions, recent
  order history, and TPSL plans are reconciled against local state. Bot-owned `llt-*` entries can be
  adopted after a crash; unexplained orders, positions, plans, symbols, size mismatches, or
  unverified protection fail startup closed. A disk persistence failure invalidates reconciliation
  so the next cycle must recover from exchange state.
- **Still open:** corrected cache/report regeneration, remaining live/backtest risk parity,
  exchange contract/demo integration, timeout/crash fault injection against Bitget demo, and the
  operational controls below. Demo and live trading therefore remain **NO-GO**.

### 1. Repair and revalidate market data

- Use overlapping pages so boundary candles cannot be lost.
- Enforce Bitget's 90-day request span in addition to the candle-count limit.
- Validate exact expected cadence, duplicates, monotonic indexes, OHLC validity, and freshness.
- Fail closed if a required timeframe is missing or incomplete.
- Invalidate affected Bitget caches.
- Regenerate the completed-candle result files and documentation.

### 2. Rebuild the Bitget adapter

- Sign the exact URL-encoded query string sent on GET.
- Canonicalize private REST symbols.
- Fetch contract precision/minimums and use `Decimal` quantization.
- Explicitly set and verify position and margin mode.
- Correct close-side/reduce-only semantics.
- Add deterministic `clientOid` idempotency keyed by account, symbol, decision bar, and action.
- Implement appropriate retries only where they cannot duplicate state.
- Add time synchronization and API permission preflight.

### 3. Implement the validated order lifecycle

- Verify SL and TP protection after every fill.
- Implement a real 70% TP1 partial exit.
- Move the remaining stop to break-even.
- Place/manage the TP2 remainder.
- Modify the existing stop using its TPSL plan ID.
- Record exact fill quantity, price, fee, and timestamp.
- Initialize trailing eligibility from the actual exposure/fill time.
- Maintain per-lot lifecycle state for pyramiding, or intentionally aggregate positions and revalidate
  that different strategy.
- Reconcile exchange positions, orders, fills, and TPSL plans at startup before analysis can run.

### 4. Restore strategy and accounting parity

- Implement live cooldown and consecutive-loss penalty behavior.
- Include entry fees in trade outcomes and all derived statistics.
- Apply `max_position_usd` consistently to every simulator or remove it from live parity claims.
- Align balance/equity sizing rules.
- Align cross/isolated margin and liquidation assumptions.
- Add the research harness's slippage/liquidation/MTM reporting to normal backtests.
- Revalidate TP fee assumptions against the exact live order types.

### 5. Add demo integration and operational controls

Required Bitget demo scenarios:

- Authenticated GET and POST requests.
- Contract precision and minimum-size rejection prevention.
- Maker order fill, partial fill, expiry, and fill/cancel race.
- TP1 partial exit, TP2 remainder, and break-even stop.
- Existing-stop modification and stop monotonicity.
- Signal-flip close in the configured position mode.
- Timeout after exchange acceptance.
- Crash between acceptance and local persistence.
- Corrupt/missing state recovery.
- Unknown exchange order/position adoption.
- Restart without duplicate execution of a claimed bar.

Operational controls should include structured metrics, a heartbeat, alerts, log rotation, disk-space
checks, a supervisor/service definition, an account-scoped lock, API keys with no withdrawal
permission, IP restrictions, and an external kill switch.

## Paper-trading acceptance gates

Do not advance from demo/paper until all of the following hold:

- Zero precision or minimum-size rejects.
- Zero duplicate orders and zero orphan orders.
- Every open position continuously has verified active SL and TP protection.
- TP1 partial, TP2 remainder, and trailing behavior are observed on the exchange.
- Trailing ratchets exactly once after each eligible completed 4-hour bar.
- Restart and timeout fault-injection tests produce no duplicate or unmanaged exposure.
- Live shadow decisions match the corrected backtest bar by bar.
- Exchange fills, fees, funding, PnL, and stop changes reconcile against local records.
- At least 8-12 weeks and roughly 100 completed trades across BTC, ETH, and SOL.

Only after these gates pass should tiny live capital be considered. The first live deployment should
use the capped policy, withdrawal-disabled/IP-restricted keys, monitoring, and an external kill
switch. The uncapped aggressive profile should not be the first live deployment.
