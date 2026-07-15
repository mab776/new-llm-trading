# Paper/Live Trading Readiness Review

**Original review:** 2026-07-13 (commit `8cc6912`, `main`)
**Last updated:** 2026-07-15 — all credential-free prerequisites done; rolling-VWAP
reproducibility fix + LLM-consensus bug fixes applied; **403 tests pass**.
**Only remaining gate:** item 1, Bitget demo integration (needs Marc's demo API keys).
**Scope:** Architecture, Bitget integration, order lifecycle, risk controls, historical data,
backtest/live parity, testing, configuration, operational readiness, and the remaining work to
reach paper trading.

> This is now the single source of truth for paper/live readiness. It absorbs the former
> `LATEST_SESSION.md` and `NEXT_SESSION.md` handoffs (both deletable once this is committed).
> **Goal: reach Bitget demo/paper trading ASAP.** The blocker list in
> [Remaining work before paper trading](#remaining-work-before-paper-trading-do-these-then-start)
> is the actionable to-do; everything below it is the detailed audit that produced it.

---

## Executive verdict

**Live trading (real money): hard NO-GO.**
**Bitget demo/paper trading: NO-GO until item 1 (exchange-observed demo validation) passes — every
credential-free prerequisite is now complete.**

Almost every critical execution blocker from the original review has been **fixed in code**
(authenticated signing, symbol canonicalization, contract precision, real TP1/TP2/break-even
lifecycle, per-lot durable state, startup reconciliation, account preflight, fail-closed data
routing, entry-fee accounting). **2026-07-15 update:** items 2–4 below (cache/report regeneration,
risk/accounting parity, operational controls + log instrumentation) are **DONE** — 403 tests pass
and full/fast parity is exact at the new execution settings. The only gate left before paper
trading is **item 1: running the lifecycle scenarios against the Bitget demo account** (plus the
one-time demo account setup listed there).

Credential-free dry-run and `analyze` mode remain safe for observation. Dry-run still does **not**
realistically simulate maker fills, partial exits, exchange stops, or restart recovery, so passing
unit tests do not yet establish exchange compatibility. **No paper/testnet/live process has ever
been started.** Production credentials must not be added until the items below are cleared and Marc
gives explicit approval.

---

## Remaining work before paper trading (do these, then start)

**Status 2026-07-15: items 2, 3, and 4 are DONE (see the per-item notes). Item 1 is the only
remaining gate and requires the demo API keys.**

### 1. Bitget demo integration (the real gate — everything else is mocked) — ⏳ OPEN

One-time demo account setup before the scenarios (the bot fails closed if these don't match):

- Create demo API keys (no withdrawal permission; IP-restricted if available) and put them in a
  git-ignored local config (e.g. `config.local.json` / `config-btc.local.json` with `_extends`).
- Set the demo futures account to **one-way position mode** and **isolated margin** — the shipped
  configs now pin `position_mode: one_way` + `margin_mode: isolated` (isolated matches the
  research harness's liquidation model) and `preflight()` refuses to start on a mismatch.

The full HTTP signing/payload path is unit-tested against mocks only. It must be exercised against
the **Bitget demo endpoint** (`paptrading: 1` header, `bitget.testnet: true`). Run and observe each
scenario on the exchange:

- Authenticated GET and POST requests succeed (balance, positions, pending, history, place, modify).
- Contract precision / minimum-size: zero rejects across BTC, ETH, SOL.
- Maker order: fill, partial fill, expiry, and fill/cancel race.
- TP1 partial exit (70%), TP2 remainder, and break-even stop move — all observed on the exchange.
- Existing-stop modification uses the stored TPSL plan ID and never weakens the stop.
- Signal-flip close in the configured (one-way) position mode; `reduceOnly` respected.
- Timeout **after** exchange acceptance → no duplicate on retry (clientOid adoption path).
- Crash between acceptance and local persistence → recovered from exchange state on restart.
- Corrupt/missing state → fail-closed, then recover from exchange.
- Unknown exchange order/position adoption vs rejection.
- Restart without duplicate execution of an already-claimed bar.

### 2. Regenerate corrected caches and reports — ✅ DONE 2026-07-15

- Every cached Bitget month passed the strict gap-free validation on load (the caches had already
  been rebuilt by the fixed fetcher; nothing needed refetching).
- `opt/completed_candle_results.json` and `opt/completed_candle_queue_results.json` regenerated
  **with the full live execution model** (see item 3). Headline (gap-free, sub-bar, maker,
  funding, liquidation, 2bps slip, 66% per-trade margin rail, 2021-01→2025-06):
  - **Standard:** continuous **842,919.58×**, 18.77% reported / 18.85% MTM maxDD; held-out TEST
    216.76×; standalone BTC/ETH/SOL 303.97× / 2,755.59× / 430,108.74× (the huge SOL dispersion
    underlines that these are path-dependent, not a portable per-asset edge); every annual fold green.
  - **Aggressive:** continuous **72.9 trillion×**, 36.17% reported / 35.32% MTM maxDD; TEST
    3,348,599×.
  - (2026-07-15 later: regenerated after the rolling-VWAP reproducibility fix — see change log.)
  - **Queue stress:** harsh 5bps penetration + 70% fill retains **64.6%** median log growth,
    every annual fold green (worst +398.6%).
  - Sizing-rail history: the first 2026-07-15 regeneration enforced the then-shipped absolute
    `$100` `max_position_usd` (standard collapsed to 1,053.88× — proof the old 445k× headline
    was never live-reproducible with that cap). Marc then replaced the absolute cap with the
    scale-invariant `max_position_pct: 0.66`, which normal ~2-3% sizing never reaches, so the
    strategy compounds unconstrained again and the rail only guards against sizing bugs.
- `README.md` and `AGENTS.md` updated with the new numbers.
- `opt/lower_timeframe_results.json` was NOT regenerated: it is a rejected research artifact
  (1h/5m transplants), not on the paper-trading path.
- The post-June-2025 holdout stays frozen; nothing was tuned against it.

### 3. Close the last risk/accounting parity gaps — ✅ DONE 2026-07-15

- **Live cooldown + consecutive-loss penalty implemented** (they were missing, not just
  unverified): per-symbol counters persisted in live state **v4** tick once per completed primary
  bar (with downtime catch-up), the penalty raises the effective entry thresholds and the
  conviction-sizing normalizer exactly like the engine, a losing SL-family close arms the
  cooldown, and `COOLDOWN_SKIP` blocks entries. Lot outcomes are classified with the same
  recorded-price/fee convention the backtest uses.
- **Per-trade margin rail enforced in every simulator** (full engine, fastbt, shared
  multi-asset harness) via one shared `Portfolio` sizing point + `PendingEntry.max_margin_pct`,
  so live and research sizing can no longer diverge. The rail is `max_position_pct: 0.66`
  (fraction of the sizing balance — replaced the absolute `max_position_usd`, which froze
  compounding once equity outgrew it).
- **Sizing basis aligned:** live now sizes and caps on **realized balance** (equity − open PnL,
  matching the backtests' realized `Portfolio.balance`), still bounded by available balance
  because reserved maker margin cannot be committed twice.
- **Margin model aligned:** shipped configs switched to `margin_mode: "isolated"`, matching the
  harness's isolated-margin liquidation model (and bounding per-position loss). Preflight
  enforces it; set the demo account to isolated (item 1).
- **`--mode backtest` gained the research realism:** config-driven `slippage_pct` (market fills
  only) and `model_liquidation`/`maintenance_margin`, mirroring `opt/fastbt` semantics exactly;
  `config.json` ships 2bps + liquidation on. MTM drawdown reporting was already in snapshots.
- **`max_holding_hours` implemented in live** (bar-floored force-close mirroring `time_expired`;
  shipped configs keep it disabled).
- **Full/fast 2024 maker parity re-verified digit-equal (rolling-VWAP settings):** +177.54%, 590
  trades, 16.74% maxDD, zero mismatches (`opt/validate_parity.py` now mirrors the config's
  execution settings into fastbt).
- ⚠️ **One known, deliberate divergence remains:** live TP1/TP2 plans execute at **market**
  (taker fee + real execution price) while the backtest fills TPs at the exact target with
  maker fee (`use_maker_fee_for_tp: true`). Options: flip the config to taker-for-TP, or switch
  the live plans to limit execution. **Decide after measuring actual TP executions on demo**
  (item 1) — this is the drift metric to watch first.

### 4. Operational controls + log instrumentation — ✅ mostly DONE 2026-07-15

**Logging (explicitly requested) — DONE:**

- **One file per local day:** `logs/trading-YYYY-MM-DD.log` (human-readable, local timestamps,
  symbol-tagged) and `logs/decisions-YYYY-MM-DD.jsonl` (structured stream). The old unbounded
  `trading.log`/`decisions.jsonl` appends are gone. (Decision *bars* stay UTC-aligned — that is
  exchange reality, not a logging choice.)
- **90-day retention** (config: `scheduling.log_retention_days`): dated files older than the
  window are deleted on startup and at each day rollover; files without a parsable date
  suffix are never touched.
- **Structured, evaluation-ready records:** every record carries `symbol` + a local `timestamp`
  with explicit UTC offset;
  placements carry decision bar, targets, size/margin/risk_pct, leverage, score/confidence,
  loss penalty, equity/available/realized/peak balances; fills carry size/fee/fill-time;
  `TP1_PARTIAL` carries price + break-even move; `LOT_CLOSED` carries reason, exit price, and an
  estimated realized net PnL (entry fee included); `TRAIL_RATCHET` is logged once per completed
  4h bar; `COOLDOWN_SKIP`/`WAIT` capture why an entry did not happen; a `HEARTBEAT` record every
  position check (15 min) snapshots equity, realized/peak balance, open lots, pending orders,
  cooldown state, and free disk (with a loud low-disk warning). This is the per-trade
  live-vs-backtest drift dataset.
- Still to wire once paper runs: Grafana dashboard over `logs/decisions-*.jsonl` on the
  Portainer server.

**Other operational controls — DONE in code:**

- **Account-scoped process lock** (`llm_trading_bot/process_lock.py`): keyed on the exchange
  account (api key + demo flag + product type), lives in the system temp dir, enforced by BOTH
  the standalone scheduler and the shared orchestrator — a second process on the same account is
  rejected regardless of log directory.
- **Scheduler-loop exception guard** in both loops: a crashing job is logged and the loop
  survives; startup reconciliation failures still abort loudly before any trading.
- Fail-loud on missing credentials for paper/live (already shipped: `preflight()` refuses
  credential-free startup), clock-drift preflight already runs at every startup reconciliation.
- `.gitignore` extended to cover per-symbol local secret configs (`config-*.local.json`,
  `*.local.json`, `*.secret.json`).

**Deferred to the demo/paper deployment (not code):**

- Native Portainer stack + Grafana dashboard (standing preference in
  `~/Documents/portainer/CLAUDE.md`), external kill switch, alerting on missing heartbeat,
  API keys without withdrawal permission + IP restriction, centralized rate limiter /
  retry-backoff session hardening if demo shows API pressure.

### How to start paper trading (once 1–4 are cleared and Marc approves)

Use **one shared orchestrator process**, not independent per-symbol stacks:

```bash
python -m llm_trading_bot.main --mode live --shared-configs \
  config-aggressive.json config-eth-aggressive.json config-sol-aggressive.json
```

⚠️ **This command trades immediately.** Supply Bitget demo credentials and get explicit approval
first. Deploy the single orchestrator as a native Portainer stack with Grafana over
`logs/decisions.jsonl`, then measure actual maker fill rate, bar-close latency, and
live-vs-backtest drift. The first run should use the **capped standard policy**, not the uncapped
aggressive profile.

### Paper-trading acceptance gates (do not advance to real money until all hold)

- Zero precision or minimum-size rejects.
- Zero duplicate orders and zero orphan orders.
- Every open position continuously has verified active SL and TP protection.
- TP1 partial, TP2 remainder, and trailing behavior observed on the exchange.
- Trailing ratchets exactly once after each eligible completed 4-hour bar.
- Restart and timeout fault-injection produce no duplicate or unmanaged exposure.
- Live shadow decisions match the corrected backtest bar by bar.
- Exchange fills, fees, funding, PnL, and stop changes reconcile against local records.
- At least 8–12 weeks and roughly 100 completed trades across BTC, ETH, and SOL.

Only after these gates pass should tiny live capital be considered, with withdrawal-disabled/
IP-restricted keys, monitoring, and the external kill switch.

---

## Strategy context (self-contained — carried over from the round handoffs)

The strategy research is promising; the central problem was never the edge, it was that the
validated strategy is not yet what the live exchange path provably executes.

- **Standard capped profile (default):** BTC+ETH+SOL shared continuous **842,919.58×** at 18.77%
  reported / 18.85% MTM maxDD (2026-07-15 regeneration with the full live execution model plus the
  rolling-VWAP reproducibility fix: gap-free data, 66% per-trade margin rail, slippage, liquidation);
  held-out TEST 216.76×, every
  annual fold green; 4.4% equity-margin and 1.10× equity-notional caps. Configs: `config.json`,
  `config-eth.json`, `config-sol.json`. **Acceptance targets ~25% maxDD** — a research/selection
  target, NOT a live kill switch.
- **Aggressive profile (explicit opt-in):** `config-aggressive.json` + ETH/SOL peers disable the
  shared caps for the uncapped anti-martingale return path: continuous **72.9 trillion×** at
  36.17% reported / 35.32% MTM maxDD, TEST 3,348,599×. The huge multiples are **path-dependent,
  not forecasts** (no market-impact modeling); live DD can be materially worse.
- The post-June-2025 holdout (standard ~6.25×, aggressive ~29.78×, pre-cap replay) stays frozen
  and untouched; it still relies on the backtest execution model.
- **Strategy internals (unchanged, shipped):** 4h primary, score→route→trade; trailing stops
  (act 0.94% / cb 0.33%, ratchet once per completed 4h bar); pyramiding (max 3, same-direction);
  conviction sizing (exp 1.0); opposite-signal exit (threshold 20); DD circuit-breaker
  (25% → 1 slot, risk×0.5); lev 25 aggressive / 12 conservative tier; ATR stop 2.26×;
  TP RR 2.02/3.34 (70% @ TP1). Maker entry shipped. LLM consensus is explicit opt-in only.
- **Non-negotiable methodology** (each rule caught a real bug): never trust an in-sample max (select
  on TRAIN, report held-out TEST + chrono); intrabar = worst case (SL before TP in one bar); trailing
  ratchets ONCE per completed 4h bar (hourly ratcheting collapses the edge ~84×→5×); after any engine
  change re-verify engine==fastbt digit-equal; keep engine + `openwebui_filter.py` + scheduler in
  sync; never mix risk profiles in one report.
- **Harness:** `opt/fastbt.py` (causal, digit-equal to the engine, ~4000× faster) + `opt/driver.py`
  (`setup`/`evaluate`, folds). Run with `PYTHONPATH=. /tmp/tmlvenv/bin/python …`; the `/tmp/tmlvenv`
  venv has all deps (system python has no pip). `pytest -q` → 403 tests pass. Parity checker:
  `PYTHONPATH=. /tmp/tmlvenv/bin/python -m opt.validate_parity` (must stay exact).

---

## What passed (audit)

- The tracked worktree was clean at the time of review; Python compilation and dependency checks pass.
- Strong completed-candle causality tests, conservative SL-before-TP handling, shared trailing-stop
  math, and atomic state persistence.
- Bitget demo header implementation is correct: `paptrading: 1`.
- No hardcoded or historically committed API credentials.
- `openwebui_filter.py` remains the scoring/indicator source of truth, imported by the typed layer.

Coverage was ~66%, and the real Bitget HTTP signing/payload path is mocked rather than
contract-tested — hence item 1 above.

---

## Critical blockers (original findings — status after the foundation work)

All numbered items below were fixed in code as part of the live-execution foundation
(380 tests pass); they remain here as the **verification checklist for Bitget demo (item 1)**, since
"fixed in code" ≠ "observed on the exchange."

### 1. Private GET authentication signing — **fixed in code**

Authenticated GETs now sign the exact URL-encoded query string that is sent.
- `llm_trading_bot/exchange.py` · spec <https://www.bitget.com/api-doc/common/signature>

### 2. REST symbol mismatch — **fixed in code**

All private REST methods canonicalize `BTC-USDT` → `BTCUSDT` etc.
- `config.json:10` · spec <https://www.bitget.com/api-doc/contract/trade/Place-Order>

### 3. Contract precision / minimum size — **fixed in code**

Contract metadata is loaded and cached per symbol; sizes/entries/stops/targets use `Decimal` step
rounding (stops only toward stronger protection); qty/notional/leverage limits fail closed.
- `llm_trading_bot/exchange.py` · <https://www.bitget.com/api-doc/contract/market/Get-All-Symbols-Contracts>

### 4. Trailing stops not actually modified — **fixed in code**

Trailing now modifies each lot's existing sized loss plan via its TPSL plan ID and ratchets once per
eligible completed primary bar.
- Place/Modify TPSL: <https://www.bitget.com/api-doc/contract/plan/Modify-Tpsl-Order>

### 5. Live TP did not reproduce the backtest — **fixed in code**

After a confirmed fill, deterministic client IDs establish sized per-lot `loss_plan`, 70% TP1
`profit_plan`, and TP2-remainder `profit_plan`. An observed TP1 execution resizes the remainder and
moves the stop to break-even. ⚠️ Live TP plans currently execute at **market** for reliable exits —
TP fee parity vs the backtest's maker-fee assumption is an open validation item (parity item 3).

### 6. Position-mode close semantics — **fixed in code**

One-way closes are explicitly `reduceOnly`; hedge closes use Bitget's documented pairing; one-way
pending orders no longer interpret `posSide=net` as short. Account preflight verifies position/margin
mode.

### 7. Restart/timeout idempotency — **fixed in code**

Entries carry a deterministic account/symbol/decision-bar/action `clientOid`; an ambiguous transport
failure is queried by `clientOid` and adopted rather than re-POSTed; startup adopts accepted bot
orders and rejects unexplained exchange state. Live state v3 keys independently pyramided lots.
**Real Bitget-demo crash/timeout fault-injection is still open (item 1).**

### 8. Trailing using pre-entry extremes — **fixed in code**

Per-lot trailing eligibility now initializes from actual fill time; symbol keys are normalized so
`BTC-USDT` tracked trades match `BTCUSDT` positions.

---

## Historical-data pagination defect — **fixed; caches verified gap-free 2026-07-15**

`bitget_csv.py::fetch_ohlcv_range` now overlaps strict page/month boundaries, constrains every
request to ≤90 days, deduplicates, and validates exact cadence/OHLC values, discarding incomplete
legacy monthly caches. The original bug lost candles at monthly boundaries (34 missing daily candles
per asset across 2021–June-2025, plus 4h/hourly gaps) and rejected long daily windows. The 2026-07-15
regeneration loaded every cached month through the strict validator with zero refetches and zero
gaps, and the result files were rebuilt from it (item 2).

Gap-free replay reference (still backtest-model-dependent, not a forecast):

| Replay | Standard | Aggressive |
|---|---:|---:|
| Published 2021–Jun 2025 | 445,508×; 18.03% MTM DD | 4.976T×; 38.67% MTM DD |
| Gap-free 2021–Jun 2025 | 369,856×; 18.90% MTM DD | 4.829T×; 34.82% MTM DD |
| New Jun 2025–Jul 2026 holdout | 6.25×; 19.35% MTM DD; 1,216 trades | 29.78×; 32.51% MTM DD; 1,806 trades |

The `held_out_test` split has been consulted across many rounds and is no longer a pristine holdout;
freeze the new post-June-2025 period instead.

---

## Backtest/live parity defects (audit detail — all resolved 2026-07-15 except the TP-fee note)

- **Live risk controls — FIXED:** cooldown + consecutive-loss penalty + `max_holding_hours` now
  implemented in live with persisted per-symbol counters (see item 3).
- **Per-trade margin cap — FIXED:** now `max_position_pct: 0.66` (scale-invariant fraction),
  enforced identically in live, the full engine, fastbt, and the shared harness; headlines
  regenerated with it.
- **Margin/liquidation — FIXED:** shipped configs now use isolated margin (harness model);
  `--mode backtest` models slippage + isolated-margin liquidation from config.
- **Risk-capital — FIXED:** live sizes and caps on realized balance (equity − open PnL).
- **Entry fees — fixed in code:** each trade's `net_pnl` and outcome include the entry fee from
  inception (the review had reproduced a `+$0.009994` trade whose equity actually lost `$0.010006`),
  so streaks, win rate, profit factor, anti-martingale, and totals use the true outcome.
- **Drawdown reporting — fixed in code:** portfolio snapshots update `max_drawdown_pct` from
  mark-to-market equity.
- **TP fee/execution — OPEN (deliberate):** live TP plans execute at market (taker) vs the
  backtest's exact-price maker fill; decide config-flip vs limit-execution after demo measurement.
- **Profile naming:** "standard" still selects the 25× leverage tier (`config.json`); it means
  exposure-capped, not conservative leverage. Keep this clear in any report.

---

## Data/config fail-open behavior — **fixed in code**

Multi-timeframe fetches now fail the whole decision if any configured timeframe fails; completed live
snapshots require every timeframe; indicator failures abort the decision; primary-timeframe scoring no
longer falls back; missing ADX/ATR explicitly fails its filter. Cache identity now includes warmup,
market, and exchange, and Binance consistently receives the configured market. **End-to-end shadow
testing remains an acceptance gate** (item 1 / acceptance gates).

## Configuration safety — **fixed in code (bounds), some open**

Pydantic now rejects unsafe leverage, signal-threshold ordering, confidence outside `[5, 95]`,
non-positive ATR/R:R, invalid TP1 fractions, trailing percentages, fees, position risk/size,
scheduling, cache TTL, market type, and invalid active-tier/timeframe references. The exchange safety
validator now rejects non-finite values, invalid sizes, and wrong-side targets. **A few remaining
risk-management bounds are still open.**

## Optional LLM consensus defects (not blockers — consensus disabled in shipped profiles)

- Consensus flip mutates `scoring_result.direction` but leaves `targets.direction`; execution follows
  the old targets.
- A two-model 1-1 tie can pick LONG at exactly 50% despite the clear-majority rule.
- LLM confidences clamped to `[0, 100]`, violating the global `[5, 95]` invariant.
- `openwebui_client.py:86` / `:180` / `scheduler.py:339`.

---

## Change log

- **2026-07-15 (pre-paper deep review — fixes applied):**
  - **VWAP made live-reproducible (parity bug).** `openwebui_filter.compute_vwap` was a
    cumulative-from-series-start VWAP whose value depends on how much history is loaded — the
    backtest anchored it years back, but the live loader fetches only ~`warmup_periods+30`
    candles, so the SAME bar scored differently live vs backtest (the `volume` category, weight
    0.20, silently diverged on every decision). Replaced with a fixed 100-bar rolling VWAP
    (causal; identical in engine, fast harness, and live). The edge survives and improves:
    standard 338,592×→**842,919×** (TEST 153.65→**216.76×**), aggressive 4.67T→**72.9T×**;
    every annual fold green; engine==fastbt parity re-verified **exact**. `opt/completed_candle_results.json`
    + README/AGENTS regenerated. (`opt/queue_fill_sensitivity_results.json` predates this and
    should be rerun.) Window is `openwebui_filter.VWAP_WINDOW` if it ever needs tuning.
  - **LLM-consensus correctness fixes (opt-in path; shipped profiles use `deterministic`):**
    (1) a consensus decision that *disagrees* with the scored direction now WAITs instead of
    silently executing the old (opposite-direction) targets with unverified filters
    (`scheduler.execute_decision`); (2) an even vote split (e.g. a two-model 1 LONG/1 SHORT tie =
    exactly 50%) now resolves to WAIT, not whichever side sorts first (`build_consensus`, `<= 50`);
    (3) parsed LLM confidence clamps to the system invariant `[5, 95]` instead of `[0, 100]`.
    Regression tests added/updated; **403 tests pass**.
  - **DD under-sampling: investigated, no change.** The continuous headline DD is already marked
    every bar via the independent `equity_curve`/`analyze_drawdowns` (the 18.85%/35.32% MTM
    figures). The every-10-bar `record_snapshot` cadence also drives `peak_balance` (hence the
    DD-throttle), so changing it would alter trades and break parity — left as-is deliberately.
  - **Still demo-gated (unchanged, need Bitget demo to calibrate):** maker fill rate (the sim
    fills ~99.9% of touched limits), TP maker-fee-vs-live-market execution, and market-exit
    slippage magnitude (2 bps is thin for 25× stop-outs). These remain the first drift metrics to
    measure on demo before trusting any headline.
- **2026-07-15 (later):** Replaced the absolute per-trade margin cap (`max_position_usd`, $100
  standard / $1B aggressive) with the scale-invariant **`max_position_pct: 0.66`** everywhere
  (live + all simulators + all configs; aggressive overrides deleted — they inherit the rail).
  Normal ~2-3% sizing never reaches it, so compounding is unconstrained and the rail only stops
  runaway size computations. Logs switched to LOCAL-day files with 90-day retention. Results
  regenerated: standard 338,592.06× (17.94%/18.02% DD), aggressive 4.67T× (35.03%/34.82% DD),
  queue harsh-case retention 64.6%; parity still exact; 402 tests pass.
- **2026-07-15:** Completed every credential-free prerequisite (items 2–4). **Logging:** daily
  `trading-YYYY-MM-DD.log` + `decisions-YYYY-MM-DD.jsonl` with configurable 30-day retention,
  UTC timestamps, and evaluation-ready structured records (placements with sizing/account
  snapshot, fills with fees, TP1/close events with estimated realized PnL, TRAIL_RATCHET,
  COOLDOWN_SKIP, HEARTBEAT with disk check). **Parity:** live cooldown/consecutive-loss
  penalty/max-holding implemented (live state v4); `max_position_usd` enforced in all three
  simulators; realized-balance sizing in live; isolated margin in shipped configs; engine
  slippage+liquidation from config; full/fast 2024 maker parity re-verified exact at the new
  settings (+150.42%, 571 trades, zero mismatches). **Ops:** account-scoped process lock (both
  entry points), scheduler-loop crash guard, extended secret ignores. **Results regenerated**
  from verified gap-free caches with the full execution model: standard continuous 1,053.88×
  (14.36%/13.16% DD, TEST 153.65×), aggressive 5.20B× (35.03%/34.82% DD, TEST 1,486,262×),
  queue harsh-case retention 84.2%. README/AGENTS updated. **401 tests pass.** Remaining gate:
  item 1 (Bitget demo integration) — needs the demo API keys.
- **2026-07-14:** Consolidated `LATEST_SESSION.md` + `NEXT_SESSION.md` into this document. Recorded
  the live-execution foundation (blockers 1–8 fixed in code, per-lot lifecycle/state, startup
  reconciliation, account preflight; 380 tests pass). Added the actionable
  "Remaining work before paper trading" section, the log-instrumentation task (daily rotation +
  30-day retention + evaluation-ready structured records), and the paper-trading start command and
  acceptance gates. Verdict unchanged: demo/paper NO-GO until items 1–4 clear.
- **2026-07-13:** Original readiness review at commit `8cc6912`.
