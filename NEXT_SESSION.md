# Prompt for the next optimization session

Copy-paste everything below the line into a fresh Claude Code session started in
`~/Documents/new-llm-trading`.

---

## LATEST handoff — Round 22 lower-timeframe audit (2026-07-13)

This section supersedes the earlier "pre-paper backlog complete" / "paper is the only next task"
statement below. No paper/testnet/live process has been started.

- Research branch: `experiment/lower-timeframes`.
- A static, leakage-free BTC transplant used common Binance USDT-perpetual data for all cadences,
  the shipped numeric parameters, maker entry, funding, liquidation, and 2bps market-exit slip.
  The causal 4h control produced **225.91× / 15.80% DD**, 1h produced **76.94× / 28.06% DD**
  with 2025H1 **-6.76%**, and 5m produced **0.237× / 79.26% DD** with every annual fold losing.
- The pre-existing 12× risk dial reduced 1h continuous DD to 15.11% and returned 9.02×, but did
  not repair robustness: 2025H1 remained negative at -5.64%. It is not a deployable winner.
- 5m has only a thin gross edge: removing fees, slippage, and funding yields 4.06× / 30.40% DD,
  but realistic costs turn it into a 76% capital loss. Funding is minor; fee/slippage drag against
  much smaller moves is decisive. Do not retune 5m from this evidence.
- **A pre-paper causality blocker was discovered:** Bitget candles are stamped at bar open, while
  full/fast backtests currently choose secondary rows with `secondary_open <= primary_open`.
  That exposes higher-timeframe OHLCV before the secondary candle completes. On the identical
  native Bitget BTC run, legacy alignment reproduces **301.18×**, while last-completed alignment
  gives **204.21× / 17.93% DD**. The edge survives, but the current headline is overstated.
- Live `analyze_market()` also calculates signals from the latest possibly forming candles and the
  scheduler runs hourly even for a 4h primary. That is not backtest parity and can re-evaluate the
  same primary bar. Do not start paper trading until close-aware slicing plus a persisted
  once-per-completed-primary-bar decision gate are implemented and revalidated in full/fast/live.
- Reproducible artifact: `opt/lower_timeframe_results.json`; runner:
  `PYTHONPATH=. /tmp/tmlvenv/bin/python -m opt.lower_timeframes`.
- Verification: **333 tests pass**. Production strategy/config/runtime files were not changed on
  this research branch.

### Next task before paper trading

Fix completed-candle alignment consistently in the full engine, fastbt, analysis/live scheduler,
and Binance timestamp normalization; add regression tests; then rerun engine↔fast parity and the
standard/aggressive BTC+ETH+SOL shared validation. Only after the corrected results are accepted is
paper trading again the next externally visible task.

---

## Prior handoff — pre-paper backlog complete before Round 22 audit (2026-07-13)

This section supersedes older Round 16/17 numbers and backlog text below. No paper/testnet/live
process has been started.

- **Round 18 fixed a sub-bar harness cadence bug:** `exit_granularity="sub"` had ratcheted trailing
  after each 1h bar. It now replays 1h exits with the stop fixed intrabar and ratchets exactly once
  after the completed 4h bar, matching engine/live strategy cadence.
- Corrected standard shared continuous: **292,212.44×**, 19.95% reported / 20.67% independent 4h
  mark-to-market maxDD. Standalone BTC/ETH/SOL: 301.18× / 2,436.13× / 66,125.23×.
- Corrected aggressive shared continuous: **5,748,971,553,896.69×**, 34.28% reported / 34.11% 4h
  mark-to-market maxDD. This path-dependent multiple is not a forecast; live DD can be far worse.
- Queue sensitivity is complete across five deterministic seeds. The harsh combined 5bps
  penetration + 70% eligible-fill case retains 65.6% median log growth, stays green in every
  annual fold, and reaches 38.15% worst 4h MTM DD.
- The standard exposure policy was re-searched. A looser TRAIN winner failed held-out DD at 28.6%,
  so the shipped 4.4% margin / 1.10× notional caps remain unchanged.
- **Round 19 shared live parity shipped:** one multi-symbol orchestrator serializes the account-wide
  exposure check/size/place sequence; a process lock rejects duplicates; realized-balance peak,
  maker pending orders, and trailing context persist atomically across restarts.
- Shipped configs now use deterministic marginal execution, matching fast/full backtests and the
  Round 8c signal-only winner. LLM consensus remains explicit opt-in only.
- **Round 20 walk-forward expansion:** return advantage is robust (all seeds/search sizes beat
  static; median ratios 1.64×, 1.89×, and 2.02× at 60/300/1,000 trials), but every seed/window chose
  a different parameter winner and even high turnover penalties did not produce convergence. Keep
  static configs for the first paper run; operational retuning is post-paper research.
- **Round 21 regime switching rejected:** the unchanged static strategy won all five independent
  TRAIN searches. No regime overlay was ported.
- Verification: **329 tests pass**; full-engine↔fastbt 2024 maker parity remains exactly equal
  (+226.20%, 562 trades, 22.03% maxDD, no mismatches).

### Previously expected next task: paper trading (now blocked by Round 22 alignment audit)

Use one shared testnet process, not independent symbol stacks:

```bash
python -m llm_trading_bot.main --mode live --shared-configs \
  config-aggressive.json config-eth-aggressive.json config-sol-aggressive.json
```

The command trades immediately. Before running it, supply testnet credentials and get explicit
approval to start. Then add the requested Portainer/Grafana deployment around this single
orchestrator and measure actual maker fill rate plus live-vs-backtest drift.

Artifacts: `opt/cadence_correction_results.json`, `opt/queue_fill_sensitivity_results.json`,
`opt/portfolio_exposure_cadence_results.json`, `opt/walk_forward_robustness_results.json`,
`opt/walk_forward_turnover_results.json`, and `opt/regime_search_results.json`.

---

## Historical handoff through Round 17 (superseded where the latest section differs)

Continue the profit-maximization loop on this trading bot. Read `AGENTS.md` and
`opt/README.md` first — they document the architecture and all seventeen completed
optimization rounds. This file is the handoff; trust it over stale prose elsewhere.

## Current state (2026-07-13, git log has the full story)

- **Chosen high-return research profile (honest sub exits + funding + liquidation + 2bps market
  slip, 2021-01→2025-06): the shared BTC+ETH+SOL aggressive portfolio compounds 920,165.82× at
  35.95% reported maxDD / 36.15% independent 4h-close mark-to-market maxDD.** Configs:
  `config-aggressive.json`, `config-eth-aggressive.json`, `config-sol-aggressive.json`.
  This is the user's chosen aggressive profile, but the multiple is not a forecast and live DD can
  be materially worse.
- **Standard capped profile remains the default:** standalone BTC 30.08×, ETH 92.46×, SOL 492.23×
  with every yearly fold green and maxDD ~20–24%; shared continuous is 1,905.59× at 25.03% maxDD.
  Configs: `config.json`, `config-eth.json`, `config-sol.json`.
- **Maker entry is shipped (Round 11):** honest same-fill-bar exits, engine/fastbt parity,
  post-only live lifecycle, persisted reconciliation, and one-primary-bar expiry. Strict sub-bar
  maker results remain better on BTC/ETH/SOL; all three configs now use `entry_mode: "maker"`.
- **Shared exposure controls shipped (Round 16):** BTC+ETH+SOL interleaved against one balance now
  caps committed margin at 4.4% of equity and entry notional at 1.10× equity, with the bounded
  Round 15 anti-martingale overlay. Continuous growth is 1,905.59× at 25.03% maxDD; TEST is 6.48×
  at 25.03% maxDD and every yearly fold is green. **Acceptance now targets approximately 25% maxDD**:
  small reporting/model noise around 25% is acceptable, but materially higher DD is rejected.
  This remains a research/selection target, NOT a live kill switch or forced-liquidation rule.
- **Separate aggressive profile shipped (Round 17):** `config-aggressive.json` plus ETH/SOL peers
  inherit the standard configs but disable shared margin/notional caps and raise the per-trade USD
  ceiling to $1B so sizing continues to compound. All currently inherit `bitget.testnet: true`,
  and no live/testnet process was started. Continuous historical growth is 920,165.82× at 35.95%
  reported maxDD (36.15% independent 4h mark-to-market). The three deepest 4h episodes were 36.15%,
  35.02%, and 33.72%; ≥33% occurred in 11/231 weeks. This is an explicitly accepted aggressive
  research profile; the capped Round 16 files remain the defaults and the result is not a forecast.
- **Walk-forward retuning is promising but unstable (Round 13):** with Round 14 points, the
  60-trial cadence produced 13.08× unseen vs 9.63× static across 2023-2025H1, but badly lagged
  static in 2025H1 and has only three deployment windows.
- **Scoring points shipped (Round 14):** after a 120-trial overfit warning, a 500-trial TRAIN
  winner improved BTC TEST + chrono and transferred strongly to untouched ETH/SOL. Nine point
  overrides are in all configs; canonical defaults/logic remain in `openwebui_filter.py`.
- **Anti-martingale is a return overlay, not a DD control (Rounds 15–17):** bounded to
  0.70×–1.10×; it runs under caps in the standard profile and uncapped at portfolio level in the
  explicitly named aggressive profile.
- Strategy: 4h primary, score→route→trade; trailing stops (act 0.94%/cb 0.33%),
  pyramiding (max_positions 3, same-direction), conviction sizing (exponent 1.0),
  opposite-signal exit (threshold 20), DD circuit-breaker (25%→1 slot, risk×0.5),
  lev 25 aggressive / 12 conservative tier, ATR stop 2.26×, TP RR 2.02/3.34 (70% @TP1).
- Tests: 309 pass (`pytest -q`).
  Venv `/tmp/tmlvenv` has everything (pandas/pydantic/ccxt/matplotlib/schedule/pytest);
  the system python has no pip. If the venv is gone, recreate:
  `python3 -m venv --without-pip /tmp/tmlvenv` then bootstrap pip from another venv or get-pip.

## The optimization harness (use it — 4000× faster than the engine)

- `opt/fastbt.py` — precomputes indicators once (causal ⇒ numerically identical to the
  engine; validated digit-equal repeatedly). Models slippage, isolated-margin
  liquidation, funding, and strategy variants behind a `strat=` dict.
  `exit_granularity="sub"` replays 1h sub-bars for honest intrabar sequencing.
- `opt/driver.py` — `setup(symbol=None)` loads 2020-08→2025-06 Bitget candles + Binance
  funding (both disk-cached under `history/`, gitignored);
  `evaluate(overrides, folds=..., slip=..., funding=True, strat=..., exit_granularity=...)`
  → dict with per-fold returns, geo-mean, compound, worst fold, maxDD.
  Folds: `FOLDS` (yearly), `TRAIN_FOLDS`/`TEST_FOLDS` (interleaved half-years).
- `opt/drawdown.py` + `MultiAssetResult.equity_curve` — non-mutating 4h-close mark-to-market
  sampling and peak-to-recovery episode analysis. It deliberately does not update the portfolio's
  strategy peak, DD throttle, sizing, or entries. Round 17 results:
  `opt/aggressive_profile_results.json`.
- Typical eval: ~0.2s for 5 folds. Run scripts with `PYTHONPATH=. /tmp/tmlvenv/bin/python`.

## NON-NEGOTIABLE methodology (each rule exists because it caught a real error)

1. **Never trust an in-sample max.** Select on TRAIN folds, report held-out TEST +
   chronological (21-23 → 24-25) splits. Slippage ≥2bps and `funding=True` always.
2. **Intrabar = worst case.** Adverse extreme first; SL before TP in one bar. Guarded by
   `tests/test_intrabar_conservatism.py`.
3. **Trailing ratchets ONCE per COMPLETED 4h bar, stop fixed intrabar.** Hourly
   ratcheting collapses the edge 84×→5× and nothing recovers it. Live scheduler is
   bar-close gated (`tests/test_trailing_cadence.py`). Never revert to per-tick trailing.
4. **After ANY engine change, re-verify engine==fastbt digit-equal** (pattern: /tmp
   scripts in git history; run one year, compare return/trades/maxDD exactly).
5. **Bitget data gotchas:** history endpoint is 200-cap END-anchored (handled in
   `bitget_csv.py`); 1h perp history is placeholder junk before 2021-01-02 (fastbt
   auto-masks); Bitget funding API only serves ~3 months → Binance series is the proxy.
6. Keep engine + `openwebui_filter.py` + scheduler in sync (single source of truth);
   run the full test suite before every commit; commit after each validated round.
7. **Do not mix risk profiles.** The default shared portfolio targets approximately 25% maxDD
   (Round 16: 25.03%). The explicitly named aggressive configs accept ~36% historical maxDD for the
   uncapped anti-martingale return path. Always state which profile is being tested and assume live
   aggressive DD can be materially worse than its backtest.
   Do NOT implement a live drawdown kill switch, synthetic threshold fill, or forced portfolio
   close to manufacture compliance. For the standard profile, reach its target through ex-ante
   exposure controls; for both profiles, report natural realized and 4h mark-to-market DD honestly.

## Done so far (don't retry — see opt/README.md rounds)

- ~~**Funding as a SIGNAL**~~ — **DONE / REJECTED (Round 7).** Real but trend-confounded, barely
  intersects entries, every win in-sample-concentrated, fails held-out TEST. Don't retry without
  a materially different mechanism. Machinery/EDA kept in `fastbt`/`opt/eda_funding*.py`.
- ~~**Single-LLM gate**~~ — **DONE / REJECTED (Rounds 8/8b/8c, 2026-07-13).** `qwen3.6:35b-a3b-q8_0`
  as a MARGINAL-entry gate, leakage-blinded, fixed-point replay. Non-thinking strongly rejected
  (229.51×→144.50×); a mixed n=36 thinking pilot was then EXPANDED and came back worse across all
  splits. Signal-only trading wins outright (the model mostly turns entries into WAIT). Do not retry
  as a per-entry accept/reject gate. `opt/llm_gate_pilot.py` + caches kept for reference only.
- ~~**Maker-entry**~~ — **DONE / SHIPPED (Round 11).** Honest pending lifecycle and parity.
- ~~**More assets**~~ — **DONE (Round 10).** SOL green every fold with the unchanged config;
  `config-sol.json` added. The config is now green on 3 assets (BTC/ETH/SOL).
- ~~**Multi-asset shared portfolio**~~ — **DONE (Rounds 12/16).** Harness added in Round 12;
  validated global exposure caps shipped in Round 16.
- ~~**Scoring internals constrained search**~~ — **DONE / SHIPPED (Round 14).** BTC TEST +
  chrono and ETH/SOL transfer validated nine overrides.
- ~~**Walk-forward retuning pilot**~~ — **DONE / PROMISING (Round 13).** Expand before adoption.
- ~~**Anti-martingale sizing**~~ — **DONE / SHIPPED AS AN OVERLAY (Rounds 15/16).** It failed as
  standalone DD control but was validated under the portfolio-wide exposure caps.
- ~~**Portfolio-wide exposure controls**~~ — **DONE / SHIPPED (Round 16).** 4.4% equity-margin and
  1.10× equity-notional caps produce 25.03% shared maxDD across TEST/annual/full validation.
- ~~**Uncapped aggressive profile**~~ — **DONE / SHIPPED SEPARATELY (Round 17).** Standard configs
  remain capped; the `*-aggressive.json` inheritance profiles reproduce 920,165.82× / 35.95% DD.

## Historical improvement backlog (completed/superseded by the latest section)

1. **Queue/fill sensitivity** — haircut touched maker fills or probabilistically model queue
   position. This is now the highest-value stress test because the aggressive result assumes every
   touched maker limit fills; report how much of 920,165.82× survives realistic fill haircuts.
2. **Shared-live orchestration/state parity** — standard-profile cap checks can race across
   independent symbol stacks, while each live process also keeps its own in-session DD peak that
   resets on restart. Use one orchestrator with persisted shared peak/state before treating live
   behavior as equivalent to the shared backtest; this applies to both profiles.
3. **Expand walk-forward retuning** — multiple seeds/search sizes, stability of selected params,
   and turnover/operational costs. Adopt only if the 1.36× pilot advantage remains robust.
4. **Regime-switching params** — `detect_market_regime` exists; different thresholds/leverage per
   regime (e.g. wider trailing in VOLATILE). Medium odds, self-contained.
5. **Ship it — paper-trade (needs Marc's explicit go-ahead; externally visible).** Testnet keys are
   the default (`bitget.testnet: true`). Native Portainer stacks (BTC + ETH, optionally SOL) — see
   CLAUDE.md standing preference in ~/Documents/portainer — + Grafana from `logs/decisions.jsonl`.
   Live-vs-backtest drift is the ultimate validation. Maker lifecycle is ready. Do not start
   unprompted.

All non-paper items in this historical backlog were completed in Rounds 18–21. Use the latest
section at the top for current status. Ask Marc before anything irreversible or externally visible;
he reviews via git log, so keep commits self-explanatory.
