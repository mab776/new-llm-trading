# Prompt for the next optimization session

Copy-paste everything below the line into a fresh Claude Code session started in
`~/Documents/new-llm-trading`.

---

Continue the profit-maximization loop on this trading bot. Read `AGENTS.md` and
`opt/README.md` first — they document the architecture and all eight completed
optimization rounds. This file is the handoff; trust it over stale prose elsewhere.

## Current state (2026-07-13, git log has the full story)

- **Headline (funding + liquidation + 2bps slip, 2021-01→2025-06, compounding):
  BTC 228× (84× @5bps), ETH 1015× (296×) with the SAME unchanged config — every year
  green on both, maxDD ~22-30%.** Configs: `config.json` (BTC), `config-eth.json`.
- **Now green on a THIRD asset too (Round 10):** unchanged config on SOL is positive every
  yearly fold — honest sub-bar exits give 56.8× (taker) / 244× (maker), worst fold +45%/+59%.
  `config-sol.json` added. The green-everywhere robustness is the finding, not the multiple.
- **Maker-entry is an EV-positive edge NOT yet shipped (Round 9):** limit-at-close instead of
  market/taker beats taker on all three assets + held-out TEST + both exit modes, but only as a
  *fastbt screen* (opt-in `strat["entry_mode"]`). It carries a one-bar exit-delay optimism →
  must be ported to the engine with an honest pending-order lifecycle before it's trusted/live.
- Strategy: 4h primary, score→route→trade; trailing stops (act 0.94%/cb 0.33%),
  pyramiding (max_positions 3, same-direction), conviction sizing (exponent 1.0),
  opposite-signal exit (threshold 20), DD circuit-breaker (25%→1 slot, risk×0.5),
  lev 25 aggressive / 12 conservative tier, ATR stop 2.26×, TP RR 2.02/3.34 (70% @TP1).
- Tests: 263 pass (`PYTHONPATH=. /tmp/tmlvenv/bin/python -m pytest tests/ -q`).
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

## Done so far (don't retry — see opt/README.md rounds)

- ~~**Funding as a SIGNAL**~~ — **DONE / REJECTED (Round 7).** Real but trend-confounded, barely
  intersects entries, every win in-sample-concentrated, fails held-out TEST. Don't retry without
  a materially different mechanism. Machinery/EDA kept in `fastbt`/`opt/eda_funding*.py`.
- ~~**Single-LLM gate**~~ — **DONE / REJECTED (Rounds 8/8b/8c, 2026-07-13).** `qwen3.6:35b-a3b-q8_0`
  as a MARGINAL-entry gate, leakage-blinded, fixed-point replay. Non-thinking strongly rejected
  (229.51×→144.50×); a mixed n=36 thinking pilot was then EXPANDED and came back worse across all
  splits. Signal-only trading wins outright (the model mostly turns entries into WAIT). Do not retry
  as a per-entry accept/reject gate. `opt/llm_gate_pilot.py` + caches kept for reference only.
- ~~**Maker-entry (fastbt screen)**~~ — **DONE, EV-positive (Round 9).** Opt-in
  `strat["entry_mode"]="maker"`. NOT shipped — needs the engine port below (#1).
- ~~**More assets**~~ — **DONE (Round 10).** SOL green every fold with the unchanged config;
  `config-sol.json` added. The config is now green on 3 assets (BTC/ETH/SOL).

## Improvement backlog, ranked (2026-07-13)

1. **Ship maker-entry to the engine (finish Round 9) — TOP PRIORITY.** The fastbt screen is
   EV-positive on all 3 assets + held-out TEST but books the fill once per 4h bar *after* that
   bar's exit step → a one-bar exit-delay optimism. Port `entry_mode: maker` into
   `backtesting.py` + `openwebui_filter.py`/scheduler with a real pending-order lifecycle (place
   limit at close → fill iff next bar trades back to it, else cancel → honest same-bar exit after
   fill), add parity + fill-lifecycle tests, re-verify engine==fastbt digit-equal. Only then is
   the magnitude trustworthy or shippable. Live also adds non-fill/queue-position risk not modelled.
2. **Multi-asset shared portfolio** — BTC+ETH(+SOL) compounding one balance (fastbt is currently
   single-symbol per sim). Interleave Precomputed streams by timestamp with a shared Portfolio +
   per-symbol slots. Measures the real diversification benefit vs isolated instances (their DDs
   partially overlap — quantify it). Biggest structural lift.
3. **Scoring internals evolution** — the hand-tuned point values inside
   `openwebui_filter.calc_*_score` (EMA-stack ±30, RSI bands ±15/20, MACD ±15, etc.) have never
   been searched. Parametrize in fastbt (pure functions), CMA-ES/random-search on TRAIN halves
   only, strict TEST + chrono validation — LARGEST overfit surface in the backlog, be brutal.
4. **Regime-switching params** — `detect_market_regime` exists; different thresholds/leverage per
   regime (e.g. wider trailing in VOLATILE). Medium odds, self-contained.
5. **Anti-martingale sizing** — scale risk up after wins / down after losses (the DD throttle only
   handles deep-DD; the win-streak side is untested). Small, quick to screen.
6. **Walk-forward re-tuning cadence** — does yearly re-optimization on a trailing window beat the
   static config? (Tune on year N-2..N-1, trade year N.) Pure harness experiment, zero engine risk;
   also a fragility sanity-check on the static config.
7. **Ship it — paper-trade (needs Marc's explicit go-ahead; externally visible).** Testnet keys are
   the default (`bitget.testnet: true`). Native Portainer stacks (BTC + ETH, optionally SOL) — see
   CLAUDE.md standing preference in ~/Documents/portainer — + Grafana from `logs/decisions.jsonl`.
   Live-vs-backtest drift is the ultimate validation. Best done AFTER #1 lands. Do not start
   unprompted.

Work the loop: pick the top item, implement in fastbt first, validate walk-forward,
port winners to engine+config+scheduler with tests, verify parity, update
`opt/README.md` + `AGENTS.md`, commit. Ask Marc before anything irreversible or
externally visible. He reviews via git log — keep commits self-explanatory.
