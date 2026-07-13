# Prompt for the next optimization session

Copy-paste everything below the line into a fresh Claude Code session started in
`~/Documents/new-llm-trading`.

---

Continue the profit-maximization loop on this trading bot. Read `AGENTS.md` and
`opt/README.md` first — they document the architecture and all eight completed
optimization rounds. This file is the handoff; trust it over stale prose elsewhere.

## Current state (2026-07-12, git log has the full story)

- **Headline (funding + liquidation + 2bps slip, 2021-01→2025-06, compounding):
  BTC 228× (84× @5bps), ETH 1015× (296×) with the SAME unchanged config — every year
  green on both, maxDD ~22-30%.** Configs: `config.json` (BTC), `config-eth.json`.
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

## Improvement backlog, ranked (untried ideas)

1. ~~**Funding as a SIGNAL**~~ — **DONE / REJECTED (Round 7, 2026-07-12).** Measured: the
   raw funding→forward-return effect is real but trend-confounded and barely intersects the
   strategy's actual entries; every apparent win is in-sample-concentrated and fails
   held-out TEST. Config unchanged. Opt-in machinery + EDA kept in `fastbt`/`opt/eda_funding*.py`/
   `opt/probe_funding.py`; full write-up in `opt/README.md` Round 7. Don't retry without a
   materially different mechanism.
2. ~~**Single-LLM gate backtest**~~ — **DONE / REJECTED (Round 8, 2026-07-12).** Dropped the stale
   three-model consensus plan in favor of exactly `qwen3.6:35b-a3b-q8_0` on Marc's Ollama
   (`192.168.0.70:11435`). Leakage-blinded prompts, 2bps+funding+liquidation, and iterative
   fixed-point replay produced 1,049 valid decisions (0 failures): the model accepted 748
   and WAITed on 301. Full gate: 229.51×→144.50×, maxDD 21.7%→23.6%; held-out TEST
   7.51×→6.72×. Auto-trading MARGINAL wins decisively; production config unchanged. Full
   methodology/results in `opt/README.md` Round 8. Don't retry without a materially different
   information source—the model only saw the same indicators already encoded by the score.
3. **Multi-asset shared portfolio** — BTC+ETH compounding one balance (fastbt currently
   single-symbol per sim). Interleave two Precomputed streams by timestamp with shared
   Portfolio + per-symbol position slots. Measures the real diversification benefit vs
   two isolated instances (BTC/ETH DDs partially overlap — quantify it).
4. **Maker-entry modeling** — entries are market/taker (0.06% + slip). Limit-at-close:
   maker 0.02%, no slip, but misses fills when price runs. Model fill rule: filled iff
   next bar's low < limit (long). If EV-positive, big cost saving at this trade count.
5. **Scoring internals evolution** — the hand-tuned point values inside
   `openwebui_filter.calc_*_score` (EMA-stack ±30, RSI bands ±15/20, MACD ±15, etc.)
   have never been searched. Parametrize them in fastbt (they're pure functions),
   CMA-ES/random-search on TRAIN halves only, strict TEST + chrono validation — huge
   overfit surface, be brutal about held-out discipline.
6. **More assets** — SOL/others through the same pipeline (`driver.setup(symbol=...)`;
   fetch is automatic). A config that's green on 3+ assets is near-unfalsifiable.
7. **Regime-switching params** — `detect_market_regime` exists; different
   thresholds/leverage per regime (e.g. wider trailing in VOLATILE). Medium odds.
8. **Anti-martingale sizing** — scale risk up after wins / down after losses (the DD
   throttle only handles deep-DD; the win-streak side is untested).
9. **Walk-forward re-tuning cadence** — does yearly re-optimization on a trailing
   window beat the static config? (Simulate: tune on year N-2..N-1, trade year N.)
10. **Ship it** — deploy paper-trading on the Portainer box (testnet keys are already
    the default, `bitget.testnet: true`): two stacks (BTC + ETH configs) as native
    Portainer stacks (see CLAUDE.md standing preference in ~/Documents/portainer),
    Grafana dashboard from `logs/decisions.jsonl`. Live-vs-backtest drift is the
    ultimate validation of everything above.

Work the loop: pick the top item, implement in fastbt first, validate walk-forward,
port winners to engine+config+scheduler with tests, verify parity, update
`opt/README.md` + `AGENTS.md`, commit. Ask Marc before anything irreversible or
externally visible. He reviews via git log — keep commits self-explanatory.
