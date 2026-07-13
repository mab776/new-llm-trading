# Prompt for the next optimization session

Copy-paste everything below the line into a fresh Claude Code session started in
`~/Documents/new-llm-trading`.

---

Continue the profit-maximization loop on this trading bot. Read `AGENTS.md` and
`opt/README.md` first — they document the architecture and all fourteen completed
optimization rounds. This file is the handoff; trust it over stale prose elsewhere.

## Current state (2026-07-13, git log has the full story)

- **Headline (honest sub exits + funding + liquidation + 2bps market slip,
  2021-01→2025-06, compounding): BTC 70.28×, ETH 157.45×, SOL 777.00× with the SAME
  maker-entry/scoring config — every yearly fold green, maxDD ~22-30%.** The green-everywhere
  robustness is the finding, not the multiple. Configs: `config.json`, `config-eth.json`,
  `config-sol.json`.
- **Maker entry is shipped (Round 11):** honest same-fill-bar exits, engine/fastbt parity,
  post-only live lifecycle, persisted reconciliation, and one-primary-bar expiry. Strict sub-bar
  maker results remain better on BTC/ETH/SOL; all three configs now use `entry_mode: "maker"`.
- **Shared portfolio exists (Round 12):** BTC+ETH+SOL interleaved against one balance. Honest
  maker sub replay stays green every year, but maxDD rises to ~38%. **New acceptance criterion:
  reject any shared-portfolio strategy whose validated maxDD exceeds 25%.** This is a research/
  selection threshold, NOT a live kill switch or forced-liquidation rule.
- **Walk-forward retuning is promising but unstable (Round 13):** with Round 14 points, the
  60-trial cadence produced 13.08× unseen vs 9.63× static across 2023-2025H1, but badly lagged
  static in 2025H1 and has only three deployment windows.
- **Scoring points shipped (Round 14):** after a 120-trial overfit warning, a 500-trial TRAIN
  winner improved BTC TEST + chrono and transferred strongly to untouched ETH/SOL. Nine point
  overrides are in all configs; canonical defaults/logic remain in `openwebui_filter.py`.
- **Anti-martingale sizing rejected for DD control (Round 15):** 96 shared-portfolio TRAIN-only
  variants found no candidate below 25% maxDD. The minimum-TRAIN-DD candidate reduced TRAIN DD
  37.8%→31.8% but worsened held-out TEST DD 35.3%→36.0%; nothing was shipped.
- Strategy: 4h primary, score→route→trade; trailing stops (act 0.94%/cb 0.33%),
  pyramiding (max_positions 3, same-direction), conviction sizing (exponent 1.0),
  opposite-signal exit (threshold 20), DD circuit-breaker (25%→1 slot, risk×0.5),
  lev 25 aggressive / 12 conservative tier, ATR stop 2.26×, TP RR 2.02/3.34 (70% @TP1).
- Tests: 290 pass (`PYTHONPATH=. /tmp/tmlvenv/bin/python -m pytest tests/ -q`).
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
7. **Shared-portfolio maxDD must be ≤25% to be accepted.** Enforce this in search/validation
   objectives across TRAIN, held-out TEST, chronological OOS, and the full-period report. Do NOT
   implement a live drawdown kill switch, synthetic threshold fill, or forced portfolio close to
   manufacture compliance. Reach the target through ex-ante exposure controls (global slots,
   margin/notional caps, and/or risk scaling), then report the natural realized maxDD honestly.

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
- ~~**Multi-asset shared portfolio**~~ — **DONE AS HARNESS (Round 12).** Needs global exposure cap.
- ~~**Scoring internals constrained search**~~ — **DONE / SHIPPED (Round 14).** BTC TEST +
  chrono and ETH/SOL transfer validated nine overrides.
- ~~**Walk-forward retuning pilot**~~ — **DONE / PROMISING (Round 13).** Expand before adoption.
- ~~**Anti-martingale sizing**~~ — **DONE / REJECTED (Round 15).** A causal per-asset closed-trade
  streak improved return but failed the ≤25% shared-DD constraint and worsened held-out TEST DD.
  Harness/results retained; don't retry the same streak mechanism.

## Improvement backlog, ranked (2026-07-13)

1. **Portfolio-wide exposure controls — TOP PRIORITY.** Shared BTC+ETH+SOL permits nine 25× slots
   and reaches ~38% maxDD, so it currently FAILS the new ≤25% acceptance criterion. Search global
   slot, margin, notional, and risk caps on TRAIN only; require ≤25% maxDD on held-out TEST,
   chronological OOS, and full-period validation. Preserve normal trade exits—no kill switch or
   forced close. Mirror only the validated ex-ante sizing/exposure controls in live orchestration
   before any shared deployment. Optimize return subject to the DD constraint, not DD alone.
2. **Expand walk-forward retuning** — multiple seeds/search sizes, stability of selected params,
   and turnover/operational costs. Adopt only if the 1.36× pilot advantage remains robust.
3. **Regime-switching params** — `detect_market_regime` exists; different thresholds/leverage per
   regime (e.g. wider trailing in VOLATILE). Medium odds, self-contained.
4. **Queue/fill sensitivity** — haircut touched maker fills or probabilistically model queue
   position to bound the backtest's optimistic assumption that every touched limit fills.
5. **Ship it — paper-trade (needs Marc's explicit go-ahead; externally visible).** Testnet keys are
   the default (`bitget.testnet: true`). Native Portainer stacks (BTC + ETH, optionally SOL) — see
   CLAUDE.md standing preference in ~/Documents/portainer — + Grafana from `logs/decisions.jsonl`.
   Live-vs-backtest drift is the ultimate validation. Maker lifecycle is ready. Do not start
   unprompted.

Work the loop: pick the top item, implement in fastbt first, validate walk-forward,
port winners to engine+config+scheduler with tests, verify parity, update
`opt/README.md` + `AGENTS.md`, commit. Ask Marc before anything irreversible or
externally visible. He reviews via git log — keep commits self-explanatory.
