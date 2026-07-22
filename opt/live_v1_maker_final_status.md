# Live v1-maker era — final status snapshot (archived at counter reset)

Frozen 2026-07-21 ~19:10 UTC, just after the maker-v2 deploy (`6913f3d`, 18:42 UTC) and
just before the dashboard/exporter counters were reset to the v2 era
(`--counters-since 2026-07-21T18:42:00+00:00`). Raw decision logs are untouched —
this is the human-readable tally of the go-live → v2 window (2026-07-16 16:00 UTC →
2026-07-21 18:42 UTC, standard profile, real money).

> **Update 2026-07-21 ~20:15 UTC:** the `--counters-since` filter flag was **dropped** in favour
> of physically archiving the v1 logs to `logs/v1maker-live-run/` (see that folder's README). The
> exporter now runs flagless and reads only the live v2 file. This snapshot's numbers are unchanged.

## Account
- Equity ~$194.0 (peak 194.14; includes the +$97 deposit 2026-07-19 — NOT profit)
- Realized trading P/L: **−$3.32** (gains +$2.00 / losses −$5.32; 3 W / 5 L)
  - by reason: signal_flip −$3.23 (3 lots, BTC), sl BTC +$2.00 (3 trailing wins),
    sl SOL −$2.10 (2 stop-outs)
- Open at snapshot: BTC LONG 0.0013 @ 66,304.7 (SL 64,828 / TP1 69,287.7); ETH, SOL flat

## v1 maker funnel (the reason v2 exists)
- Placements: 28 (BTC 20 buy + 2 sell, ETH 2, SOL 4)
- **MAKER_FILL 9** (BTC 7, SOL 2)
- **MAKER_POST_ONLY_CANCELLED 15** (BTC 13, SOL 2) — would-cross rejections:
  post-only refused a fillable price, tape then ran
- MAKER_CANCEL_UNFILLED (expired untouched) 1
- → **36% fill rate on resolved-touched orders.** Release-gate sim: that fill rate
  earns 2.99× on the clean OOS holdout vs the canonical 5.65× (63% log-growth
  retention — below the 86% maker-vs-taker bar, i.e. v1-maker was losing to
  taker's 3.29×).

## Other counters (context)
- MIN_SIZE_SKIP 24 (ETH 14 / SOL 8 / BTC 2) — small-balance granularity tax; the
  gate-passing rescue idea is GOOD_IDEAS.md #2 (not deployed)
- TRAIL_RATCHET 4, SIGNAL_FLIP_CLOSE 2, COOLDOWN_SKIP 2 (SOL)

## Why reset
The maker-vs-taker 86% decision must be measured on the **v2** execution
(retry re-peg at min(intended, bid)); mixing v1's broken fill data would poison it.
New clock: 2026-07-21 18:42 UTC → decision ~**2026-08-04**. Prometheus retains the
pre-reset series (step-drop at reset); `logs/decisions-*.jsonl` keep everything raw.
