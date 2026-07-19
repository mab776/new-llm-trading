"""Re-grid the consecutive-loss penalty on the NEW alignment base (Marc,
2026-07-19). Mechanism for suspicion: the penalty stacks onto entry thresholds
(12.6/21.3) that were tuned when scores carried up to ±10 of secondary-TF
alignment; the deployed {"1h": 0, "1d": 3} compresses that to ±3, so the shipped
5/loss (cap 20, decay 10) now bites relatively harder than what the original
random search (loss_pen ∈ {0,2,5,8}, OLD base) selected. An exploratory 1-D
slice showed monotone TRAIN+TEST improvement toward 0 — but the penalty is an
INSURANCE knob (regime breaks), so return-only selection is not allowed.

PRE-COMMITTED PROTOCOL (written before results):
  Select on TRAIN geo, subject to ALL of:
    (1) TEST geo ≥ shipped TEST geo − 2 pts        (held-out agreement)
    (2) 2022 bear-fold geo > 0 AND ≥ shipped − 5   (insurance preserved)
    (3) tie-break: lower FULL maxDD
  Then 3-asset OOS holdout invariance (shipped vs selected, $100 + real mins):
    adopt only if selected/shipped compound ratio ≥ 0.92 (not-worse-within-noise).
  If no candidate clears all gates → verdict "keep 5/20/10", stop.

Grid: step ∈ {0,1,2,3,5,8} × cap ∈ {10,20} × decay ∈ {5,10,20} (step=0 once —
cap/decay inert). Cooldown (2, searched) deliberately untouched.

Run: PYTHONPATH=. /tmp/tmlvenv/bin/python opt/probe_penalty.py
"""
from __future__ import annotations

import opt.driver as drv
from opt.driver import evaluate, HALF_FOLDS, TRAIN_FOLDS, TEST_FOLDS, FOLDS

SLIP = 0.0002
BEAR_FOLDS = [HALF_FOLDS[2], HALF_FOLDS[3]]   # 2022H1 + 2022H2
SHIPPED = (5.0, 20.0, 10)


def row(step: float, cap: float, decay: int) -> dict:
    ov = {"risk.consecutive_loss_penalty": step,
          "risk.max_consecutive_loss_penalty": cap,
          "risk.loss_penalty_decay_candles": decay}
    tr = evaluate(ov, folds=TRAIN_FOLDS, slip=SLIP, funding=True)
    te = evaluate(ov, folds=TEST_FOLDS, slip=SLIP, funding=True)
    be = evaluate(ov, folds=BEAR_FOLDS, slip=SLIP, funding=True)
    fl = evaluate(ov, folds=FOLDS, slip=SLIP, funding=True)
    return {"step": step, "cap": cap, "decay": decay,
            "train": tr["geo_pct"], "test": te["geo_pct"],
            "bear": be["geo_pct"], "bear_wf": be["worst_fold"],
            "full_cx": fl["compound_x"], "dd": fl["max_dd"],
            "trades": fl["total_trades"]}


def main() -> None:
    drv.setup()  # BTC, live config base (alignment 1h=0/1d=3 from config)
    rows = []
    print(f"{'step':>5}{'cap':>5}{'decay':>6}{'TRAIN':>8}{'TEST':>7}{'BEAR22':>8}"
          f"{'bearWF':>8}{'FULLcx':>9}{'maxDD':>6}{'trades':>7}")
    grid = [(0.0, 20.0, 10)] + [(s, c, d) for s in (1.0, 2.0, 3.0, 5.0, 8.0)
                                for c in (10.0, 20.0) for d in (5, 10, 20)]
    for step, cap, decay in grid:
        r = row(step, cap, decay)
        rows.append(r)
        mark = "  ← shipped" if (step, cap, decay) == SHIPPED else ""
        print(f"{step:>5.0f}{cap:>5.0f}{decay:>6d}{r['train']:>+8.1f}{r['test']:>+7.1f}"
              f"{r['bear']:>+8.1f}{r['bear_wf']:>+8.0f}{r['full_cx']:>9.1f}"
              f"{r['dd']:>6.1f}{r['trades']:>7d}{mark}", flush=True)

    ship = next(r for r in rows if (r["step"], r["cap"], r["decay"]) == SHIPPED)
    ok = [r for r in rows
          if r["test"] >= ship["test"] - 2.0
          and r["bear"] > 0 and r["bear"] >= ship["bear"] - 5.0]
    ok.sort(key=lambda r: (-r["train"], r["dd"]))
    print(f"\nShipped: TRAIN {ship['train']:+.1f}  TEST {ship['test']:+.1f}  "
          f"BEAR {ship['bear']:+.1f}  ddFULL {ship['dd']:.1f}")
    print("Candidates clearing pre-committed gates (TRAIN-ranked):")
    for r in ok[:6]:
        print(f"  step={r['step']:.0f} cap={r['cap']:.0f} decay={r['decay']} | "
              f"TRAIN {r['train']:+.1f} TEST {r['test']:+.1f} BEAR {r['bear']:+.1f} "
              f"dd {r['dd']:.1f}")
    if not ok or ok[0]["train"] <= ship["train"]:
        print("\nVERDICT: no candidate beats shipped under the gates — keep 5/20/10.")
        return
    sel = ok[0]

    # ── Holdout invariance (spends holdout once, on the single selected cell) ──
    from opt.holdout_oos import HOLD_START, HOLD_END, PROFILES, SYMBOLS, _load
    from opt.multi_asset import simulate_multi
    MQ = {"BTC": 0.0001, "ETH": 0.01, "SOL": 0.1}
    print(f"\nSelected: step={sel['step']:.0f} cap={sel['cap']:.0f} "
          f"decay={sel['decay']} → holdout invariance vs shipped:")
    assets = {l: _load(SYMBOLS[l], c) for l, c in PROFILES["standard"].items()}
    for it in assets.values():
        it.config.backtesting.initial_balance = 100.0

    def hold(tag, step, cap, decay):
        for it in assets.values():
            rm = it.config.risk_management
            rm.consecutive_loss_penalty = step
            rm.max_consecutive_loss_penalty = cap
            rm.loss_penalty_decay_candles = decay
        res = simulate_multi(assets, HOLD_START, HOLD_END, slip=.0002,
                             exit_granularity="sub",
                             strat={"min_qty": MQ, "size_step": MQ})
        x = max(.01, 1 + res.return_pct / 100)
        print(f"  {tag:<28} {x:6.2f}x  maxDD {res.max_dd_pct:4.1f}%  "
              f"trades {res.trades}  win {res.win_rate:.0f}%")
        return x

    xs = hold("shipped 5/20/10", *SHIPPED)
    xc = hold(f"selected {sel['step']:.0f}/{sel['cap']:.0f}/{sel['decay']}",
              sel["step"], sel["cap"], sel["decay"])
    ratio = xc / xs
    print(f"\n  ratio selected/shipped = {ratio:.3f}  "
          f"(adopt gate ≥ 0.92) → {'ADOPT candidate' if ratio >= .92 else 'KEEP shipped'}")


if __name__ == "__main__":
    main()
