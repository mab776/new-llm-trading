"""Reconcile the LIVE track record against the strategy's own backtest expectation.

Answers the recurring question: *if we simulate the same period, do we get
similar results?* It runs the frozen (currently deployed) standard-profile
configs over the live window on real Bitget minimums, in BOTH maker mode
(fill-on-touch = what the strategy expected) and taker mode (guaranteed fills =
the counterfactual if we'd missed no fills), then prints the **actual live
outcome** parsed from the decision logs right next to it.

What it compares — and why:
  The absolute account balance is NOT a valid comparison target because live
  balance includes deposits/withdrawals the sim knows nothing about (e.g. the
  +$100 spot->futures top-up on 2026-07-19). So the tool compares **realized
  trading P/L** — the sum of closed-lot PnL — which is deposit-independent and
  is the honest apples-to-apples number. It auto-detects mid-window balance
  jumps (deposits) and prints a warning so you never read the balance line as
  performance.

Caveats it will remind you of:
  * The sim applies the *current* deployed config across the whole window. Live
    history may have spanned a config change (alignment weights + loss penalty
    were deployed 2026-07-19), so pre-change bars are "what today's strategy
    would have done", not a bar-exact replay of the mixed live history. Use
    --start after a config change for a constant-config comparison.
  * The sim uses the optimistic execution model (maker fill-on-touch, exact-
    price TP fills, thin slippage). Live maker fills are throttled by post-only
    cancels; that gap is exactly what the maker-vs-taker decision measures. The
    maker/taker pair here brackets it.
  * Funding is omitted (negligible over a few days).

Run:
  PYTHONPATH=. /tmp/tmlvenv/bin/python -m opt.live_reconcile              # go-live -> last closed bar
  PYTHONPATH=. /tmp/tmlvenv/bin/python -m opt.live_reconcile --start "2026-07-19 04:00" --auto-initial
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import opt.fastbt as fb
from llm_trading_bot.config import load_config
from llm_trading_bot.data import configure_cache, fetch_multi_timeframe
from opt.multi_asset import AssetInput, simulate_multi

# --- live facts (defaults) --------------------------------------------------
GO_LIVE_START = "2026-07-16 16:00"   # first 4h bar the live bot acted on (UTC)
GO_LIVE_BALANCE = 100.37             # futures equity at go-live
# Warmup pad. 1h fetch reaches load_start - 30d (data._period_for_warmup floor);
# keep that reach off a month boundary, where Bitget's rolling 1h retention leaves
# a ragged/missing first candle (e.g. the missing 2026-06-01 00:00 BTC bar). The
# 4h primary still gets ample warmup from load_start - 60d. Auto-nudged forward on
# a startup gap (see _build_assets), so this only sets the first attempt.
DEFAULT_LOAD_START = "2026-07-08"
DEPOSIT_JUMP_USD = 5.0               # realized-balance step this large = deposit, not PnL

CFG = {"BTC": "config.json", "ETH": "config-eth.json", "SOL": "config-sol.json"}
SYM = {"BTC": "BTC/USDT:USDT", "ETH": "ETH/USDT:USDT", "SOL": "SOL/USDT:USDT"}
MIN_QTY = {"BTC": 0.0001, "ETH": 0.01, "SOL": 0.1}
SIZE_STEP = {"BTC": 0.0001, "ETH": 0.01, "SOL": 0.1}


# --- window helpers ---------------------------------------------------------
def _utc(s: str) -> datetime:
    """Parse a naive 'YYYY-MM-DD[ HH:MM]' string as UTC."""
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"unparseable UTC time: {s!r}")


def last_closed_4h(now: datetime | None = None) -> datetime:
    """Timestamp of the most recently *closed* 4h bar boundary (UTC)."""
    now = now or datetime.now(timezone.utc)
    floored = now.replace(minute=0, second=0, microsecond=0)
    floored -= timedelta(hours=floored.hour % 4)
    return floored


# --- live actuals from the decision logs ------------------------------------
def parse_live_actuals(log_dir: str, start: datetime, end: datetime) -> dict:
    """Pull realized trading P/L, closed lots, and deposits from decisions-*.jsonl."""
    closes: list[dict] = []
    balances: list[tuple[datetime, float]] = []  # (ts, realized_balance) from heartbeats
    for path in sorted(glob.glob(os.path.join(log_dir, "decisions-*.jsonl"))):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts_raw = r.get("timestamp")
                if not ts_raw:
                    continue
                ts = datetime.fromisoformat(ts_raw).astimezone(timezone.utc)
                if not (start <= ts < end):
                    continue
                action = r.get("action")
                if action == "LOT_CLOSED":
                    closes.append({
                        "ts": ts, "symbol": r.get("symbol"),
                        "direction": r.get("direction"), "reason": r.get("reason"),
                        "entry": r.get("entry"), "exit": r.get("exit_price"),
                        "size": r.get("filled_size"), "pnl": r.get("net_pnl_est", 0.0),
                    })
                elif action == "HEARTBEAT" and r.get("realized_balance") is not None:
                    balances.append((ts, float(r["realized_balance"])))

    balances.sort()
    deposits: list[tuple[datetime, float]] = []
    for (t0, b0), (t1, b1) in zip(balances, balances[1:]):
        if abs(b1 - b0) >= DEPOSIT_JUMP_USD:
            deposits.append((t1, b1 - b0))

    pnl = sum(c["pnl"] for c in closes)
    wins = sum(1 for c in closes if c["pnl"] > 0)
    return {
        "closes": closes,
        "realized_pnl": pnl,
        "trades": len(closes),
        "win_rate": (100.0 * wins / len(closes)) if closes else 0.0,
        "deposits": deposits,
        "first_balance": balances[0][1] if balances else None,
        "last_balance": balances[-1][1] if balances else None,
    }


# --- sim --------------------------------------------------------------------
def _load_asset(label: str, entry_mode: str, initial: float,
                load_start: str, end: str):
    cfg = load_config(CFG[label])
    configure_cache(cfg.data_cache.ttl_seconds)
    ds = cfg.data_source
    ds.exchange_symbol = SYM[label]
    cfg.trading.entry_mode = entry_mode
    cfg.backtesting.initial_balance = initial
    data = fetch_multi_timeframe(
        SYM[label], cfg.trading.timeframes,
        start_date=load_start, end_date=end,
        warmup_periods=0, source=ds.source, market=ds.market,
    )
    print(f"  {label} 4h rows={len(data['4h'])} "
          f"{data['4h'].index[0]} -> {data['4h'].index[-1]}", file=sys.stderr)
    pre = fb.precompute(data, cfg.trading.primary_timeframe, 200)
    return AssetInput(pre, cfg, None)


def _build_assets(entry_mode: str, initial: float, load_start: str, end: str,
                  max_nudges: int = 6):
    """Load all assets, nudging load_start forward on a ragged-edge 1h gap.

    Bitget's rolling 1h retention can drop the first candle of the warmup range,
    which the fail-closed loader rejects ("Incomplete ... history"). Advancing the
    load start by a couple days moves the 1h reach off the missing candle without
    touching the sim window itself.
    """
    ls = datetime.fromisoformat(load_start) if "T" in load_start else _utc(load_start)
    for attempt in range(max_nudges):
        ls_str = ls.strftime("%Y-%m-%d")
        try:
            return {lab: _load_asset(lab, entry_mode, initial, ls_str, end)
                    for lab in CFG}, ls_str
        except ValueError as e:
            msg = str(e).lower()
            if ("incomplete" in msg or "unavailable" in msg) and attempt < max_nudges - 1:
                ls += timedelta(days=2)
                print(f"[warmup gap at load_start; nudging to {ls:%Y-%m-%d} and retrying]",
                      file=sys.stderr)
                continue
            raise
    raise RuntimeError("could not find a gap-free warmup start")


def run_sim(entry_mode: str, initial: float, start: str, end: str,
            load_start: str, exit_granularity: str = "sub"):
    assets, _ = _build_assets(entry_mode, initial, load_start, end)
    strat = {"min_qty": MIN_QTY, "size_step": SIZE_STEP}
    try:
        return simulate_multi(assets, start, end, slip=.0002,
                              exit_granularity=exit_granularity, strat=strat), exit_granularity
    except Exception as e:  # pragma: no cover - sub-bar data occasionally short
        print(f"[sub granularity failed: {e}; falling back to primary]", file=sys.stderr)
        return simulate_multi(assets, start, end, slip=.0002,
                              exit_granularity="primary", strat=strat), "primary"


def _closed_lots(res):
    port = res.portfolio
    if port is None:
        return []
    return [t for t in port.trades if not t.is_open]


# --- reporting --------------------------------------------------------------
def show_sim(tag: str, res, initial: float) -> None:
    pl = initial * res.return_pct / 100.0
    print(f"\n=== SIM · {tag} ===")
    print(f"  trading P/L : ${pl:+.2f}   (return {res.return_pct:+.2f}% on ${initial:.2f})")
    print(f"  trades={res.trades}  win_rate={res.win_rate:.0f}%  "
          f"pf={res.profit_factor:.2f}  maxDD={res.max_dd_pct:.1f}%")
    print(f"  maker funnel: orders={res.maker_orders} touches={res.maker_touches} "
          f"queue_eligible={res.maker_queue_eligible} fills={res.maker_fills}")
    lots = _closed_lots(res)
    if lots:
        print("  closed lots:")
        for t in lots:
            print(f"    {t.symbol:9} {t.direction:5} entry={t.entry_price:.2f} "
                  f"exit={t.exit_price} pnl={t.net_pnl:+.2f} reason={t.exit_reason}")
    if res.portfolio and res.portfolio.open_trades:
        print("  still-open at window end:")
        for t in res.portfolio.open_trades:
            print(f"    {t.symbol:9} {t.direction:5} entry={t.entry_price:.2f} "
                  f"size={t.remaining_size}")


def show_live(live: dict) -> None:
    print("\n=== LIVE · actual (from decision logs) ===")
    print(f"  trading P/L : ${live['realized_pnl']:+.2f}   "
          f"(realized closed-lot PnL — deposit-independent)")
    print(f"  trades={live['trades']}  win_rate={live['win_rate']:.0f}%")
    if live["closes"]:
        print("  closed lots:")
        for c in live["closes"]:
            print(f"    {c['symbol']:9} {c['direction']:5} entry={c['entry']} "
                  f"exit={c['exit']} pnl={c['pnl']:+.2f} reason={c['reason']} "
                  f"@{c['ts'].strftime('%m-%d %H:%M')}Z")
    if live["deposits"]:
        print("  ⚠ deposits/withdrawals detected IN WINDOW (excluded from P/L above):")
        for t, d in live["deposits"]:
            print(f"    {t.strftime('%m-%d %H:%M')}Z  {d:+.2f}  "
                  f"← balance step, NOT trading performance")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default=GO_LIVE_START, help="window start, UTC (default go-live)")
    p.add_argument("--end", default=None,
                   help="window end, UTC (default: last closed 4h bar)")
    p.add_argument("--initial", type=float, default=None,
                   help="sim starting balance (default: go-live, or --auto-initial)")
    p.add_argument("--auto-initial", action="store_true",
                   help="use the first in-window live realized balance as the sim initial")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--load-start", default=DEFAULT_LOAD_START,
                   help="candle-load start for warmup (default 2026-07-01)")
    p.add_argument("--modes", default="maker,taker",
                   help="comma list of entry modes to sim (default maker,taker)")
    args = p.parse_args()

    end = args.end or last_closed_4h().strftime("%Y-%m-%d %H:%M")
    start_dt, end_dt = _utc(args.start), _utc(end)

    live = parse_live_actuals(args.log_dir, start_dt, end_dt)

    initial = args.initial
    if initial is None:
        initial = (live["first_balance"] if args.auto_initial and live["first_balance"]
                   else GO_LIVE_BALANCE)

    print(f"Live window {args.start} -> {end} UTC | sim initial ${initial:.2f} | "
          f"real Bitget minimums | funding omitted")

    show_live(live)
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        res, gran = run_sim(mode, initial, args.start, end, args.load_start)
        label = {"maker": "maker (fill-on-touch = expected)",
                 "taker": "taker (guaranteed fills = counterfactual)"}.get(mode, mode)
        show_sim(f"{label}  [exit_granularity={gran}]", res, initial)

    print("\nRead trading P/L, not balance: live P/L excludes the deposits flagged "
          "above; sim P/L is pure trading on the same window.")
    if live["deposits"]:
        print("⚠ A deposit landed inside this window — absolute-balance comparison is "
              "meaningless here; the trading-P/L lines are the valid comparison.")
    print("Note: sim uses the CURRENTLY deployed config across the whole window; if live "
          "spanned a config change, pre-change bars won't match bar-exactly.")


if __name__ == "__main__":
    main()
