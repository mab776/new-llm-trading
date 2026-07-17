"""Prometheus exporter over the bot's decisions JSONL — the live drift dataset.

Stdlib-only on purpose: it must run wherever the bot runs (x86 server today,
Raspberry Pi later) with no dependency on the trading venv. Each scrape
re-parses every logs/decisions-*.jsonl present (90-day retention bounds this
to a few MB), so the exporter is stateless and restart-safe.

Exposed series (prefix llt_):
- llt_heartbeat_timestamp_seconds{symbol}   freshness -> alert on age
- llt_equity_usdt / llt_realized_balance_usdt / llt_peak_balance_usdt
- llt_open_lots / llt_pending_orders / llt_positions {symbol}
- llt_consecutive_losses / llt_cooldown_remaining {symbol}
- llt_disk_free_mb
- llt_decisions_total{action,symbol}        the maker fill funnel lives here
- llt_lot_closed_total{reason,symbol} + llt_lot_closed_pnl_usdt_total{...}

Run: PYTHONPATH=. python3 -m llm_trading_bot.metrics_exporter \
        [--log-dir logs] [--port 9105]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _ts(record: dict) -> float:
    raw = record.get("timestamp")
    if not raw:
        return 0.0
    try:
        return datetime.fromisoformat(raw).timestamp()
    except ValueError:
        return 0.0


def _fmt_labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return "{" + inner + "}"


def collect(log_dir: str) -> str:
    started = time.monotonic()
    heartbeat_ts: dict[str, float] = {}
    latest_beat: dict = {}
    latest_beat_ts = 0.0
    decisions: dict[tuple[str, str], int] = {}
    closed_count: dict[tuple[str, str], int] = {}
    closed_pnl: dict[tuple[str, str], float] = {}
    per_symbol_beat: dict[str, dict] = {}

    files = sorted(glob.glob(os.path.join(log_dir, "decisions-*.jsonl")))
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                action = str(rec.get("action") or "")
                symbol = str(rec.get("symbol") or "unknown")
                if not action:
                    continue
                if action == "HEARTBEAT":
                    ts = _ts(rec)
                    if ts >= heartbeat_ts.get(symbol, 0.0):
                        heartbeat_ts[symbol] = ts
                        per_symbol_beat[symbol] = rec
                    if ts >= latest_beat_ts:
                        latest_beat_ts, latest_beat = ts, rec
                    continue
                decisions[(action, symbol)] = decisions.get((action, symbol), 0) + 1
                if action == "LOT_CLOSED":
                    reason = str(rec.get("reason") or "unknown")
                    key = (reason, symbol)
                    closed_count[key] = closed_count.get(key, 0) + 1
                    closed_pnl[key] = closed_pnl.get(key, 0.0) + float(
                        rec.get("net_pnl_est") or 0.0
                    )

    out: list[str] = []

    def emit(name: str, kind: str, help_text: str,
             samples: list[tuple[dict, float]]) -> None:
        out.append(f"# HELP {name} {help_text}")
        out.append(f"# TYPE {name} {kind}")
        for labels, value in samples:
            out.append(f"{name}{_fmt_labels(labels)} {value}")

    emit("llt_heartbeat_timestamp_seconds", "gauge",
         "Unix time of the newest HEARTBEAT per symbol",
         [({"symbol": s}, ts) for s, ts in sorted(heartbeat_ts.items())])
    for field, help_text in (
        ("equity", "Account equity from the newest heartbeat"),
        ("realized_balance", "Realized balance from the newest heartbeat"),
        ("peak_balance", "Peak balance (drawdown anchor)"),
    ):
        if field in latest_beat:
            emit(f"llt_{field}_usdt", "gauge", help_text,
                 [({}, float(latest_beat[field]))])
    if "disk_free_mb" in latest_beat:
        emit("llt_disk_free_mb", "gauge", "Free disk in the log dir",
             [({}, float(latest_beat["disk_free_mb"]))])
    for field in ("open_lots", "pending_orders", "positions",
                  "consecutive_losses", "cooldown_remaining"):
        samples = [({"symbol": s}, float(beat[field]))
                   for s, beat in sorted(per_symbol_beat.items())
                   if field in beat]
        if samples:
            emit(f"llt_{field}", "gauge",
                 f"{field} from each symbol's newest heartbeat", samples)
    emit("llt_decisions_total", "counter",
         "Decision records by action and symbol (full retention window)",
         [({"action": a, "symbol": s}, float(n))
          for (a, s), n in sorted(decisions.items())])
    emit("llt_lot_closed_total", "counter", "Closed lots by exit reason",
         [({"reason": r, "symbol": s}, float(n))
          for (r, s), n in sorted(closed_count.items())])
    emit("llt_lot_closed_pnl_usdt_total", "counter",
         "Net estimated PnL summed by exit reason",
         [({"reason": r, "symbol": s}, round(v, 6))
          for (r, s), v in sorted(closed_pnl.items())])
    emit("llt_decision_files", "gauge", "decisions-*.jsonl files parsed",
         [({}, float(len(files)))])
    emit("llt_scrape_parse_seconds", "gauge", "Time spent parsing the logs",
         [({}, round(time.monotonic() - started, 4))])
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--port", type=int, default=9105)
    args = parser.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            if self.path.rstrip("/") not in ("", "/metrics"):
                self.send_error(404)
                return
            try:
                body = collect(args.log_dir).encode()
            except Exception as exc:  # keep serving through transient log issues
                body = f"# collect failed: {exc}\nllt_up 0\n".encode()
                self.send_response(500)
            else:
                self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args) -> None:  # quiet
            pass

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"llt metrics exporter on :{args.port}, log dir {args.log_dir!r}")
    server.serve_forever()


if __name__ == "__main__":
    main()
