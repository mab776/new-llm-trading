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
- llt_realized_gains_usdt / llt_realized_losses_usdt / llt_realized_net_usdt   (sign-split PnL)
- llt_lot_closed_wins / llt_lot_closed_losses   (win/loss trade counts)

Run: PYTHONPATH=. python3 -m llm_trading_bot.metrics_exporter \
        [--log-dir logs] [--port 9105]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
import urllib.parse
import urllib.request
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


_TICKER_CACHE: dict = {"ts": 0.0, "prices": {}}


def _mark_prices() -> dict[str, float]:
    """Last price per contract from Bitget's PUBLIC tickers endpoint, ~10s cache."""
    now = time.monotonic()
    if now - _TICKER_CACHE["ts"] < 10 and _TICKER_CACHE["prices"]:
        return _TICKER_CACHE["prices"]
    url = ("https://api.bitget.com/api/v2/mix/market/tickers?"
           "productType=usdt-futures")
    with urllib.request.urlopen(url, timeout=5) as resp:
        payload = json.load(resp)
    prices = {row["symbol"]: float(row.get("lastPr") or 0)
              for row in payload.get("data") or []}
    _TICKER_CACHE.update(ts=now, prices=prices)
    return prices


def _live_state(log_dir: str) -> dict:
    try:
        return json.load(open(os.path.join(log_dir, "shared_live_state.json"),
                              encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def collect(log_dir: str) -> str:
    started = time.monotonic()
    heartbeat_ts: dict[str, float] = {}
    latest_beat: dict = {}
    latest_beat_ts = 0.0
    decisions: dict[tuple[str, str], int] = {}
    closed_count: dict[tuple[str, str], int] = {}
    closed_pnl: dict[tuple[str, str], float] = {}
    realized_gains = 0.0   # sum of positive closed-lot PnL
    realized_losses = 0.0  # sum of negative closed-lot PnL (<= 0)
    win_count = 0
    loss_count = 0
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
                    pnl = float(rec.get("net_pnl_est") or 0.0)
                    closed_pnl[key] = closed_pnl.get(key, 0.0) + pnl
                    if pnl > 0:
                        realized_gains += pnl
                        win_count += 1
                    else:
                        realized_losses += pnl
                        loss_count += 1

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
    for field in ("positions", "consecutive_losses", "cooldown_remaining"):
        samples = [({"symbol": s}, float(beat[field]))
                   for s, beat in sorted(per_symbol_beat.items())
                   if field in beat]
        if samples:
            emit(f"llt_{field}", "gauge",
                 f"{field} from each symbol's newest heartbeat", samples)

    # Live (scrape-cadence) series from the state file + PUBLIC mark prices —
    # much fresher than the 15-min heartbeat. Credential-free by design.
    state = _live_state(log_dir)
    lots_by_symbol: dict[str, list] = {s: [] for s in _SYMBOLS}
    for lot in (state.get("lots") or {}).values():
        lots_by_symbol.setdefault(lot.get("symbol", "unknown"), []).append(lot)
    pending_by_symbol: dict[str, int] = {s: 0 for s in _SYMBOLS}
    for order in (state.get("pending_orders") or {}).values():
        sym = order.get("symbol", "unknown")
        pending_by_symbol[sym] = pending_by_symbol.get(sym, 0) + 1
    emit("llt_open_lots", "gauge", "Open lots (live, from state file)",
         [({"symbol": s}, float(len(lots)))
          for s, lots in sorted(lots_by_symbol.items())])
    emit("llt_pending_orders", "gauge", "Pending entry orders (live, from state file)",
         [({"symbol": s}, float(n)) for s, n in sorted(pending_by_symbol.items())])
    try:
        prices = _mark_prices()
    except Exception:
        prices = {}
    if prices:
        upnl_samples, mark_samples = [], []
        for sym in sorted(lots_by_symbol):
            mark = prices.get(_rest_symbol(sym))
            if mark is None:
                continue
            mark_samples.append(({"symbol": sym}, mark))
            upnl = sum(
                (mark - float(lot["entry"])) * float(lot["remaining_size"])
                * (1 if lot.get("direction") == "LONG" else -1)
                for lot in lots_by_symbol[sym]
            )
            upnl_samples.append(({"symbol": sym}, round(upnl, 6)))
        emit("llt_mark_price", "gauge", "Last price (Bitget public tickers)",
             mark_samples)
        emit("llt_unrealized_pnl_usdt", "gauge",
             "Mark-to-market PnL of open lots (live, excludes fees)",
             upnl_samples)
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
    emit("llt_realized_gains_usdt", "gauge",
         "Sum of positive closed-lot PnL (total gained)", [({}, round(realized_gains, 6))])
    emit("llt_realized_losses_usdt", "gauge",
         "Sum of negative closed-lot PnL (total lost, <= 0)", [({}, round(realized_losses, 6))])
    emit("llt_realized_net_usdt", "gauge",
         "Net realized trading PnL (gains + losses, deposit-independent)",
         [({}, round(realized_gains + realized_losses, 6))])
    emit("llt_lot_closed_wins", "gauge", "Count of profitable closed lots",
         [({}, float(win_count))])
    emit("llt_lot_closed_losses", "gauge", "Count of losing closed lots",
         [({}, float(loss_count))])
    emit("llt_decision_files", "gauge", "decisions-*.jsonl files parsed",
         [({}, float(len(files)))])
    emit("llt_scrape_parse_seconds", "gauge", "Time spent parsing the logs",
         [({}, round(time.monotonic() - started, 4))])
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Trading view (/chart): Bitget-style candles with the bot's TP/SL drawn on.
# /candles proxies Bitget's PUBLIC market endpoint (no credentials involved);
# /levels reads the bot's own state + decisions logs. Same stdlib-only rule.

_GRAN = {"1h": ("1H", 3600), "4h": ("4H", 14400), "1d": ("1D", 86400)}
_SYMBOLS = ("BTC-USDT", "ETH-USDT", "SOL-USDT")


def _rest_symbol(symbol: str) -> str:
    return symbol.split(":", 1)[0].replace("/", "").replace("-", "").upper()


def fetch_candles(symbol: str, tf: str, limit: int = 300) -> list[dict]:
    gran, _ = _GRAN[tf]
    url = ("https://api.bitget.com/api/v2/mix/market/candles?"
           + urllib.parse.urlencode({
               "symbol": _rest_symbol(symbol), "productType": "usdt-futures",
               "granularity": gran, "limit": min(int(limit), 1000)}))
    with urllib.request.urlopen(url, timeout=10) as resp:
        payload = json.load(resp)
    out = []
    for row in payload.get("data") or []:
        ts, o, h, low, c = (float(row[0]) / 1000, float(row[1]),
                            float(row[2]), float(row[3]), float(row[4]))
        if h < low:  # defensive: never trust column order blindly
            h, low = low, h
        out.append({"time": int(ts), "open": o, "high": h, "low": low, "close": c})
    out.sort(key=lambda c: c["time"])
    return out


def levels(symbol: str, log_dir: str) -> dict:
    """Open lots + pending orders (price lines) and fills/closes (markers)."""
    lots, pending = [], []
    state_path = os.path.join(log_dir, "shared_live_state.json")
    try:
        state = json.load(open(state_path, encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        state = {}
    for lot in (state.get("lots") or {}).values():
        if lot.get("symbol") != symbol:
            continue
        lots.append({
            "direction": lot.get("direction"),
            "entry": lot.get("entry"), "sl": lot.get("current_sl"),
            "tp1": lot.get("take_profit_1"), "tp2": lot.get("take_profit_2"),
            "size": lot.get("remaining_size"),
        })
    for order in (state.get("pending_orders") or {}).values():
        if order.get("symbol") != symbol:
            continue
        pending.append({
            "direction": order.get("direction"), "entry": order.get("entry"),
            "sl": order.get("stop_loss"), "tp1": order.get("take_profit_1"),
            "tp2": order.get("take_profit_2"), "size": order.get("size"),
        })
    markers = []
    for path in sorted(glob.glob(os.path.join(log_dir, "decisions-*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("symbol") != symbol:
                    continue
                action = rec.get("action")
                if action == "MAKER_FILL":
                    markers.append({
                        "time": int(_ts(rec)), "kind": "fill",
                        "side": rec.get("side"), "price": rec.get("entry"),
                        "size": rec.get("size")})
                elif action == "LOT_CLOSED":
                    markers.append({
                        "time": int(_ts(rec)), "kind": "close",
                        "reason": rec.get("reason"), "price": rec.get("exit_price"),
                        "pnl": rec.get("net_pnl_est")})
    return {"symbol": symbol, "lots": lots, "pending": pending,
            "markers": markers[-300:]}


_CHART_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>llt — candles + TP/SL</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://unpkg.com/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
 body{margin:0;background:#111418;color:#d8dee6;font:14px system-ui,sans-serif}
 #bar{display:flex;flex-wrap:wrap;gap:.4em;align-items:center;padding:.5em .7em}
 .grp{display:inline-flex;flex-wrap:wrap;gap:.4em}
 .sep{opacity:.4}
 button{background:#22262c;color:#d8dee6;border:1px solid #3a3f46;border-radius:6px;
        padding:.45em .8em;min-height:38px;font-size:14px;line-height:1;
        cursor:pointer;flex:0 0 auto}
 button.on{background:#2f6feb;border-color:#2f6feb;color:#fff}
 #meta{flex-basis:100%;opacity:.75;font-size:.85em}
 #chart{position:absolute;top:52px;bottom:0;left:0;right:0}
</style></head><body>
<div id="bar">
 <span id="syms" class="grp"></span> <span class="sep">|</span> <span id="tfs" class="grp"></span>
 <span id="meta"></span>
</div>
<div id="chart"></div>
<script>
const SYMS=["BTC-USDT","ETH-USDT","SOL-USDT"], TFS=["1h","4h","1d"];
let sym=localStorage.sym||"BTC-USDT", tf=localStorage.tf||"4h";
const chart=LightweightCharts.createChart(document.getElementById("chart"),{
  autoSize:true,
  layout:{background:{color:"#111418"},textColor:"#d8dee6"},
  grid:{vertLines:{color:"#1d2126"},horzLines:{color:"#1d2126"}},
  timeScale:{timeVisible:true,secondsVisible:false},
  crosshair:{mode:LightweightCharts.CrosshairMode.Normal}});
const series=chart.addCandlestickSeries({
  upColor:"#26a69a",downColor:"#ef5350",borderVisible:false,
  wickUpColor:"#26a69a",wickDownColor:"#ef5350"});
let priceLines=[];
function chip(box,items,cur,cb){
  const el=document.getElementById(box); el.innerHTML="";
  for(const it of items){const b=document.createElement("button");
    b.textContent=it; if(it===cur)b.classList.add("on");
    b.onclick=()=>cb(it); el.appendChild(b);}
}
async function load(){
  chip("syms",SYMS,sym,s=>{sym=s;localStorage.sym=s;load();});
  chip("tfs",TFS,tf,t=>{tf=t;localStorage.tf=t;load();});
  const [candles,lv]=await Promise.all([
    fetch(`/candles?symbol=${sym}&tf=${tf}`).then(r=>r.json()),
    fetch(`/levels?symbol=${sym}`).then(r=>r.json())]);
  series.setData(candles);
  for(const l of priceLines) series.removePriceLine(l); priceLines=[];
  const line=(price,color,title,style)=>{ if(price==null)return;
    priceLines.push(series.createPriceLine({price,color,title,
      lineStyle:style??LightweightCharts.LineStyle.Dashed,lineWidth:1}));};
  for(const lot of lv.lots){
    line(lot.entry,"#9aa4b2",`entry ${lot.size}`,LightweightCharts.LineStyle.Solid);
    line(lot.sl,"#ef5350","SL"); line(lot.tp1,"#26a69a","TP1");
    line(lot.tp2,"#66bb6a","TP2");}
  for(const o of lv.pending){
    line(o.entry,"#2f6feb",`pending ${o.direction} ${o.size}`);
    line(o.sl,"#7a3a3a","SL (preset)"); line(o.tp1,"#2a5f5a","TP1 (preset)");}
  const tfSec={"1h":3600,"4h":14400,"1d":86400}[tf];
  const t0=candles.length?candles[0].time:0;
  series.setMarkers(lv.markers.filter(m=>m.time>=t0).map(m=>({
    time:m.time-(m.time%tfSec),
    position:m.kind==="fill"?"belowBar":"aboveBar",
    color:m.kind==="fill"?"#2f6feb":(m.pnl>=0?"#26a69a":"#ef5350"),
    shape:m.kind==="fill"?"arrowUp":"arrowDown",
    text:m.kind==="fill"?`fill @${m.price}`:`${m.reason} ${m.pnl>=0?"+":""}${(+m.pnl).toFixed(2)}`})));
  const last=candles.at(-1);
  document.getElementById("meta").textContent=
    `${sym} ${tf} — last ${last?last.close:"?"} · lots ${lv.lots.length} · pending ${lv.pending.length}`;
  fitChart();
}
function fitChart(){document.getElementById("chart").style.top=document.getElementById("bar").offsetHeight+"px";}
addEventListener("resize",fitChart);
load(); setInterval(load,60000);
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--port", type=int, default=9105)
    args = parser.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, body: bytes, ctype: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            url = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(url.query)
            route = url.path.rstrip("/")
            try:
                if route in ("", "/metrics"):
                    self._reply(collect(args.log_dir).encode(),
                                "text/plain; version=0.0.4")
                elif route == "/chart":
                    self._reply(_CHART_HTML.encode(), "text/html; charset=utf-8")
                elif route == "/candles":
                    symbol = q.get("symbol", ["BTC-USDT"])[0]
                    tf = q.get("tf", ["4h"])[0]
                    if symbol not in _SYMBOLS or tf not in _GRAN:
                        self.send_error(400, "unknown symbol or tf")
                        return
                    self._reply(json.dumps(fetch_candles(symbol, tf)).encode(),
                                "application/json")
                elif route == "/levels":
                    symbol = q.get("symbol", ["BTC-USDT"])[0]
                    if symbol not in _SYMBOLS:
                        self.send_error(400, "unknown symbol")
                        return
                    self._reply(json.dumps(levels(symbol, args.log_dir)).encode(),
                                "application/json")
                else:
                    self.send_error(404)
            except Exception as exc:  # keep serving through transient failures
                if route in ("", "/metrics"):
                    self._reply(f"# collect failed: {exc}\nllt_up 0\n".encode(),
                                "text/plain; version=0.0.4", 500)
                else:
                    self._reply(json.dumps({"error": str(exc)}).encode(),
                                "application/json", 500)

        def log_message(self, *_args) -> None:  # quiet
            pass

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"llt metrics exporter on :{args.port}, log dir {args.log_dir!r}")
    server.serve_forever()


if __name__ == "__main__":
    main()
