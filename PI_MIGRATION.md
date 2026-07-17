# Migrating the live bot to the Raspberry Pi

Reference for a future session. Goal: run the **live loop + metrics exporter** on the
battery-backed Pi (WiFi stays up in a power outage → the bot keeps managing trailing
stops and entries while the big server is down). Research stays on the server —
backtests, `opt/` sims, and parity runs are far too slow on the Pi.

## Feasibility (verified 2026-07-16, don't re-derive)

Target is the **Pi 400** at `192.168.0.75` (SSH user `mab776pi`, password auth; also
runs Pi-hole + PiVPN + wstunnel — no port conflicts, the bot listens on nothing and the
exporter uses :9105 which is free there). Verified: aarch64, Python 3.13.5, 3.2 GB RAM
free, 47 GB disk free, 4× Cortex-A72, **NTP synced** (preflight refuses >30 s clock
drift). Bot footprint measured on the server: ~190 MB RSS, ~0.1% average CPU, logs
~0.5 MB/day. All deps in `requirements.txt` have native aarch64 wheels.

## ⚠️ The one thing that can actually go wrong

**The account lock does NOT protect across machines.** `process_lock.py` uses
`fcntl.flock` on a file in the *local* tempdir — two hosts have two tempdirs, so a bot
on the server and a bot on the Pi will happily trade the same account simultaneously
(double exposure, corrupted per-lot state). The protection is purely procedural:

> **Stop the server bot and confirm its process is gone BEFORE starting the Pi bot.
> Never leave both configured to autostart.**

After migrating, remove/disable the server-side launch (tmux session, and the Claude-RC
autostart note if any) so a server reboot can't resurrect a second copy.

## Pre-migration checklist

- [ ] Pick a moment **mid-bar with no pending maker order** (they're good-for-one-bar;
      check `llt_pending_orders` in Grafana or `logs/shared_live_state.json`
      `pending_orders: {}`). Open *positions* are fine — restart-onto-open-position was
      validated 2026-07-16 (reconcile adopts lots, verifies protection, cleans presets).
- [ ] `timedatectl show -p NTPSynchronized` on the Pi → `yes`.
- [ ] If the Bitget keys ever get IP-restricted (planned hardening), add the Pi's
      IP/egress first.
- [ ] GitHub access from the Pi: repo is private (`git@github.com:mab776/new-llm-trading`).
      Either a read-only deploy key on the Pi or `git archive`/rsync from the server.

## Migration steps

```bash
# 1. On the Pi — code + venv (rsync from server avoids the deploy-key question;
#    --exclude history/ skips the multi-GB backtest cache the live loop doesn't need)
rsync -a --exclude history/ --exclude .git --exclude logs \
    mab776@192.168.0.70:~/Documents/new-llm-trading/ ~/new-llm-trading/
cd ~/new-llm-trading
python3 -m venv ~/llt-venv && ~/llt-venv/bin/pip install -r requirements.txt
# Fresh install on Py3.13 pulls newer lib versions than the server's Py3.12 venv:
PYTHONPATH=. ~/llt-venv/bin/python -m pytest -q        # must be all-green before go-live

# 2. Secrets + state (gitignored, so rsync above already copied config*.local.json —
#    verify mode 600; they embed the real API keys)
chmod 600 config*.local.json
mkdir -p logs
scp mab776@192.168.0.70:~/Documents/new-llm-trading/logs/shared_live_state.json logs/
# Optional but recommended — carry the decisions history so the Grafana fill-funnel
# counters don't reset to zero (exporter parses these):
scp "mab776@192.168.0.70:~/Documents/new-llm-trading/logs/decisions-*.jsonl" logs/

# 3. STOP the server bot (see warning above) — on the server:
#    tmux kill-session -t trading-bot     (kills bot + metrics windows)
#    ps aux | grep llm_trading_bot        → must be empty before step 4

# 4. On the Pi — systemd units (finally reboot-proof, unlike the server tmux setup)
sudo tee /etc/systemd/system/llt-bot.service >/dev/null <<'EOF'
[Unit]
Description=new-llm-trading live bot (Bitget)
After=network-online.target time-sync.target
Wants=network-online.target time-sync.target

[Service]
User=mab776pi
WorkingDirectory=/home/mab776pi/new-llm-trading
Environment=PYTHONPATH=.
ExecStart=/home/mab776pi/llt-venv/bin/python -m llm_trading_bot.main --mode live \
    --shared-configs config.local.json config-eth.local.json config-sol.local.json
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF
sudo tee /etc/systemd/system/llt-metrics.service >/dev/null <<'EOF'
[Unit]
Description=new-llm-trading metrics exporter
After=network-online.target

[Service]
User=mab776pi
WorkingDirectory=/home/mab776pi/new-llm-trading
Environment=PYTHONPATH=.
ExecStart=/usr/bin/python3 -m llm_trading_bot.metrics_exporter --port 9105
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now llt-metrics llt-bot
journalctl -u llt-bot -f    # watch for "Startup reconciliation passed" ×3 symbols
```

Notes on the units: the bot deliberately gets `Restart=on-failure` (not `always`) —
a `SafetyViolation` refusal loop shouldn't hammer the exchange; the loop crash guard
handles transient errors internally. `time-sync.target` matters for the clock-drift
preflight. The exporter uses **system python3** (it's stdlib-only by design).

## Rewire monitoring (on the server)

1. `prometheus/config/prometheus.yml` (root-owned, sudo): job `llm-trading-bot`
   target `192.168.0.70:9105` → `192.168.0.75:9105` (a comment there marks the line),
   then `docker restart prometheus`.
2. Verify: Prometheus targets page → `llm-trading-bot` **up**; Grafana
   `llt-live-drift` heartbeat age green; alert rule "Trading bot heartbeat stale"
   back to `inactive` (it will fire Pending during the gap — that's the alert working).
3. Update `portainer/CLAUDE.md` + the mab776-portainer skill port-registry row for
   9105, and the memory files (bot location changed).

## Verify go-live

- `journalctl -u llt-bot` → "Startup reconciliation passed" for BTC/ETH/SOL, then
  normal per-minute cycles; no `SafetyViolation`.
- `curl -s localhost:9105/metrics | grep llt_equity` on the Pi.
- Position/TPSL sanity vs exchange (same read-only probe pattern as prior sessions).
- After the next 4h bar close: decisions logged, orders placed (or WAIT), heartbeats
  every 15 min in Grafana.

## Rollback

Reverse order: `sudo systemctl disable --now llt-bot llt-metrics` on the Pi → confirm
dead → copy `logs/shared_live_state.json` (+ new `decisions-*.jsonl`) back to the
server → relaunch server tmux (command in memory / git history) → retarget Prometheus
back to `.70:9105`. Same no-double-run rule in reverse.
