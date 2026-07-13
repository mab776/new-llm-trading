"""Durable state shared by live symbol schedulers."""

from __future__ import annotations

import json
import threading
from pathlib import Path


class SharedLiveState:
    """Thread-safe state persisted with atomic file replacement."""

    VERSION = 2

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.peak_balance = 0.0
        self.pending_orders: dict[str, dict] = {}
        self.tracked_trades: dict[str, dict] = {}
        self.last_analysis_bars: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        try:
            if not self.path.exists():
                return
            payload = json.loads(self.path.read_text())
            if not isinstance(payload, dict):
                return
            self.peak_balance = max(0.0, float(payload.get("peak_balance", 0) or 0))
            pending = payload.get("pending_orders", {})
            tracked = payload.get("tracked_trades", {})
            analysis = payload.get("last_analysis_bars", {})
            if isinstance(pending, dict):
                self.pending_orders.update(pending)
            if isinstance(tracked, dict):
                self.tracked_trades.update(tracked)
            if isinstance(analysis, dict):
                self.last_analysis_bars.update(
                    (str(symbol), str(timestamp))
                    for symbol, timestamp in analysis.items()
                )
        except (OSError, ValueError, TypeError):
            self.peak_balance = 0.0
            self.pending_orders.clear()
            self.tracked_trades.clear()
            self.last_analysis_bars.clear()

    def save(self) -> None:
        """Persist current state atomically."""
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": self.VERSION,
                "peak_balance": self.peak_balance,
                "pending_orders": self.pending_orders,
                "tracked_trades": self.tracked_trades,
                "last_analysis_bars": self.last_analysis_bars,
            }
            temp = self.path.with_name(f".{self.path.name}.tmp")
            temp.write_text(json.dumps(payload, indent=2, sort_keys=True))
            temp.replace(self.path)

    def update_peak(self, realized_balance: float) -> float:
        """Persist a new portfolio-wide realized-balance peak when observed."""
        if realized_balance <= 0:
            return self.peak_balance
        with self.lock:
            if realized_balance > self.peak_balance:
                self.peak_balance = realized_balance
                self.save()
            return self.peak_balance
