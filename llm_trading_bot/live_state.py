"""Durable state shared by live symbol schedulers."""

from __future__ import annotations

import json
import threading
from hashlib import sha256
from pathlib import Path


class LiveStateError(RuntimeError):
    """Raised when persisted live safety state cannot be trusted."""


class SharedLiveState:
    """Thread-safe state persisted with atomic file replacement."""

    VERSION = 4

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.peak_balance = 0.0
        self.pending_orders: dict[str, dict] = {}
        # Keyed by stable lot ID (normally the entry clientOid), never by symbol.
        # ``tracked_trades`` remains an alias for compatibility with older callers.
        self.lots: dict[str, dict] = {}
        self.tracked_trades = self.lots
        self.last_analysis_bars: dict[str, str] = {}
        # Per-symbol backtest-parity risk counters (cooldown after SL, consecutive
        # losses for the entry-threshold penalty). Added in version 4; older files
        # load with empty counters.
        self.risk_counters: dict[str, dict] = {}
        self._load()

    @staticmethod
    def _legacy_lot_id(key: str, lot: dict) -> str:
        stable = str(lot.get("client_oid") or lot.get("order_id") or key)
        return "legacy-" + sha256(stable.encode()).hexdigest()[:24]

    def _load_lots(self, payload: dict, version: int) -> None:
        raw = payload.get("lots") if version >= 3 else payload.get("tracked_trades", {})
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise LiveStateError("Persisted live lots must be a JSON object")
        for key, value in raw.items():
            if not isinstance(value, dict):
                raise LiveStateError(f"Persisted live lot {key!r} is not an object")
            lot = dict(value)
            if version < 3:
                lot.setdefault("symbol", str(key))
                lot.setdefault("lifecycle", "unreconciled")
                lot.setdefault("protection_verified", False)
                lot_id = self._legacy_lot_id(str(key), lot)
            else:
                lot_id = str(key)
            self.lots[lot_id] = lot

    def _load(self) -> None:
        try:
            if not self.path.exists():
                return
            payload = json.loads(self.path.read_text())
            if not isinstance(payload, dict):
                raise LiveStateError("Persisted live state must be a JSON object")
            version = int(payload.get("version", 1) or 1)
            if version > self.VERSION:
                raise LiveStateError(
                    f"Live state version {version} is newer than supported version {self.VERSION}"
                )
            self.peak_balance = max(0.0, float(payload.get("peak_balance", 0) or 0))
            pending = payload.get("pending_orders", {})
            analysis = payload.get("last_analysis_bars", {})
            if not isinstance(pending, dict) or not isinstance(analysis, dict):
                raise LiveStateError("Persisted pending orders and analysis bars must be objects")
            self.pending_orders.update(pending)
            self._load_lots(payload, version)
            self.last_analysis_bars.update(
                (str(symbol), str(timestamp))
                for symbol, timestamp in analysis.items()
            )
            counters = payload.get("risk_counters", {})
            if counters is None:
                counters = {}
            if not isinstance(counters, dict) or not all(
                isinstance(value, dict) for value in counters.values()
            ):
                raise LiveStateError("Persisted risk counters must be objects")
            self.risk_counters.update(
                (str(symbol), dict(value)) for symbol, value in counters.items()
            )
        except LiveStateError:
            raise
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise LiveStateError(
                f"Cannot trust persisted live state at {self.path}: {exc}"
            ) from exc

    def save(self) -> None:
        """Persist current state atomically."""
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": self.VERSION,
                "peak_balance": self.peak_balance,
                "pending_orders": self.pending_orders,
                "lots": self.lots,
                "last_analysis_bars": self.last_analysis_bars,
                "risk_counters": self.risk_counters,
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
