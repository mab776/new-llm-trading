"""One-process, shared-state orchestration for multi-symbol live trading."""

from __future__ import annotations

import fcntl
import time
from pathlib import Path

import schedule

from llm_trading_bot.config import AppConfig
from llm_trading_bot.live_state import SharedLiveState
from llm_trading_bot.scheduler import TradingScheduler


class SharedTradingOrchestrator:
    """Run several symbol schedulers sequentially against one account state."""

    def __init__(self, configs: list[AppConfig], log_dir: str | Path = "logs"):
        if not configs:
            raise ValueError("At least one live config is required")
        self.configs = configs
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._validate_configs()
        self.state = SharedLiveState(self.log_dir / "shared_live_state.json")
        self.schedulers = [
            TradingScheduler(config, shared_state=self.state, log_dir=self.log_dir)
            for config in configs
        ]
        self.interval_minutes = configs[0].scheduling.interval_minutes
        self.analysis_poll_minutes = 1
        self.position_interval_minutes = (
            configs[0].scheduling.check_positions_interval_minutes
        )
        self._lock_handle = None

    def _validate_configs(self) -> None:
        symbols = [config.trading.symbol for config in self.configs]
        if len(symbols) != len(set(symbols)):
            raise ValueError("Shared live configs must use unique trading symbols")
        first = self.configs[0]
        account = (
            first.bitget.api_key, first.bitget.api_secret,
            first.bitget.passphrase, first.bitget.testnet,
            first.bitget.product_type, first.bitget.position_mode,
            first.bitget.margin_mode,
        )
        cadence = (
            first.scheduling.interval_minutes,
            first.scheduling.check_positions_interval_minutes,
        )
        for config in self.configs[1:]:
            other_account = (
                config.bitget.api_key, config.bitget.api_secret,
                config.bitget.passphrase, config.bitget.testnet,
                config.bitget.product_type, config.bitget.position_mode,
                config.bitget.margin_mode,
            )
            if other_account != account:
                raise ValueError("Shared live configs must use the same Bitget account")
            other_cadence = (
                config.scheduling.interval_minutes,
                config.scheduling.check_positions_interval_minutes,
            )
            if other_cadence != cadence:
                raise ValueError("Shared live configs must use the same scheduling cadence")

    def _acquire_process_lock(self) -> None:
        lock_path = self.log_dir / "shared_orchestrator.lock"
        self._lock_handle = open(lock_path, "a+")
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_handle.close()
            self._lock_handle = None
            raise RuntimeError(
                "Another shared trading orchestrator already owns this log directory"
            ) from exc

    def run_cycle(self) -> None:
        for scheduler in self.schedulers:
            scheduler.run_cycle()

    def check_positions(self) -> None:
        for scheduler in self.schedulers:
            scheduler.check_positions()

    def start(self) -> None:
        """Start one scheduling loop for the whole account."""
        self._acquire_process_lock()
        symbols = ", ".join(config.trading.symbol for config in self.configs)
        print(f"Shared live orchestrator: {symbols}")
        # Complete every symbol's account/exchange reconciliation before any one
        # scheduler is allowed to fetch a signal or place a new order.
        allowed_symbols = {config.trading.symbol for config in self.configs}
        for scheduler in self.schedulers:
            scheduler.reconcile_startup(allowed_symbols=allowed_symbols)
        self.run_cycle()
        schedule.every(self.analysis_poll_minutes).minutes.do(self.run_cycle)
        schedule.every(self.position_interval_minutes).minutes.do(self.check_positions)
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shared trading orchestrator stopped by user")
        finally:
            if self._lock_handle is not None:
                fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
                self._lock_handle.close()
                self._lock_handle = None
