"""Account-scoped live-process lock.

One Bitget account must never be traded by two bot processes at once: the
shared exposure caps, peak-balance drawdown throttle, and per-lot lifecycle
state all assume a single writer. The previous lock was tied to a *log
directory*, so pointing a second process at another directory silently
bypassed it. This lock derives its identity from the exchange account itself
(api key + demo flag + product type), lives in the system temp directory, and
is enforced by both the standalone scheduler and the shared orchestrator.
"""

from __future__ import annotations

import fcntl
import hashlib
import tempfile
from pathlib import Path
from typing import IO


class AccountLockError(RuntimeError):
    """Another live process already owns this exchange account."""


def account_lock_path(bitget_config) -> Path:
    """Deterministic lock-file path for one exchange account."""
    material = "|".join((
        bitget_config.api_key,
        str(bitget_config.testnet),
        bitget_config.product_type,
    ))
    digest = hashlib.sha256(material.encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"llm-trading-bot-{digest}.lock"


def acquire_account_lock(bitget_config) -> IO:
    """Take the exclusive account lock or raise AccountLockError.

    The returned handle must stay referenced for the lifetime of the process;
    release it with :func:`release_account_lock`.
    """
    path = account_lock_path(bitget_config)
    handle = open(path, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise AccountLockError(
            "Another live trading process already owns this Bitget account"
        ) from exc
    return handle


def release_account_lock(handle: IO | None) -> None:
    """Release and close a lock previously returned by acquire_account_lock."""
    if handle is None:
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()
