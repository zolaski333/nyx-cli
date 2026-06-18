"""Shared interactive approval serialization."""
from __future__ import annotations

import threading
from typing import Callable, TypeVar

T = TypeVar("T")

_approval_lock = threading.RLock()


def run_exclusive_approval(callback: Callable[[], T]) -> T:
    """Run one approval prompt at a time across CLI frontends and subagents."""
    with _approval_lock:
        return callback()
