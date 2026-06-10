"""
Nyx — Rate limiter, retry with exponential backoff, and robust timeout.

Provides:
- Token-bucket rate limiter for API calls
- Exponential backoff with jitter for retries
- Robust timeout context manager for HTTP calls
"""
from __future__ import annotations

import logging
import random
import threading
import time
from collections import deque
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Token-bucket rate limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """Token-bucket rate limiter.

    Allows *rate* requests per second, with a burst capacity of *burst*.
    Thread-safe.
    """

    def __init__(self, rate: float = 10.0, burst: int = 20) -> None:
        """
        Args:
            rate: Maximum sustained requests per second.
            burst: Maximum burst size (token bucket capacity).
        """
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0, block: bool = True) -> bool:
        """Acquire *tokens* from the bucket.

        If *block* is True, blocks until tokens are available.
        Returns True if tokens were acquired, False otherwise.
        """
        if not block:
            return self._try_acquire(tokens)

        while True:
            if self._try_acquire(tokens):
                return True
            # Sleep for a short interval before retrying
            sleep_time = self._estimate_wait(tokens)
            time.sleep(max(sleep_time, 0.001))

    def _try_acquire(self, tokens: float = 1.0) -> bool:
        """Non-blocking attempt to acquire tokens."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
            self._last_refill = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def _estimate_wait(self, tokens: float = 1.0) -> float:
        """Estimate how long until *tokens* are available."""
        with self._lock:
            deficit = tokens - self._tokens
            if deficit <= 0:
                return 0.0
            return deficit / self._rate if self._rate > 0 else float("inf")

    @property
    def rate(self) -> float:
        return self._rate

    @rate.setter
    def rate(self, value: float) -> None:
        with self._lock:
            self._rate = value


# ---------------------------------------------------------------------------
# Exponential backoff with jitter
# ---------------------------------------------------------------------------


def exponential_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
) -> float:
    """Calculate delay for retry attempt using exponential backoff.

    delay = min(base_delay * 2^attempt, max_delay)
    If jitter is True, adds random jitter: delay *= random[0.5, 1.5)
    """
    delay = min(base_delay * (2 ** attempt), max_delay)
    if jitter:
        delay *= random.uniform(0.5, 1.5)
    return delay


def retry_with_backoff(
    fn: Callable[..., T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
    ),
    on_retry: Callable[[int, Exception, float], None] | None = None,
    **kwargs: Any,
) -> T:
    """Execute *fn* with retry and exponential backoff.

    Args:
        fn: The callable to execute.
        max_retries: Maximum number of retries (0 = no retry).
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay in seconds.
        retryable_exceptions: Tuple of exception types that trigger a retry.
        on_retry: Optional callback(attempt, exception, delay) called before each retry.
        **kwargs: Passed through to fn.

    Returns:
        The return value of fn.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return fn(**kwargs)
        except retryable_exceptions as e:
            last_exc = e
            if attempt < max_retries:
                delay = exponential_backoff(attempt, base_delay, max_delay)
                if on_retry:
                    on_retry(attempt + 1, e, delay)
                logger.warning(
                    "Retry %d/%d after %s: %s",
                    attempt + 1, max_retries, e, delay,
                )
                time.sleep(delay)
            else:
                logger.error("All %d retries exhausted: %s", max_retries, e)
                raise
        except Exception as e:
            # Non-retryable exception — raise immediately
            raise

    # Should not reach here, but just in case
    if last_exc:
        raise last_exc
    raise RuntimeError("Unexpected: retry loop ended without result or exception")


# ---------------------------------------------------------------------------
# Robust timeout context manager
# ---------------------------------------------------------------------------


class TimeoutError(Exception):
    """Raised when an operation times out."""
    pass


class timeout:
    """Context manager that raises TimeoutError if the block takes too long.

    Uses a background thread with a daemon timer. Note: this only works
    for Python code that respects the timeout flag — it does NOT kill
    C extensions or system calls. For those, use signal-based timeouts
    or subprocess timeouts.
    """

    def __init__(self, seconds: float, message: str = "Operation timed out") -> None:
        self._seconds = seconds
        self._message = message
        self._timed_out = False
        self._timer: threading.Timer | None = None

    def __enter__(self) -> "timeout":
        self._timer = threading.Timer(self._seconds, self._trigger)
        self._timer.daemon = True
        self._timer.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool | None:
        if self._timer:
            self._timer.cancel()
            self._timer = None
        if self._timed_out:
            raise TimeoutError(self._message)
        return None

    def _trigger(self) -> None:
        self._timed_out = True
        # Raise an exception in the main thread
        import _thread
        _thread.interrupt_main()


# ---------------------------------------------------------------------------
# Combined: rate-limited + retry + timeout wrapper
# ---------------------------------------------------------------------------


class ResilientClient:
    """Wraps API calls with rate limiting, retry, and timeout.

    Example:
        client = ResilientClient(rate=5.0, max_retries=3)
        result = client.execute(my_api_call, url=..., data=...)
    """

    def __init__(
        self,
        rate: float = 10.0,
        burst: int = 20,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        default_timeout: float = 120.0,
        retryable_exceptions: tuple[type[Exception], ...] | None = None,
    ) -> None:
        self._rate_limiter = RateLimiter(rate=rate, burst=burst)
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._default_timeout = default_timeout
        self._retryable_exceptions = retryable_exceptions or (
            ConnectionError,
            TimeoutError,
            OSError,
        )

    def execute(
        self,
        fn: Callable[..., T],
        timeout_seconds: float | None = None,
        **kwargs: Any,
    ) -> T:
        """Execute *fn* with rate limiting, retry, and timeout."""
        # 1. Rate limit
        self._rate_limiter.acquire()

        # 2. Execute with timeout and retry
        timeout_seconds = timeout_seconds or self._default_timeout

        def _wrapped() -> T:
            with timeout(seconds=timeout_seconds):
                return fn(**kwargs)

        return retry_with_backoff(
            _wrapped,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            max_delay=self._max_delay,
            retryable_exceptions=self._retryable_exceptions,
        )

    @property
    def rate_limiter(self) -> RateLimiter:
        return self._rate_limiter