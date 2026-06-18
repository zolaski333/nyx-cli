"""
Nyx — Rate limiter, retry with exponential backoff, and timeout policy.

Provides:
- Token-bucket rate limiter for API calls
- Exponential backoff with jitter for retries
- Compatibility timeout context; hard timeouts belong to transports
"""
from __future__ import annotations

import logging
import random
import threading
import time
import urllib.error
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
    return float(delay)


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
            if not _is_retryable_exception(e, retryable_exceptions):
                raise
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
        except Exception:
            # Non-retryable exception — raise immediately
            raise

    # Should not reach here, but just in case
    if last_exc:
        raise last_exc
    raise RuntimeError("Unexpected: retry loop ended without result or exception")


def _is_retryable_exception(
    exc: Exception,
    retryable_exceptions: tuple[type[Exception], ...],
) -> bool:
    if not isinstance(exc, retryable_exceptions):
        return False
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or exc.code >= 500
    return True


# ---------------------------------------------------------------------------
# Robust timeout context manager
# ---------------------------------------------------------------------------


class TimeoutError(Exception):
    """Raised when an operation times out."""
    pass


class timeout:
    """Compatibility no-op context manager.

    Hard timeout enforcement belongs to native transports or subprocess APIs.
    This context manager intentionally does not claim to interrupt blocking
    calls from another thread.
    """

    def __init__(self, seconds: float, message: str = "Operation timed out") -> None:
        self._seconds = seconds
        self._message = message

    def __enter__(self) -> "timeout":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool | None:
        return None


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
        retryable_exceptions: tuple[type[Exception], ...] | None = None,
    ) -> None:
        self._rate_limiter = RateLimiter(rate=rate, burst=burst)
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._retryable_exceptions = retryable_exceptions or (
            ConnectionError,
            TimeoutError,
            OSError,
        )

    def execute(
        self,
        fn: Callable[..., T],
        **kwargs: Any,
    ) -> T:
        """Execute *fn* with rate limiting and retry.

        Hard timeout enforcement belongs to the network transport. Pass native
        timeout options to the transport callable itself.
        """
        # 1. Rate limit
        self._rate_limiter.acquire()

        # 2. Execute with retry. Native timeout kwargs, if any, are forwarded to fn.
        return retry_with_backoff(
            fn,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            max_delay=self._max_delay,
            retryable_exceptions=self._retryable_exceptions,
            **kwargs,
        )

    @property
    def rate_limiter(self) -> RateLimiter:
        return self._rate_limiter
