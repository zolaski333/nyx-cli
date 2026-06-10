"""Tests for rate limiter, retry with backoff, and robust timeout."""
from __future__ import annotations

import time

import pytest

from nyx.rate_limiter import (
    RateLimiter,
    exponential_backoff,
    retry_with_backoff,
    timeout,
    TimeoutError,
    ResilientClient,
)


class TestRateLimiter:
    """Test the token-bucket rate limiter."""

    def test_acquire_allows_burst(self):
        """Should allow burst of tokens up to capacity."""
        limiter = RateLimiter(rate=10.0, burst=20)
        for _ in range(20):
            assert limiter.acquire(block=False) is True

    def test_acquire_blocks_when_exhausted(self):
        """Should block when burst capacity is exhausted."""
        limiter = RateLimiter(rate=1000.0, burst=5)
        for _ in range(5):
            assert limiter.acquire(block=False) is True
        # Next acquire should fail (non-blocking)
        assert limiter.acquire(block=False) is False

    def test_acquire_blocking_eventually_succeeds(self):
        """Blocking acquire should eventually succeed after refill."""
        limiter = RateLimiter(rate=100.0, burst=1)
        assert limiter.acquire(block=False) is True
        # Blocking acquire should wait for refill (very short since rate is high)
        start = time.time()
        assert limiter.acquire(block=True) is True
        elapsed = time.time() - start
        assert elapsed < 0.1  # Should be very fast with rate=100

    def test_rate_property(self):
        """Should get/set rate."""
        limiter = RateLimiter(rate=5.0)
        assert limiter.rate == 5.0
        limiter.rate = 10.0
        assert limiter.rate == 10.0

    def test_high_rate_no_blocking(self):
        """Very high rate should never block for single tokens."""
        limiter = RateLimiter(rate=1_000_000.0, burst=1_000_000)
        for _ in range(100):
            assert limiter.acquire(block=False) is True

    def test_thread_safety(self):
        """Multiple threads should be able to acquire concurrently."""
        import threading

        limiter = RateLimiter(rate=1000.0, burst=100)
        errors = []

        def worker():
            for _ in range(20):
                try:
                    limiter.acquire(block=True)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestExponentialBackoff:
    """Test exponential backoff calculation."""

    def test_backoff_increases(self):
        """Each attempt should have a longer delay."""
        delays = [exponential_backoff(i, base_delay=1.0, jitter=False) for i in range(5)]
        for i in range(1, len(delays)):
            assert delays[i] >= delays[i - 1]

    def test_backoff_values(self):
        """Should calculate correct exponential values (no jitter)."""
        assert exponential_backoff(0, base_delay=1.0, jitter=False) == 1.0
        assert exponential_backoff(1, base_delay=1.0, jitter=False) == 2.0
        assert exponential_backoff(2, base_delay=1.0, jitter=False) == 4.0
        assert exponential_backoff(3, base_delay=1.0, jitter=False) == 8.0

    def test_backoff_max_delay(self):
        """Should cap at max_delay."""
        delay = exponential_backoff(10, base_delay=1.0, max_delay=30.0, jitter=False)
        assert delay == 30.0

    def test_backoff_jitter(self):
        """Jitter should vary the delay."""
        delays = [exponential_backoff(2, base_delay=1.0, max_delay=10.0, jitter=True) for _ in range(20)]
        # With jitter, delays should vary (not all identical)
        unique_delays = set(round(d, 2) for d in delays)
        assert len(unique_delays) > 1

    def test_custom_base_delay(self):
        """Should support custom base delay."""
        assert exponential_backoff(0, base_delay=2.0, jitter=False) == 2.0
        assert exponential_backoff(1, base_delay=3.0, jitter=False) == 6.0


class TestRetryWithBackoff:
    """Test retry with exponential backoff."""

    def test_success_no_retry(self):
        """Should succeed on first attempt without retry."""
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            return "success"

        result = retry_with_backoff(fn, max_retries=3)
        assert result == "success"
        assert call_count == 1

    def test_retry_on_failure(self):
        """Should retry on retryable exceptions."""
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("retryable")
            return "success"

        result = retry_with_backoff(fn, max_retries=3, base_delay=0.01)
        assert result == "success"
        assert call_count == 3

    def test_exhaust_retries(self):
        """Should raise after exhausting retries."""
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("always fails")

        with pytest.raises(ConnectionError):
            retry_with_backoff(fn, max_retries=2, base_delay=0.01)
        assert call_count == 3  # initial + 2 retries

    def test_non_retryable_exception(self):
        """Should not retry on non-retryable exceptions."""
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            retry_with_backoff(fn, max_retries=3, base_delay=0.01)
        assert call_count == 1  # no retry

    def test_zero_retries(self):
        """max_retries=0 should not retry."""
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("fail")

        with pytest.raises(ConnectionError):
            retry_with_backoff(fn, max_retries=0, base_delay=0.01)
        assert call_count == 1

    def test_on_retry_callback(self):
        """Should call on_retry callback before each retry."""
        call_count = 0
        retry_attempts = []

        def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("fail")
            return "ok"

        def on_retry(attempt, exc, delay):
            retry_attempts.append((attempt, str(exc), delay))

        result = retry_with_backoff(
            fn, max_retries=3, base_delay=0.01,
            on_retry=on_retry,
        )
        assert result == "ok"
        assert len(retry_attempts) == 2
        assert retry_attempts[0][0] == 1
        assert retry_attempts[1][0] == 2

    def test_kwargs_passthrough(self):
        """Should pass kwargs through to the function."""
        def fn(greeting, name):
            return f"{greeting}, {name}!"

        result = retry_with_backoff(fn, greeting="Hello", name="World")
        assert result == "Hello, World!"


class TestTimeout:
    """Test the timeout context manager."""

    def test_no_timeout(self):
        """Should not raise if operation completes in time."""
        with timeout(seconds=5.0):
            result = "done"
        assert result == "done"

    def test_timeout_raises(self):
        """Should raise TimeoutError if operation takes too long."""
        import threading

        # Use an event to simulate a long operation
        slow_op = threading.Event()

        with pytest.raises(TimeoutError, match="Operation timed out"):
            with timeout(seconds=0.05):
                slow_op.wait(1.0)  # This will be interrupted

    def test_custom_message(self):
        """Should use custom timeout message."""
        with pytest.raises(TimeoutError, match="Custom timeout message"):
            with timeout(seconds=0.01, message="Custom timeout message"):
                import threading
                threading.Event().wait(1.0)

    def test_nested_timeout(self):
        """Nested timeouts should work."""
        with timeout(seconds=5.0):
            with timeout(seconds=5.0):
                result = "nested ok"
        assert result == "nested ok"


class TestResilientClient:
    """Test the combined resilient client."""

    def test_execute_success(self):
        """Should execute successfully."""
        client = ResilientClient(rate=1000.0, max_retries=0)

        def fn(greeting):
            return f"{greeting}, World!"

        result = client.execute(fn, greeting="Hello")
        assert result == "Hello, World!"

    def test_execute_with_retry(self):
        """Should retry on failure."""
        client = ResilientClient(rate=1000.0, max_retries=3, base_delay=0.01)
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("retry")
            return "ok"

        result = client.execute(fn)
        assert result == "ok"
        assert call_count == 3

    def test_execute_rate_limited(self):
        """Should respect rate limits."""
        client = ResilientClient(rate=100.0, burst=2, max_retries=0)

        def fn():
            return "ok"

        # First two should pass immediately (burst)
        assert client.execute(fn) == "ok"
        assert client.execute(fn) == "ok"

        # Third should still work (blocking acquire)
        start = time.time()
        assert client.execute(fn) == "ok"
        elapsed = time.time() - start
        assert elapsed < 0.1  # Fast because rate is high

    def test_execute_timeout(self):
        """Should timeout on slow operations."""
        client = ResilientClient(
            rate=1000.0, max_retries=0, default_timeout=0.05,
        )

        def slow_fn():
            import threading
            threading.Event().wait(1.0)
            return "too late"

        with pytest.raises((TimeoutError, OSError)):
            client.execute(slow_fn)

    def test_rate_limiter_property(self):
        """Should expose the rate limiter."""
        client = ResilientClient(rate=5.0, burst=10)
        assert client.rate_limiter.rate == 5.0