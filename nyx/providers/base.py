"""Abstract base class for all LLM providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from nyx.rate_limiter import ResilientClient


@dataclass
class ToolDefinition:
    """Describes a tool/function the LLM can call."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


@dataclass
class ToolCall:
    """A tool invocation requested by the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Standardised response from any LLM provider."""
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


class HttpxResponseWrapper:
    """Wrapper to make httpx.Response compatible with urllib response."""

    def __init__(self, response) -> None:
        self.response = response

    def read(self) -> bytes:
        return self.response.content

    def __iter__(self):
        for line in self.response.iter_lines():
            yield (line + "\n").encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        self.response.close()


class Urllib3ResponseWrapper:
    """Wrapper to make urllib3 response compatible with urllib response."""

    def __init__(self, response) -> None:
        self.response = response

    def read(self) -> bytes:
        if hasattr(self.response, "data") and self.response.data:
            return self.response.data
        return self.response.read()

    def __iter__(self):
        buffer = b""
        for chunk in self.response.stream(amt=1024):
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                yield line + b"\n"
        if buffer:
            yield buffer

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        self.response.release_conn()


class BaseLLMProvider(ABC):
    """Abstract LLM provider."""

    def __init__(self, config) -> None:
        self.config = config
        self._httpx_client = None
        self._urllib3_pool = None
        # Build a resilient client from config if rate limiting is enabled
        # NOTE: Defaults are tuned for fast models (DeepSeek V4 Flash).
        # Rate limiting is intentionally permissive — the local token bucket
        # should not be a bottleneck. Real rate limits are enforced server-side.
        if getattr(config, "rate_limiting_enabled", True):
            self._resilient_client = ResilientClient(
                rate=getattr(config, "rate_limiting_rate", 100.0),       # was 10.0
                burst=getattr(config, "rate_limiting_burst", 50),        # was 20
                max_retries=getattr(config, "rate_limiting_max_retries", 1),  # was 3
                base_delay=getattr(config, "rate_limiting_base_delay", 0.5),  # was 1.0
                max_delay=getattr(config, "rate_limiting_max_delay", 10.0),   # was 60.0
                default_timeout=getattr(config, "request_timeout", 120),
            )
        else:
            self._resilient_client = None

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        stream: bool = False,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Send a chat completion request.

        If *stream* is True and *on_token* is provided, yield tokens
        as they arrive. The returned content will be the full text.
        """
        ...

    def _resilient_urlopen(self, request, timeout: int):
        """Open a URL with rate limiting, retry, and timeout.

        Uses the ResilientClient if configured, otherwise falls back
        to a plain urllib.request.urlopen.
        """
        import urllib.request
        import urllib.error

        # Detect if urlopen is mocked (to prevent bypassing mocks in unit tests)
        is_mocked = "mock" in type(urllib.request.urlopen).__name__.lower()

        # Extract request details from urllib.request.Request object
        url = request.full_url
        data = request.data
        headers = {k: v for k, v in request.headers.items()}
        method = request.method or ("POST" if data is not None else "GET")
        is_localhost = url.startswith(("http://127.0.0.1", "http://localhost", "http://[::1]"))

        # Try to use optional httpx or urllib3 client if not running under a unit test mock
        if not is_mocked and not is_localhost:
            try:
                import httpx
                if self._httpx_client is None:
                    self._httpx_client = httpx.Client(http2=True, timeout=timeout)

                def _httpx_call():
                    if method == "POST":
                        resp = self._httpx_client.request(method, url, content=data, headers=headers)
                    else:
                        resp = self._httpx_client.request(method, url, headers=headers)
                    
                    wrapper = HttpxResponseWrapper(resp)
                    if resp.status_code >= 400:
                        raise urllib.error.HTTPError(url, resp.status_code, resp.reason_phrase, headers, wrapper)
                    return wrapper

                if self._resilient_client:
                    return self._resilient_client.execute(_httpx_call, timeout_seconds=timeout)
                return _httpx_call()
            except ImportError:
                pass

            try:
                import urllib3
                if self._urllib3_pool is None:
                    self._urllib3_pool = urllib3.PoolManager(timeout=timeout)

                def _urllib3_call():
                    resp = self._urllib3_pool.request(method, url, body=data, headers=headers, preload_content=False)
                    wrapper = Urllib3ResponseWrapper(resp)
                    if resp.status >= 400:
                        raise urllib.error.HTTPError(url, resp.status, resp.reason, headers, wrapper)
                    return wrapper

                if self._resilient_client:
                    return self._resilient_client.execute(_urllib3_call, timeout_seconds=timeout)
                return _urllib3_call()
            except ImportError:
                pass

        if self._resilient_client:
            return self._resilient_client.execute(
                urllib.request.urlopen,
                timeout_seconds=timeout,
                url=request,
                timeout=timeout,
            )
        return urllib.request.urlopen(request, timeout=timeout)

    @property
    def resilient_client(self) -> ResilientClient | None:
        return self._resilient_client
