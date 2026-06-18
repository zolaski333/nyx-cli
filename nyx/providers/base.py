"""Abstract base class for all LLM providers."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from email.message import Message
from typing import Any, Callable, Iterator, cast

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


def _compact_error_text(value: Any, *, limit: int = 500) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "..."
    return text


def _extract_error_message(parsed: Any) -> tuple[str, str]:
    if isinstance(parsed, dict):
        error = parsed.get("error", parsed)
        if isinstance(error, dict):
            message = (
                error.get("message")
                or error.get("detail")
                or error.get("error")
                or error.get("description")
            )
            code = error.get("code") or error.get("type") or parsed.get("type") or ""
            if message:
                return str(message), str(code or "")
            return json.dumps(error, ensure_ascii=False, separators=(",", ":")), str(code or "")
        if isinstance(error, str):
            return error, str(parsed.get("code") or parsed.get("type") or "")
        message = parsed.get("message") or parsed.get("detail")
        if message:
            return str(message), str(parsed.get("code") or parsed.get("type") or "")
    return "", ""


def format_api_error(
    provider: str,
    *,
    status: int | None = None,
    reason: str = "",
    body: Any = None,
) -> str:
    """Return a concise, user-facing provider error."""
    parsed: Any = None
    raw_text = ""
    if isinstance(body, (dict, list)):
        parsed = body
    elif body is not None:
        raw_text = str(body).strip()
        if raw_text:
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError:
                parsed = None

    message, code = _extract_error_message(parsed)
    if not message:
        message = raw_text or reason or "Request failed."

    prefix = f"{provider} API error"
    if status is not None:
        prefix += f" {status}"
    if code:
        prefix += f" ({_compact_error_text(code, limit=80)})"

    hint = ""
    if status in {401, 403}:
        hint = "Check the configured API key and provider access."
    elif status == 429:
        hint = "Rate limit reached; retry shortly."
    elif status is not None and status >= 500:
        hint = "Provider service error; retry later."

    compact_message = _compact_error_text(message)
    if hint:
        separator = " " if compact_message.endswith((".", "!", "?")) else ". "
        compact_message = f"{compact_message}{separator}{hint}"
    return f"{prefix}: {compact_message}"


class HttpxResponseWrapper:
    """Wrapper to make httpx.Response compatible with urllib response."""

    def __init__(self, response: Any) -> None:
        self.response = response

    def read(self) -> bytes:
        return bytes(self.response.content)

    def __iter__(self) -> Iterator[bytes]:
        for line in self.response.iter_lines():
            yield (line + "\n").encode("utf-8")

    def __enter__(self) -> "HttpxResponseWrapper":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def close(self) -> None:
        self.response.close()


class Urllib3ResponseWrapper:
    """Wrapper to make urllib3 response compatible with urllib response."""

    def __init__(self, response: Any) -> None:
        self.response = response

    def read(self) -> bytes:
        if hasattr(self.response, "data") and self.response.data:
            return bytes(self.response.data)
        return bytes(self.response.read())

    def __iter__(self) -> Iterator[bytes]:
        buffer = b""
        for chunk in self.response.stream(amt=1024):
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                yield line + b"\n"
        if buffer:
            yield buffer

    def __enter__(self) -> "Urllib3ResponseWrapper":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def close(self) -> None:
        self.response.release_conn()


class BaseLLMProvider(ABC):
    """Abstract LLM provider."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self._httpx_client: Any | None = None
        self._urllib3_pool: Any | None = None
        self._resilient_client: ResilientClient | None
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

    def _resilient_urlopen(self, request: Any, timeout: int) -> Any:
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

                client = self._httpx_client
                assert client is not None

                def _httpx_call() -> HttpxResponseWrapper:
                    if method == "POST":
                        resp = client.request(method, url, content=data, headers=headers)
                    else:
                        resp = client.request(method, url, headers=headers)
                    
                    wrapper = HttpxResponseWrapper(resp)
                    if resp.status_code >= 400:
                        raise urllib.error.HTTPError(
                            url,
                            resp.status_code,
                            resp.reason_phrase,
                            cast(Message[str, str], headers),
                            cast(Any, wrapper),
                        )
                    return wrapper

                if self._resilient_client:
                    return self._resilient_client.execute(_httpx_call)
                return _httpx_call()
            except ImportError:
                pass

            try:
                import urllib3
                if self._urllib3_pool is None:
                    self._urllib3_pool = urllib3.PoolManager(timeout=timeout)

                pool = self._urllib3_pool
                assert pool is not None

                def _urllib3_call() -> Urllib3ResponseWrapper:
                    resp = pool.request(method, url, body=data, headers=headers, preload_content=False)
                    wrapper = Urllib3ResponseWrapper(resp)
                    if resp.status >= 400:
                        raise urllib.error.HTTPError(
                            url,
                            resp.status,
                            str(resp.reason or ""),
                            cast(Message[str, str], headers),
                            cast(Any, wrapper),
                        )
                    return wrapper

                if self._resilient_client:
                    return self._resilient_client.execute(_urllib3_call)
                return _urllib3_call()
            except ImportError:
                pass

        if self._resilient_client:
            return self._resilient_client.execute(
                urllib.request.urlopen,
                url=request,
                timeout=timeout,
            )
        return urllib.request.urlopen(request, timeout=timeout)

    @property
    def resilient_client(self) -> ResilientClient | None:
        return self._resilient_client
