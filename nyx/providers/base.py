"""Abstract base class for all LLM providers."""
from __future__ import annotations

import time
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


class BaseLLMProvider(ABC):
    """Abstract LLM provider."""

    def __init__(self, config) -> None:
        self.config = config
        # Build a resilient client from config if rate limiting is enabled
        if getattr(config, "rate_limiting_enabled", True):
            self._resilient_client = ResilientClient(
                rate=getattr(config, "rate_limiting_rate", 10.0),
                burst=getattr(config, "rate_limiting_burst", 20),
                max_retries=getattr(config, "rate_limiting_max_retries", 3),
                base_delay=getattr(config, "rate_limiting_base_delay", 1.0),
                max_delay=getattr(config, "rate_limiting_max_delay", 60.0),
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
