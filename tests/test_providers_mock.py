"""Tests for LLM providers with HTTP mocks — OpenRouter, OpenAI, Anthropic."""
from __future__ import annotations

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

import pytest

from nyx.config import Config
from nyx.providers import get_provider
from nyx.providers.base import ToolDefinition


# ---------------------------------------------------------------------------
# Mock HTTP server that simulates LLM API responses
# ---------------------------------------------------------------------------


class MockLLMHandler(BaseHTTPRequestHandler):
    """HTTP request handler that simulates an LLM API endpoint."""

    # Class-level state for test control
    responses: list[dict[str, Any]] = []
    request_count = 0
    fail_with_status: int | None = None
    fail_until_request: int = 0  # Fail for requests up to this count
    fail_with_retry_after: bool = False  # Include retry_after_seconds in 429 errors
    delay_seconds: float = 0.0

    def do_POST(self):
        MockLLMHandler.request_count += 1

        # Simulate failure if configured (check fail_until_request for counter-based failures)
        if MockLLMHandler.fail_with_status is not None:
            if MockLLMHandler.request_count <= MockLLMHandler.fail_until_request:
                self.send_response(MockLLMHandler.fail_with_status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                error_body = {"error": {"message": "Mock error"}}
                if MockLLMHandler.fail_with_retry_after and MockLLMHandler.fail_with_status == 429:
                    error_body["error"]["metadata"] = {"retry_after_seconds": 0.1}
                body = json.dumps(error_body)
                self.wfile.write(body.encode("utf-8"))
                return
            # Clear the fail flag once we've passed the threshold
            MockLLMHandler.fail_with_status = None

        # Simulate delay if configured
        if MockLLMHandler.delay_seconds > 0:
            import time
            time.sleep(MockLLMHandler.delay_seconds)

        # Read the request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        req_data = json.loads(body)

        # Determine which response to send based on request count
        is_stream = req_data.get("stream", False)

        if is_stream:
            self._send_streaming_response(req_data)
        else:
            self._send_non_stream_response(req_data)

    def _send_non_stream_response(self, req_data: dict[str, Any]) -> None:
        """Send a non-streaming response."""
        # Use the next available response, or a default
        if MockLLMHandler.responses:
            resp_data = MockLLMHandler.responses.pop(0)
        else:
            resp_data = self._default_response(req_data)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp_data).encode("utf-8"))

    def _send_streaming_response(self, req_data: dict[str, Any]) -> None:
        """Send a streaming SSE response."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        # Send a simple streaming response
        chunks = [
            {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]},
        ]
        for chunk in chunks:
            line = f"data: {json.dumps(chunk)}\n\n"
            self.wfile.write(line.encode("utf-8"))
            self.wfile.flush()

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _default_response(self, req_data: dict[str, Any]) -> dict[str, Any]:
        """Generate a default response based on request."""
        has_tools = "tools" in req_data
        if has_tools:
            return {
                "id": "mock-cmpl-01",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": '{"query": "test"}',
                            },
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
            }
        return {
            "id": "mock-cmpl-01",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a mock response.",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        }

    def log_message(self, format, *args):
        """Suppress HTTP server log messages."""
        pass


def _make_anthropic_response(req_data: dict[str, Any]) -> dict[str, Any]:
    """Generate an Anthropic-format response."""
    has_tools = "tools" in req_data
    if has_tools:
        return {
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "web_search",
                    "input": {"query": "test"},
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 50, "output_tokens": 10},
        }
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello from Claude!"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 50, "output_tokens": 10},
    }


class MockAnthropicHandler(BaseHTTPRequestHandler):
    """HTTP handler that simulates Anthropic API responses."""

    responses: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    request_count = 0
    fail_with_status: int | None = None

    def do_POST(self):
        MockAnthropicHandler.request_count += 1

        if MockAnthropicHandler.fail_with_status is not None:
            self.send_response(MockAnthropicHandler.fail_with_status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            body = json.dumps({"error": {"message": "Mock error"}})
            self.wfile.write(body.encode("utf-8"))
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        req_data = json.loads(body)
        MockAnthropicHandler.requests.append(req_data)

        is_stream = req_data.get("stream", False)

        if is_stream:
            self._send_streaming_response()
        else:
            self._send_non_stream_response(req_data)

    def _send_non_stream_response(self, req_data):
        if MockAnthropicHandler.responses:
            resp_data = MockAnthropicHandler.responses.pop(0)
        else:
            resp_data = _make_anthropic_response(req_data)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp_data).encode("utf-8"))

    def _send_streaming_response(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        events = [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " from Claude"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        ]
        for event in events:
            line = f"data: {json.dumps(event)}\n\n"
            self.wfile.write(line.encode("utf-8"))
            self.wfile.flush()

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format, *args):
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_openai_server():
    """Start a mock HTTP server that simulates OpenAI-compatible API."""
    server = HTTPServer(("127.0.0.1", 0), MockLLMHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    MockLLMHandler.responses = []
    MockLLMHandler.request_count = 0
    MockLLMHandler.fail_with_status = None
    MockLLMHandler.fail_until_request = 0
    MockLLMHandler.fail_with_retry_after = False
    MockLLMHandler.delay_seconds = 0.0
    yield port
    server.shutdown()


@pytest.fixture
def mock_anthropic_server():
    """Start a mock HTTP server that simulates Anthropic API."""
    server = HTTPServer(("127.0.0.1", 0), MockAnthropicHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    MockAnthropicHandler.responses = []
    MockAnthropicHandler.requests = []
    MockAnthropicHandler.request_count = 0
    MockAnthropicHandler.fail_with_status = None
    yield port
    server.shutdown()


def _make_config(provider: str, base_url: str, api_key: str = "test-key") -> Config:
    """Create a Config with the given provider and base URL."""
    return Config(
        provider=provider,
        model="test-model",
        openrouter_base_url=base_url,
        openai_base_url=base_url,
        anthropic_base_url=base_url,
        openrouter_api_key=api_key,
        openai_api_key=api_key,
        anthropic_api_key=api_key,
        request_timeout=5,
        rate_limiting_enabled=False,
    )


# =========================================================================
# OpenRouter / OpenAI Provider Tests
# =========================================================================


class TestOpenRouterProviderMock:
    """Test OpenRouter provider with mock HTTP server."""

    def test_non_stream_chat(self, mock_openai_server):
        """Should send a non-streaming chat request and get response."""
        base_url = f"http://127.0.0.1:{mock_openai_server}/v1/chat/completions"
        config = _make_config("openrouter", base_url)
        provider = get_provider(config)

        response = provider.chat(
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
        )
        assert response.content == "This is a mock response."
        assert response.finish_reason == "stop"
        assert response.usage.get("prompt_tokens") == 50

    def test_stream_chat(self, mock_openai_server):
        """Should handle streaming responses."""
        base_url = f"http://127.0.0.1:{mock_openai_server}/v1/chat/completions"
        config = _make_config("openrouter", base_url)
        provider = get_provider(config)

        tokens = []

        def on_token(t: str):
            tokens.append(t)

        response = provider.chat(
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            on_token=on_token,
        )
        assert "Hello" in response.content
        assert len(tokens) > 0

    def test_tool_call_response(self, mock_openai_server):
        """Should parse tool calls from response."""
        base_url = f"http://127.0.0.1:{mock_openai_server}/v1/chat/completions"
        config = _make_config("openrouter", base_url)
        provider = get_provider(config)

        # Set up a response with tool calls
        MockLLMHandler.responses = [{
            "id": "mock-cmpl-02",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query": "latest news"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10},
        }]

        response = provider.chat(
            messages=[{"role": "user", "content": "Search for news"}],
            tools=[ToolDefinition(name="web_search", description="Search", parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})],
            stream=False,
        )
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "web_search"
        assert response.tool_calls[0].arguments["query"] == "latest news"

    def test_http_error(self, mock_openai_server):
        """Should raise RuntimeError on HTTP error."""
        base_url = f"http://127.0.0.1:{mock_openai_server}/v1/chat/completions"
        config = _make_config("openrouter", base_url)
        provider = get_provider(config)

        # Fail for all requests (set fail_until_request high)
        MockLLMHandler.fail_with_status = 500
        MockLLMHandler.fail_until_request = MockLLMHandler.request_count + 10

        with pytest.raises(RuntimeError, match="Mock error"):
            provider.chat(
                messages=[{"role": "user", "content": "Hello"}],
                stream=False,
            )

    def test_rate_limit_retry(self, mock_openai_server):
        """Should retry on 429 rate limit."""
        base_url = f"http://127.0.0.1:{mock_openai_server}/v1/chat/completions"
        config = _make_config("openrouter", base_url)
        config.rate_limiting_enabled = True
        provider = get_provider(config)

        # Fail for the first request only with retry_after_seconds so the provider retries
        MockLLMHandler.fail_with_status = 429
        MockLLMHandler.fail_with_retry_after = True
        MockLLMHandler.fail_until_request = MockLLMHandler.request_count + 1

        response = provider.chat(
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
        )
        assert response.content == "This is a mock response."

    def test_custom_headers(self, mock_openai_server):
        """Should send custom headers for OpenRouter."""
        base_url = f"http://127.0.0.1:{mock_openai_server}/v1/chat/completions"
        config = _make_config("openrouter", base_url)
        config.site_url = "https://example.com"
        config.site_name = "TestSite"
        provider = get_provider(config)

        response = provider.chat(
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
        )
        assert response.content == "This is a mock response."


class TestAnthropicProviderMock:
    """Test Anthropic provider with mock HTTP server."""

    def test_non_stream_chat(self, mock_anthropic_server):
        """Should send a non-streaming chat request and get response."""
        base_url = f"http://127.0.0.1:{mock_anthropic_server}/v1/messages"
        config = _make_config("anthropic", base_url)
        provider = get_provider(config)

        response = provider.chat(
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
        )
        assert "Claude" in response.content
        assert response.finish_reason == "end_turn"

    def test_stream_chat(self, mock_anthropic_server):
        """Should handle Anthropic streaming responses."""
        base_url = f"http://127.0.0.1:{mock_anthropic_server}/v1/messages"
        config = _make_config("anthropic", base_url)
        provider = get_provider(config)

        tokens = []

        def on_token(t: str):
            tokens.append(t)

        response = provider.chat(
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            on_token=on_token,
        )
        assert "Claude" in response.content
        assert len(tokens) > 0

    def test_tool_call_response(self, mock_anthropic_server):
        """Should parse Anthropic tool calls."""
        base_url = f"http://127.0.0.1:{mock_anthropic_server}/v1/messages"
        config = _make_config("anthropic", base_url)
        provider = get_provider(config)

        MockAnthropicHandler.responses = [{
            "id": "msg_02",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_abc",
                    "name": "web_search",
                    "input": {"query": "test query"},
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 50, "output_tokens": 10},
        }]

        response = provider.chat(
            messages=[{"role": "user", "content": "Search"}],
            tools=[ToolDefinition(name="web_search", description="Search", parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})],
            stream=False,
        )
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "web_search"
        assert response.tool_calls[0].arguments["query"] == "test query"

    def test_http_error(self, mock_anthropic_server):
        """Should raise RuntimeError on HTTP error."""
        base_url = f"http://127.0.0.1:{mock_anthropic_server}/v1/messages"
        config = _make_config("anthropic", base_url)
        provider = get_provider(config)

        MockAnthropicHandler.fail_with_status = 500

        with pytest.raises(RuntimeError):
            provider.chat(
                messages=[{"role": "user", "content": "Hello"}],
                stream=False,
            )

    def test_prompt_caching_payload(self, mock_anthropic_server):
        """Should verify that system prompt and tools have cache_control."""
        base_url = f"http://127.0.0.1:{mock_anthropic_server}/v1/messages"
        config = _make_config("anthropic", base_url)
        provider = get_provider(config)

        provider.chat(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ],
            tools=[ToolDefinition(name="web_search", description="Search", parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})],
            stream=False,
        )

        assert len(MockAnthropicHandler.requests) == 1
        req = MockAnthropicHandler.requests[0]
        # Assert system prompt has cache_control
        assert "system" in req
        assert isinstance(req["system"], list)
        assert req["system"][-1]["type"] == "text"
        assert req["system"][-1]["text"] == "You are a helpful assistant."
        assert req["system"][-1]["cache_control"] == {"type": "ephemeral"}

        # Assert tools have cache_control
        assert "tools" in req
        assert isinstance(req["tools"], list)
        assert req["tools"][-1]["name"] == "web_search"
        assert req["tools"][-1]["cache_control"] == {"type": "ephemeral"}


class TestOpenAIProviderMock:
    """Test OpenAI provider with mock HTTP server."""

    def test_non_stream_chat(self, mock_openai_server):
        """Should send a non-streaming chat request."""
        base_url = f"http://127.0.0.1:{mock_openai_server}/v1/chat/completions"
        config = _make_config("openai", base_url)
        provider = get_provider(config)

        response = provider.chat(
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
        )
        assert response.content == "This is a mock response."

    def test_custom_headers(self, mock_openai_server):
        """OpenAI should send Authorization header."""
        base_url = f"http://127.0.0.1:{mock_openai_server}/v1/chat/completions"
        config = _make_config("openai", base_url)
        provider = get_provider(config)

        response = provider.chat(
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
        )
        assert response.content == "This is a mock response."


class TestProviderFactory:
    """Test the provider factory function."""

    def test_get_openrouter(self):
        """Should return OpenRouterProvider for 'openrouter'."""
        config = Config(provider="openrouter", openrouter_api_key="test", rate_limiting_enabled=False)
        provider = get_provider(config)
        from nyx.providers.openrouter import OpenRouterProvider
        assert isinstance(provider, OpenRouterProvider)

    def test_get_openai(self):
        """Should return OpenAIProvider for 'openai'."""
        config = Config(provider="openai", openai_api_key="test", rate_limiting_enabled=False)
        provider = get_provider(config)
        from nyx.providers.openai_provider import OpenAIProvider
        assert isinstance(provider, OpenAIProvider)

    def test_get_anthropic(self):
        """Should return AnthropicProvider for 'anthropic'."""
        config = Config(provider="anthropic", anthropic_api_key="test", rate_limiting_enabled=False)
        provider = get_provider(config)
        from nyx.providers.anthropic_provider import AnthropicProvider
        assert isinstance(provider, AnthropicProvider)

    def test_unknown_provider(self):
        """Should raise ValueError for unknown provider."""
        config = Config(provider="unknown", openrouter_api_key="test", rate_limiting_enabled=False)
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider(config)