"""Anthropic Claude provider."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable

from .base import BaseLLMProvider, LLMResponse, ToolCall, ToolDefinition


class AnthropicProvider(BaseLLMProvider):
    """Provider for Anthropic's Claude API."""

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        stream: bool = False,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        url = self.config.get_base_url() or "https://api.anthropic.com/v1/messages"
        headers = self._build_headers()
        timeout = self.config.request_timeout

        # Convert messages to Anthropic format
        system = ""
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system += (msg["content"] + "\n")
            else:
                role = "assistant" if msg["role"] == "assistant" else "user"
                content = []
                if isinstance(msg.get("content"), str):
                    content.append({"type": "text", "text": msg["content"]})
                elif isinstance(msg.get("content"), list):
                    content = msg["content"]
                anthropic_messages.append({"role": role, "content": content})

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": anthropic_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": stream,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [self._to_anthropic_tool(t) for t in tools]

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")

        for attempt in range(2):
            try:
                if stream:
                    return self._stream_response(request, timeout, on_token)
                return self._non_stream_response(request, timeout)
            except urllib.error.HTTPError as err:
                error_body = err.read().decode("utf-8", errors="replace")
                if err.code == 429 and attempt == 0:
                    print(f"\n[Rate limit] retrying...")
                    continue
                raise RuntimeError(error_body.strip() or f"HTTP {err.code}: {err.reason}") from err
            except urllib.error.URLError as err:
                raise RuntimeError(f"Network error: {err}") from err
        raise RuntimeError("Still rate limited after retry.")

    def _build_headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.config.get_api_key(),
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _non_stream_response(self, request: urllib.request.Request, timeout: int) -> LLMResponse:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        if body.get("type") == "error":
            raise RuntimeError(body["error"]["message"])

        content_blocks = body.get("content", [])
        text = ""
        tool_calls = []
        for block in content_blocks:
            if block["type"] == "text":
                text += block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append(ToolCall(
                    id=block["id"],
                    name=block["name"],
                    arguments=block.get("input", {}),
                ))

        return LLMResponse(
            content=text,
            tool_calls=tool_calls,
            finish_reason=body.get("stop_reason", "end_turn"),
            usage=body.get("usage", {}),
            raw=body,
        )

    def _stream_response(
        self,
        request: urllib.request.Request,
        timeout: int,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        full_content: list[str] = []
        tool_calls_buffer: dict[str, ToolCall] = {}
        finish_reason = "end_turn"

        with urllib.request.urlopen(request, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                e_type = event.get("type", "")
                if e_type == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        full_content.append(text)
                        if on_token:
                            on_token(text)
                    elif delta.get("type") == "input_json_delta":
                        idx = str(event.get("index", 0))
                        partial = delta.get("partial_json", "")
                        if idx not in tool_calls_buffer:
                            tc = event.get("content_block", {}).get("tool_use", {})
                            tool_calls_buffer[idx] = ToolCall(
                                id=tc.get("id", idx),
                                name=tc.get("name", ""),
                                arguments={"__partial": ""},
                            )
                        tool_calls_buffer[idx].arguments["__partial"] += partial
                elif e_type == "content_block_start":
                    cb = event.get("content_block", {})
                    if cb.get("type") == "tool_use":
                        idx = str(event.get("index", 0))
                        tool_calls_buffer[idx] = ToolCall(
                            id=cb["id"],
                            name=cb["name"],
                            arguments={"__partial": ""},
                        )
                elif e_type == "message_delta":
                    finish_reason = event.get("delta", {}).get("stop_reason", "end_turn")
                elif e_type == "error":
                    raise RuntimeError(event.get("error", {}).get("message", "Unknown error"))

        # Finalise tool calls
        final_tool_calls = []
        for tc in tool_calls_buffer.values():
            if isinstance(tc.arguments, dict) and "__partial" in tc.arguments:
                try:
                    tc.arguments = json.loads(tc.arguments["__partial"])
                except (json.JSONDecodeError, TypeError):
                    tc.arguments = {}
            final_tool_calls.append(tc)

        return LLMResponse(
            content="".join(full_content),
            tool_calls=final_tool_calls,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _to_anthropic_tool(t: ToolDefinition) -> dict:
        return {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
