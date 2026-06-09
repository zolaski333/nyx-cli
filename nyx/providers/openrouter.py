"""OpenRouter provider — compatible with many models via a single API."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from .base import BaseLLMProvider, LLMResponse, ToolCall, ToolDefinition


class OpenRouterProvider(BaseLLMProvider):
    """Provider for OpenRouter (and any OpenAI-compatible endpoint)."""

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        stream: bool = False,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        url = self.config.get_base_url() or "https://openrouter.ai/api/v1/chat/completions"
        headers = self._build_headers()
        timeout = self.config.request_timeout

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": stream,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        if tools:
            payload["tools"] = [self._to_openai_tool(t) for t in tools]
            payload["tool_choice"] = "auto"

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
                    retry_after = self._extract_retry_after(error_body)
                    if retry_after:
                        print(f"\n[Rate limit] retry in {retry_after}s...")
                        time.sleep(retry_after)
                        continue
                raise RuntimeError(error_body.strip() or f"HTTP {err.code}: {err.reason}") from err
            except urllib.error.URLError as err:
                raise RuntimeError(f"Network error: {err}") from err
        raise RuntimeError("Still rate limited after retry.")

    def _build_headers(self) -> dict[str, str]:
        api_key = self.config.get_api_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.config.site_url:
            headers["HTTP-Referer"] = self.config.site_url
        if self.config.site_name:
            headers["X-Title"] = self.config.site_name
        return headers

    def _non_stream_response(self, request: urllib.request.Request, timeout: int) -> LLMResponse:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        if "error" in body:
            raise RuntimeError(json.dumps(body["error"], ensure_ascii=False, indent=2))
        choice = body["choices"][0]
        msg = choice["message"]
        tool_calls = []
        for tc in msg.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                args = {}
            tool_calls.append(ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=args))
        return LLMResponse(
            content=msg.get("content") or "",
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
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
        tool_calls: dict[str, ToolCall] = {}
        finish_reason = "stop"

        with urllib.request.urlopen(request, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [{}])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                finish_reason = choices[0].get("finish_reason") or finish_reason

                # Content
                content = delta.get("content", "")
                if content:
                    full_content.append(content)
                    if on_token:
                        on_token(content)

                # Tool calls
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    t_id = tc.get("id", "")
                    t_name = tc.get("function", {}).get("name", "")
                    t_args = tc.get("function", {}).get("arguments", "")
                    if idx not in tool_calls:
                        tool_calls[idx] = ToolCall(id=t_id, name=t_name, arguments={})
                    if t_id:
                        tool_calls[idx].id = t_id
                    if t_name:
                        tool_calls[idx].name = t_name
                    if t_args:
                        existing = tool_calls[idx].arguments
                        if isinstance(existing, dict):
                            pass
                        else:
                            combined = (tool_calls[idx].arguments.get("_buffer", "") if isinstance(tool_calls[idx].arguments, dict) else "") + t_args
                            tool_calls[idx].arguments = {"_buffer": combined}

        # Parse buffered arguments
        final_tool_calls = []
        for tc in tool_calls.values():
            raw_args = tc.arguments
            if isinstance(raw_args, dict) and "_buffer" in raw_args:
                try:
                    tc.arguments = json.loads(raw_args["_buffer"])
                except (json.JSONDecodeError, TypeError):
                    tc.arguments = {}
            final_tool_calls.append(tc)

        if on_token and full_content:
            pass  # tokens already streamed

        return LLMResponse(
            content="".join(full_content),
            tool_calls=final_tool_calls,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _to_openai_tool(t: ToolDefinition) -> dict:
        return {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }

    @staticmethod
    def _extract_retry_after(error_body: str) -> int:
        try:
            error = json.loads(error_body)
            ra = error.get("error", {}).get("metadata", {}).get("retry_after_seconds")
            if isinstance(ra, (int, float)):
                return max(1, int(ra))
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        return 0
