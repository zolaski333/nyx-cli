"""
Subagent system — spawns child agents that report back.

A subagent is a self-contained LLM conversation with its own system prompt.
It can be given a task, run independently, and return results.
Useful for parallel research, code generation, or complex subtasks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from nyx.config import Config
from nyx.providers.base import BaseLLMProvider
from nyx.providers import get_provider


@dataclass
class SubagentResult:
    """Result from a subagent execution."""
    task: str
    output: str
    tokens_used: int = 0
    error: str | None = None


class Subagent:
    """A subagent that can execute a task independently."""

    def __init__(
        self,
        name: str,
        system_prompt: str = "",
        provider: BaseLLMProvider | None = None,
        config: Config | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.5,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt or (
            "You are a focused subagent. Complete the assigned task concisely "
            "and return only the result. Do not ask questions."
        )
        self._provider = provider
        self._config = config
        self.max_tokens = max_tokens
        self.temperature = temperature

    def execute(
        self,
        task: str,
        context: str = "",
        tools: list | None = None,
    ) -> SubagentResult:
        """Run the subagent with a given task."""
        provider = self._provider
        if not provider and self._config:
            try:
                provider = get_provider(self._config)
            except Exception:
                pass
        if not provider:
            return SubagentResult(
                task=task,
                output="",
                error="No LLM provider configured for subagent.",
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}"})
        messages.append({"role": "user", "content": task})

        try:
            response = provider.chat(
                messages=messages,
                tools=tools or None,
                stream=False,
            )
            return SubagentResult(
                task=task,
                output=response.content,
                tokens_used=response.usage.get("total_tokens", 0) if response.usage else 0,
            )
        except Exception as e:
            return SubagentResult(task=task, output="", error=str(e))


class SubagentManager:
    """Manages spawning and tracking subagents."""

    def __init__(self, config: Config | None = None) -> None:
        self._config = config
        self._subagents: dict[str, Subagent] = {}
        self._results: dict[str, list[SubagentResult]] = {}

    def spawn(
        self,
        name: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.5,
    ) -> Subagent:
        """Create a new subagent by name."""
        if name in self._subagents:
            return self._subagents[name]
        agent = Subagent(
            name=name,
            system_prompt=system_prompt,
            config=self._config,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self._subagents[name] = agent
        return agent

    def run(
        self,
        name: str,
        task: str,
        context: str = "",
    ) -> SubagentResult:
        """Run a subagent (auto-spawn if needed)."""
        agent = self._subagents.get(name) or self.spawn(name)
        result = agent.execute(task=task, context=context)
        self._results.setdefault(name, []).append(result)
        return result

    def run_parallel(self, tasks: list[tuple[str, str, str]]) -> list[SubagentResult]:
        """Run multiple subagents sequentially with different tasks.
        (True parallel would require asyncio — for now, sequential.)
        """
        results = []
        for name, task, context in tasks:
            result = self.run(name, task, context)
            results.append(result)
        return results

    def get_results(self, name: str) -> list[SubagentResult]:
        return self._results.get(name, [])

    def summarise_all(self) -> str:
        """Return a text summary of all subagent results."""
        parts = []
        for name, results in self._results.items():
            parts.append(f"=== Subagent: {name} ===")
            for i, r in enumerate(results, 1):
                status = "✓" if not r.error else "✗"
                parts.append(f"  Task #{i} [{status}]: {r.task[:80]}")
                if r.error:
                    parts.append(f"    Error: {r.error}")
                else:
                    parts.append(f"    Output: {r.output[:200]}...")
        return "\n".join(parts)

    def clear(self) -> None:
        self._subagents.clear()
        self._results.clear()