"""
Subagent system — spawns child agents that report back.

A subagent is a self-contained LLM conversation with its own system prompt.
It can be given a task, run independently, and return results.
Useful for parallel research, code generation, or complex subtasks.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

from nyx.config import Config
from nyx.providers.base import BaseLLMProvider, ToolCall, ToolDefinition
from nyx.providers import get_provider

if TYPE_CHECKING:
    from nyx.tools import ToolContext


@dataclass
class SubagentTask:
    """Structured task contract for a subagent run."""
    name: str
    task: str
    context: str = ""
    system_prompt: str = ""
    max_steps: int = 10
    timeout_seconds: float | None = None
    max_tokens: int = 2048
    temperature: float = 0.5
    expected_output: str = "Concise answer or structured findings."


@dataclass
class SubagentResult:
    """Result from a subagent execution."""
    task: str
    output: str
    tokens_used: int = 0
    error: str | None = None
    status: str = "completed"
    error_type: str | None = None
    agent_name: str = ""
    steps: int = 0
    tool_calls: list[dict[str, Any]] | None = None
    duration_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None and self.status == "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "output": self.output,
            "tokens_used": self.tokens_used,
            "error": self.error,
            "status": self.status,
            "error_type": self.error_type,
            "agent_name": self.agent_name,
            "steps": self.steps,
            "tool_calls": self.tool_calls or [],
            "duration_seconds": self.duration_seconds,
        }


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
        tools: list[ToolDefinition] | None = None,
        tool_executor: Callable[[ToolCall], str] | None = None,
        context: ToolContext | None = None,
        on_command_approval: Callable[[str], tuple[bool, str]] | None = None,
        on_file_approval: Callable[[str, str, str], tuple[bool, str]] | None = None,
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
        self._tools = tools  # Controlled tool subset (None = no tools)
        self._tool_executor = tool_executor
        self.context = context
        self._on_command_approval = on_command_approval
        self._on_file_approval = on_file_approval

    @property
    def can_run_isolated(self) -> bool:
        """Whether this subagent can be recreated in a worker process."""
        return self._provider is None and self._tool_executor is None

    def execute(
        self,
        task: str,
        context: str = "",
        tools: list[ToolDefinition] | None = None,
        max_steps: int | None = None,
        timeout_seconds: float | None = None,
    ) -> SubagentResult:
        """Run the subagent with a given task.

        Args:
            task: The task description for the subagent.
            context: Optional context to prepend.
            tools: Optional tool list override. If None, uses self._tools.
            max_steps: Maximum reasoning steps.
            timeout_seconds: Timeout in seconds. Note that when running in-process
                (process_isolation=False), this is checked only between steps/calls,
                so a blocking provider call will not be interrupted immediately.

        Returns:
            SubagentResult with the output or error.
        """
        started = time.monotonic()
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
                status="failed",
                error_type="provider_unavailable",
                agent_name=self.name,
                duration_seconds=time.monotonic() - started,
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}"})
        messages.append({"role": "user", "content": task})

        # Use provided tools, or instance tools, or none
        effective_tools = tools if tools is not None else self._tools

        # Maximum depth of tool calls
        max_steps = max_steps if max_steps is not None else 10
        total_tokens = 0
        use_anthropic_format = self._config.provider == "anthropic" if self._config else False
        observed_tool_calls: list[dict[str, Any]] = []
        allowed_tool_names = {t.name for t in effective_tools or []}

        try:
            for step in range(max_steps):
                if timeout_seconds and (time.monotonic() - started > timeout_seconds):
                    return SubagentResult(
                        task=task,
                        output="",
                        error="Subagent task execution exceeded timeout.",
                        status="timed_out",
                        error_type="timeout",
                        agent_name=self.name,
                        steps=step,
                        tool_calls=observed_tool_calls,
                        duration_seconds=time.monotonic() - started,
                    )

                response = provider.chat(
                    messages=messages,
                    tools=effective_tools or None,
                    stream=False,
                )
                
                if response.usage:
                    total_tokens += response.usage.get("total_tokens", 0) or response.usage.get("prompt_tokens", 0) + response.usage.get("completion_tokens", 0)
                
                # Append assistant response
                content_str = response.content or ""
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": content_str,
                }
                if response.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                        for tc in response.tool_calls
                    ]
                messages.append(assistant_msg)

                # If no tool calls, this is the final answer!
                if not response.tool_calls or not effective_tools:
                    return SubagentResult(
                        task=task,
                        output=response.content or "",
                        tokens_used=total_tokens,
                        status="completed",
                        agent_name=self.name,
                        steps=step + 1,
                        tool_calls=observed_tool_calls,
                        duration_seconds=time.monotonic() - started,
                    )

                # Execute all tool calls in this turn
                for tc in response.tool_calls:
                    if timeout_seconds and (time.monotonic() - started > timeout_seconds):
                        return SubagentResult(
                            task=task,
                            output="",
                            error="Subagent task execution exceeded timeout.",
                            status="timed_out",
                            error_type="timeout",
                            agent_name=self.name,
                            steps=step + 1,
                            tool_calls=observed_tool_calls,
                            duration_seconds=time.monotonic() - started,
                        )

                    observed_tool_calls.append({"name": tc.name, "arguments": dict(tc.arguments)})
                    if allowed_tool_names and tc.name not in allowed_tool_names:
                        return SubagentResult(
                            task=task,
                            output="",
                            tokens_used=total_tokens,
                            error=f"Tool '{tc.name}' is not allowed for subagent '{self.name}'.",
                            status="failed",
                            error_type="tool_not_allowed",
                            agent_name=self.name,
                            steps=step + 1,
                            tool_calls=observed_tool_calls,
                            duration_seconds=time.monotonic() - started,
                        )
                    if self._tool_executor:
                        tool_result = self._execute_tool_call(tc, effective_tools)
                    elif self.context:
                        from nyx.tools import execute_tool
                        tool_result = execute_tool(tc, self.context)
                    else:
                        tool_result = self._execute_tool_call(tc, effective_tools)

                    # Append tool result to messages
                    if use_anthropic_format:
                        messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tc.id,
                                    "content": [{"type": "text", "text": tool_result}],
                                }
                            ],
                        })
                    else:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.name,
                            "content": tool_result,
                        })
                
                # If finish tool was called, break early and return its result
                finish_calls = [tc for tc in response.tool_calls if tc.name == "finish"]
                if finish_calls:
                    finish_args = finish_calls[0].arguments
                    summary = finish_args.get("summary", "")
                    result = finish_args.get("result", "")
                    output_val = f"Summary: {summary}\nResult: {result}" if (summary or result) else (response.content or "Task completed.")
                    return SubagentResult(
                        task=task,
                        output=output_val,
                        tokens_used=total_tokens,
                        status="completed",
                        agent_name=self.name,
                        steps=step + 1,
                        tool_calls=observed_tool_calls,
                        duration_seconds=time.monotonic() - started,
                    )

            # If we reached max_steps, return the last response content
            return SubagentResult(
                task=task,
                output=messages[-1].get("content") or "",
                error="Max reasoning steps reached.",
                status="failed",
                error_type="max_steps",
                tokens_used=total_tokens,
                agent_name=self.name,
                steps=max_steps,
                tool_calls=observed_tool_calls,
                duration_seconds=time.monotonic() - started,
            )

        except Exception as e:
            return SubagentResult(
                task=task,
                output="",
                error=str(e),
                status="failed",
                error_type=type(e).__name__,
                tokens_used=total_tokens,
                agent_name=self.name,
                duration_seconds=time.monotonic() - started,
            )

    def execute_task(self, task: SubagentTask, tools: list[ToolDefinition] | None = None) -> SubagentResult:
        """Run a structured task contract."""
        if task.system_prompt:
            self.system_prompt = task.system_prompt
        self.max_tokens = task.max_tokens
        self.temperature = task.temperature
        return self.execute(
            task=task.task,
            context=task.context,
            tools=tools,
            max_steps=task.max_steps,
            timeout_seconds=task.timeout_seconds,
        )

    def _execute_tool_call(self, tc: ToolCall, tools: list[ToolDefinition]) -> str:
        """Execute a single tool call for the subagent.

        Enforces all sandbox and permission checks by delegating to the central executor.
        """
        if self._tool_executor:
            return self._tool_executor(tc)

        from nyx.tools import ToolContext, execute_tool
        from nyx.sandbox import Sandbox
        from nyx.permissions import PermissionManager
        from nyx.audit import AuditTrail
        from nyx.diff_tool import PatchTool
        from nyx.memory import MemoryManager

        config = self._config
        if not config:
            from nyx.config import Config
            config = Config.load()

        sandbox = Sandbox(
            project_root=config.project_dir if (config and config.sandbox_enabled) else None,
            allow_paths=config.sandbox_allow_paths if config else None,
            deny_paths=config.sandbox_deny_paths if config else None,
        )
        permissions = PermissionManager(config.permissions_config if (config and config.permissions_config) else None)
        audit = AuditTrail(enabled=False)
        patch_tool = PatchTool(project_dir=config.project_dir if config else None)
        memory = MemoryManager(provider=self._provider)

        def _approval_callback(category: str, description: str, target: str) -> tuple[bool, str]:
            if category == "shell" and self._on_command_approval:
                return self._on_command_approval(target)
            if category == "filesystem" and self._on_file_approval:
                return self._on_file_approval(target, description, "")
            return False, "No approval mechanism configured."

        permissions.set_approval_callback(_approval_callback)
        patch_tool.set_approval_callback(self._on_file_approval)

        context = ToolContext(
            config=config,
            sandbox=sandbox,
            permissions=permissions,
            audit=audit,
            patch_tool=patch_tool,
            memory=memory,
            on_command_approval=self._on_command_approval,
            on_file_approval=self._on_file_approval,
        )

        return execute_tool(tc, context)


class SubagentManager:
    """Manages spawning and tracking subagents."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        process_isolation: bool = True,
        default_timeout_seconds: float | None = None,
    ) -> None:
        self._config = config
        self.process_isolation = process_isolation
        self.default_timeout_seconds = default_timeout_seconds
        self._subagents: dict[str, Subagent] = {}
        self._results: dict[str, list[SubagentResult]] = {}
        self._default_tools: list[ToolDefinition] | None = None  # Controlled tool subset
        self._tool_executor: Callable[[ToolCall], str] | None = None
        self._progress_callback: Callable[[str, int, int], None] | None = None
        self._context: ToolContext | None = None

    def set_progress_callback(self, callback: Callable[[str, int, int], None] | None) -> None:
        """Set a callback for progress updates: (label, current, total)."""
        self._progress_callback = callback

    def set_default_tools(self, tools: list[ToolDefinition] | None) -> None:
        """Set the default tool subset for all spawned subagents."""
        self._default_tools = tools
        for agent in self._subagents.values():
            agent._tools = tools

    def set_tool_executor(self, executor: Callable[[ToolCall], str] | None) -> None:
        """Set the shared tool executor used by spawned subagents."""
        self._tool_executor = executor
        for agent in self._subagents.values():
            agent._tool_executor = executor

    def set_context(self, context: ToolContext | None) -> None:
        """Set the tool execution context for all subagents spawned by this manager."""
        self._context = context
        for agent in self._subagents.values():
            agent.context = context

    def spawn(
        self,
        name: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.5,
        tools: list[ToolDefinition] | None = None,
    ) -> Subagent:
        """Create a new subagent by name.

        Args:
            name: Unique name for the subagent.
            system_prompt: Custom system prompt.
            max_tokens: Maximum tokens for responses.
            temperature: LLM temperature.
            tools: Optional tool subset. Falls back to default_tools if None.

        Returns:
            Subagent instance.
        """
        if name in self._subagents:
            return self._subagents[name]
        effective_tools = tools if tools is not None else self._default_tools
        agent = Subagent(
            name=name,
            system_prompt=system_prompt,
            config=self._config,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=effective_tools,
            tool_executor=self._tool_executor,
            context=self._context,
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
        return self.run_task(SubagentTask(name=name, task=task, context=context))

    def run_task(self, task: SubagentTask) -> SubagentResult:
        """Run a structured subagent task."""
        agent = self._subagents.get(task.name) or self.spawn(
            task.name,
            system_prompt=task.system_prompt,
            max_tokens=task.max_tokens,
            temperature=task.temperature,
        )
        if self._progress_callback:
            self._progress_callback(task.name, 0, 1)
        timeout_seconds = task.timeout_seconds or self.default_timeout_seconds
        if self.process_isolation and self._config and agent.can_run_isolated:
            from nyx.subagent_worker import run_subagent_task_in_process

            result = run_subagent_task_in_process(
                task=task,
                config=self._config,
                tools=agent._tools,
                timeout_seconds=timeout_seconds,
                on_command_approval=self._context.on_command_approval if self._context else None,
                on_file_approval=self._context.on_file_approval if self._context else None,
            )
        else:
            result = agent.execute_task(task, tools=agent._tools)
        if self._progress_callback:
            self._progress_callback(task.name, 1, 1)
        self._results.setdefault(task.name, []).append(result)
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
