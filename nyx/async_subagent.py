"""
Async subagent system — parallel execution of multiple subagents.

Uses `concurrent.futures.ThreadPoolExecutor` for CPU/IO-bound parallel
execution without requiring asyncio (compatible with Python 3.10+ stdlib).
"""
from __future__ import annotations

import concurrent.futures
import threading
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

from nyx.subagent import Subagent, SubagentResult, SubagentTask
from nyx.config import Config
from nyx.providers.base import BaseLLMProvider, ToolCall, ToolDefinition

if TYPE_CHECKING:
    from nyx.tools import ToolContext


@dataclass
class ParallelTask:
    """A task to run in parallel."""
    name: str
    task: str
    context: str = ""
    system_prompt: str = ""
    max_tokens: int = 2048
    temperature: float = 0.5
    max_steps: int = 10
    timeout_seconds: float | None = None

    def to_subagent_task(self) -> SubagentTask:
        return SubagentTask(
            name=self.name,
            task=self.task,
            context=self.context,
            system_prompt=self.system_prompt,
            max_steps=self.max_steps,
            timeout_seconds=self.timeout_seconds,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )


@dataclass
class ParallelResult:
    """Result from a parallel execution batch."""
    results: list[SubagentResult] = field(default_factory=list)
    completed: int = 0
    failed: int = 0
    total_tokens: int = 0
    timed_out: int = 0

    @property
    def all_successful(self) -> bool:
        return self.failed == 0


class AsyncSubagentManager:
    """
    Manages parallel subagent execution.

    Uses a thread pool to run multiple subagents concurrently.
    Thread-safe — multiple agents can be spawned simultaneously.
    """

    def __init__(
        self,
        config: Config | None = None,
        max_workers: int = 4,
        provider_factory: Callable[[], BaseLLMProvider] | None = None,
        process_isolation: bool = True,
        default_timeout_seconds: float | None = None,
    ) -> None:
        self._config = config
        self._max_workers = max_workers
        self._provider_factory = provider_factory
        self.process_isolation = process_isolation
        self.default_timeout_seconds = default_timeout_seconds
        self._lock = threading.Lock()
        self._agents: dict[str, Subagent] = {}
        self._results: dict[str, list[SubagentResult]] = {}
        self._default_tools: list[ToolDefinition] | None = None
        self._tool_executor: Callable[[ToolCall], str] | None = None
        self._context: ToolContext | None = None

    def set_default_tools(self, tools: list[ToolDefinition] | None) -> None:
        """Set the default tool subset for spawned subagents."""
        self._default_tools = tools
        with self._lock:
            for agent in self._agents.values():
                agent._tools = tools

    def set_tool_executor(self, executor: Callable[[ToolCall], str] | None) -> None:
        """Set the shared tool executor used by spawned subagents."""
        self._tool_executor = executor
        with self._lock:
            for agent in self._agents.values():
                agent._tool_executor = executor

    def set_context(self, context: ToolContext | None) -> None:
        """Set the tool execution context for all subagents spawned by this manager."""
        with self._lock:
            self._context = context
            for agent in self._agents.values():
                agent.context = context

    def spawn(
        self,
        name: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.5,
    ) -> Subagent:
        """Create or retrieve a subagent by name (thread-safe)."""
        with self._lock:
            if name in self._agents:
                return self._agents[name]
            provider = self._provider_factory() if self._provider_factory else None
            agent = Subagent(
                name=name,
                system_prompt=system_prompt,
                provider=provider,
                config=self._config,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=self._default_tools,
                tool_executor=self._tool_executor,
                context=self._context,
            )
            self._agents[name] = agent
            return agent

    def run(self, name: str, task: str, context: str = "") -> SubagentResult:
        """Run a single subagent (sequential fallback)."""
        return self.run_task(SubagentTask(name=name, task=task, context=context))

    def run_task(self, task: SubagentTask) -> SubagentResult:
        """Run a structured task through one subagent."""
        agent = self._agents.get(task.name) or self.spawn(
            task.name,
            system_prompt=task.system_prompt,
            max_tokens=task.max_tokens,
            temperature=task.temperature,
        )
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
        with self._lock:
            self._results.setdefault(task.name, []).append(result)
        return result

    def run_parallel(self, tasks: list[ParallelTask]) -> ParallelResult:
        """
        Execute multiple subagent tasks in parallel using a thread pool.

        Example:
            manager = AsyncSubagentManager(config)
            tasks = [
                ParallelTask("researcher", "search for X"),
                ParallelTask("coder", "implement Y function"),
            ]
            result = manager.run_parallel(tasks)
        """
        if not tasks:
            return ParallelResult()

        parallel_result = ParallelResult()
        pool_size = min(self._max_workers, len(tasks))

        ordered_results: list[SubagentResult | None] = [None] * len(tasks)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=pool_size)
        try:
            future_to_task = {}

            for idx, pt in enumerate(tasks):
                # Ensure subagent exists
                self._agents.get(pt.name) or self.spawn(
                    name=pt.name,
                    system_prompt=pt.system_prompt,
                    max_tokens=pt.max_tokens,
                    temperature=pt.temperature,
                )

                future = executor.submit(self.run_task, pt.to_subagent_task())
                future_to_task[future] = (idx, pt)

            max_timeout = None
            timeouts: list[float] = []
            for pt in tasks:
                timeout_seconds = pt.timeout_seconds or self.default_timeout_seconds
                if timeout_seconds is not None:
                    timeouts.append(timeout_seconds)
            if timeouts:
                max_timeout = max(timeouts)

            done, pending = concurrent.futures.wait(
                future_to_task,
                timeout=max_timeout,
                return_when=concurrent.futures.ALL_COMPLETED,
            )

            for future in done:
                idx, task_info = future_to_task[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = SubagentResult(
                        task=task_info.task,
                        output="",
                        error=str(e),
                        status="failed",
                        error_type=type(e).__name__,
                        agent_name=task_info.name,
                    )

                ordered_results[idx] = result

            for future in pending:
                idx, task_info = future_to_task[future]
                future.cancel()
                result = SubagentResult(
                    task=task_info.task,
                    output="",
                    error="Subagent timed out.",
                    status="timed_out",
                    error_type="timeout",
                    agent_name=task_info.name,
                )
                ordered_results[idx] = result
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        for ordered_result in ordered_results:
            if ordered_result is None:
                continue
            if ordered_result.status == "timed_out":
                parallel_result.timed_out += 1
            if ordered_result.error:
                parallel_result.failed += 1
            else:
                parallel_result.completed += 1
            parallel_result.total_tokens += ordered_result.tokens_used
            parallel_result.results.append(ordered_result)

        return parallel_result

    def run_batch(
        self,
        name_prefix: str,
        tasks: list[str],
        context: str = "",
        system_prompt: str = "",
    ) -> ParallelResult:
        """
        Convenience: run the same subagent pattern across multiple task strings.

        Each task gets a unique name: {name_prefix}_{i}
        """
        parallel_tasks = [
            ParallelTask(
                name=f"{name_prefix}_{i}",
                task=t,
                context=context,
                system_prompt=system_prompt,
            )
            for i, t in enumerate(tasks)
        ]
        return self.run_parallel(parallel_tasks)

    def get_results(self, name: str) -> list[SubagentResult]:
        with self._lock:
            return list(self._results.get(name, []))

    def summarise_all(self) -> str:
        """Return a text summary of all subagent results."""
        with self._lock:
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
        with self._lock:
            self._agents.clear()
            self._results.clear()
