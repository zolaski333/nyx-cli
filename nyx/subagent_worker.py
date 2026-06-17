"""Process-isolated subagent execution.

The worker process recreates the provider and tool context from serialisable
inputs. This gives the parent a hard cancellation boundary: if the task exceeds
its timeout, the parent can terminate the worker process.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
import time
from typing import Any

from nyx.config import Config
from nyx.providers.base import ToolDefinition
from nyx.subagent import Subagent, SubagentResult, SubagentTask


def _result_from_payload(payload: dict[str, Any]) -> SubagentResult:
    return SubagentResult(
        task=str(payload.get("task", "")),
        output=str(payload.get("output", "")),
        tokens_used=int(payload.get("tokens_used", 0) or 0),
        error=payload.get("error"),
        status=str(payload.get("status", "failed")),
        error_type=payload.get("error_type"),
        agent_name=str(payload.get("agent_name", "")),
        steps=int(payload.get("steps", 0) or 0),
        tool_calls=list(payload.get("tool_calls") or []),
        duration_seconds=float(payload.get("duration_seconds", 0.0) or 0.0),
    )


def _worker_entry(
    result_queue: mp.Queue,
    task: SubagentTask,
    config: Config,
    tools: list[ToolDefinition] | None,
) -> None:
    started = time.monotonic()
    try:
        agent = Subagent(
            name=task.name,
            system_prompt=task.system_prompt,
            config=config,
            max_tokens=task.max_tokens,
            temperature=task.temperature,
            tools=tools,
        )
        result = agent.execute_task(task, tools=tools)
        result_queue.put({"ok": True, "result": result.to_dict()})
    except BaseException as exc:
        result_queue.put(
            {
                "ok": False,
                "result": SubagentResult(
                    task=task.task,
                    output="",
                    error=str(exc),
                    status="failed",
                    error_type=type(exc).__name__,
                    agent_name=task.name,
                    duration_seconds=time.monotonic() - started,
                ).to_dict(),
            }
        )


def _timeout_result(task: SubagentTask, started: float) -> SubagentResult:
    return SubagentResult(
        task=task.task,
        output="",
        error="Subagent worker timed out and was terminated.",
        status="timed_out",
        error_type="timeout",
        agent_name=task.name,
        duration_seconds=time.monotonic() - started,
    )


def run_subagent_task_in_process(
    *,
    task: SubagentTask,
    config: Config,
    tools: list[ToolDefinition] | None,
    timeout_seconds: float | None = None,
) -> SubagentResult:
    """Run a subagent task in a child process and return a structured result."""
    started = time.monotonic()
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_worker_entry,
        args=(result_queue, task, config, tools),
        daemon=True,
        name=f"nyx-subagent-{task.name}",
    )
    proc.start()
    proc.join(timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join(2)
        if proc.is_alive():
            proc.kill()
            proc.join(2)
        return _timeout_result(task, started)

    try:
        payload = result_queue.get_nowait()
    except queue.Empty:
        status = "failed" if proc.exitcode else "completed"
        error = None if proc.exitcode == 0 else f"Subagent worker exited with code {proc.exitcode}."
        return SubagentResult(
            task=task.task,
            output="",
            error=error,
            status=status,
            error_type="worker_exit" if error else None,
            agent_name=task.name,
            duration_seconds=time.monotonic() - started,
        )

    result = _result_from_payload(payload.get("result", {}))
    if not result.duration_seconds:
        result.duration_seconds = time.monotonic() - started
    return result
