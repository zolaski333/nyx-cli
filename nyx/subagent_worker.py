"""Process-isolated subagent execution.

The worker process recreates the provider and tool context from serialisable
inputs. This gives the parent a hard cancellation boundary: if the task exceeds
its timeout, the parent can terminate the worker process.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
import time
from typing import Any, Callable

from nyx.approval import run_exclusive_approval
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
    result_queue: mp.Queue[dict[str, Any]],
    approval_queue: mp.Queue[dict[str, Any]] | None,
    approval_response_queue: mp.Queue[dict[str, Any]] | None,
    task: SubagentTask,
    config: Config,
    tools: list[ToolDefinition] | None,
) -> None:
    started = time.monotonic()
    try:
        def _request_approval(kind: str, payload: dict[str, Any]) -> tuple[bool, str]:
            if approval_queue is None or approval_response_queue is None:
                return False, "No approval mechanism configured."
            request_id = f"{task.name}-{time.time_ns()}"
            approval_queue.put({"id": request_id, "kind": kind, **payload})
            while True:
                response = approval_response_queue.get()
                if response.get("id") == request_id:
                    return bool(response.get("approved")), str(response.get("reason", ""))

        def _command_approval(command: str) -> tuple[bool, str]:
            return _request_approval("command", {"command": command})

        def _file_approval(path: str, summary: str, diff: str) -> tuple[bool, str]:
            return _request_approval("file", {"path": path, "summary": summary, "diff": diff})

        use_approval_bridge = approval_queue is not None and approval_response_queue is not None
        agent = Subagent(
            name=task.name,
            system_prompt=task.system_prompt,
            config=config,
            max_tokens=task.max_tokens,
            temperature=task.temperature,
            tools=tools,
            on_command_approval=_command_approval if use_approval_bridge else None,
            on_file_approval=_file_approval if use_approval_bridge else None,
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
    on_command_approval: Callable[[str], tuple[bool, str]] | None = None,
    on_file_approval: Callable[[str, str, str], tuple[bool, str]] | None = None,
) -> SubagentResult:
    """Run a subagent task in a child process and return a structured result."""
    started = time.monotonic()
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue[dict[str, Any]] = ctx.Queue(maxsize=1)
    use_approval_bridge = on_command_approval is not None or on_file_approval is not None
    approval_queue: mp.Queue[dict[str, Any]] | None = ctx.Queue() if use_approval_bridge else None
    approval_response_queue: mp.Queue[dict[str, Any]] | None = ctx.Queue() if use_approval_bridge else None
    proc = ctx.Process(
        target=_worker_entry,
        args=(result_queue, approval_queue, approval_response_queue, task, config, tools),
        daemon=True,
        name=f"nyx-subagent-{task.name}",
    )
    proc.start()

    deadline = (time.monotonic() + timeout_seconds) if timeout_seconds is not None else None
    while proc.is_alive():
        try:
            payload = result_queue.get_nowait()
            proc.join(2)
            result = _result_from_payload(payload.get("result", {}))
            if not result.duration_seconds:
                result.duration_seconds = time.monotonic() - started
            return result
        except queue.Empty:
            pass

        if approval_queue is not None and approval_response_queue is not None:
            while True:
                try:
                    request = approval_queue.get_nowait()
                except queue.Empty:
                    break
                kind = request.get("kind")
                try:
                    def handle_request() -> tuple[bool, str]:
                        if kind == "command" and on_command_approval:
                            return on_command_approval(str(request.get("command", "")))
                        if kind == "file" and on_file_approval:
                            return on_file_approval(
                                str(request.get("path", "")),
                                str(request.get("summary", "")),
                                str(request.get("diff", "")),
                            )
                        return False, "No approval mechanism configured."
                    approved, reason = run_exclusive_approval(handle_request)
                except Exception as exc:
                    approved, reason = False, f"Approval handler error: {exc}"
                approval_response_queue.put({
                    "id": request.get("id"),
                    "approved": approved,
                    "reason": reason,
                })

        if deadline is not None and time.monotonic() >= deadline:
            break

        proc.join(0.05)

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
