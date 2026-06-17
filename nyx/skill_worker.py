"""Worker helpers for executing Python skills."""
from __future__ import annotations

import importlib.util
import multiprocessing as mp
import queue
import sys
import time
from pathlib import Path
from typing import Any

from nyx.skill_manager import Skill, SkillResult


def _payload_to_result(payload: dict[str, Any]) -> SkillResult:
    return SkillResult(
        name=str(payload.get("name", "")),
        output=str(payload.get("output", "")),
        status=str(payload.get("status", "failed")),
        error=payload.get("error"),
        error_type=payload.get("error_type"),
        duration_seconds=float(payload.get("duration_seconds", 0.0) or 0.0),
        truncated=bool(payload.get("truncated", False)),
    )


def _import_skill_module(skill: Skill):
    path = Path(skill.file_path).resolve()
    module_name = f"_nyx_skill_{skill.name}_{abs(hash(str(path)))}"
    search_root = path.parent.parent if path.name == "__init__.py" else path.parent
    sys.path.insert(0, str(search_root))
    try:
        if path.name == "__init__.py":
            spec = importlib.util.spec_from_file_location(
                module_name,
                path,
                submodule_search_locations=[str(path.parent)],
            )
        else:
            spec = importlib.util.spec_from_file_location(module_name, path)
        if not spec or not spec.loader:
            raise RuntimeError("Could not create import spec for skill.")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        try:
            sys.path.remove(str(search_root))
        except ValueError:
            pass


def run_skill_in_current_process(skill: Skill, arguments: dict[str, Any]) -> SkillResult:
    """Execute a skill in the current process.

    This is mainly for explicit compatibility/testing. The default path uses a
    child process.
    """
    started = time.monotonic()
    try:
        mod = _import_skill_module(skill)
        execute_fn = getattr(mod, skill.entrypoint, None)
        if not callable(execute_fn):
            return SkillResult(
                name=skill.name,
                status="failed",
                error=f"Missing callable '{skill.entrypoint}(arguments)'.",
                error_type="missing_entrypoint",
                duration_seconds=time.monotonic() - started,
            )
        output = execute_fn(arguments)
        return SkillResult(
            name=skill.name,
            output=str(output),
            status="completed",
            duration_seconds=time.monotonic() - started,
        )
    except BaseException as exc:
        return SkillResult(
            name=skill.name,
            status="failed",
            error=str(exc),
            error_type=type(exc).__name__,
            duration_seconds=time.monotonic() - started,
        )


def _worker_entry(result_queue: mp.Queue, skill: Skill, arguments: dict[str, Any]) -> None:
    result = run_skill_in_current_process(skill, arguments)
    result_queue.put(result.to_dict())


def run_skill_in_process(
    *,
    skill: Skill,
    arguments: dict[str, Any],
    timeout_seconds: float | None,
) -> SkillResult:
    """Execute a skill in a child process with a hard timeout."""
    started = time.monotonic()
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_worker_entry,
        args=(result_queue, skill, arguments),
        daemon=True,
        name=f"nyx-skill-{skill.name}",
    )
    proc.start()
    proc.join(timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join(2)
        if proc.is_alive():
            proc.kill()
            proc.join(2)
        return SkillResult(
            name=skill.name,
            status="timed_out",
            error="Skill worker timed out and was terminated.",
            error_type="timeout",
            duration_seconds=time.monotonic() - started,
        )

    try:
        payload = result_queue.get_nowait()
    except queue.Empty:
        error = None if proc.exitcode == 0 else f"Skill worker exited with code {proc.exitcode}."
        return SkillResult(
            name=skill.name,
            status="completed" if error is None else "failed",
            error=error,
            error_type="worker_exit" if error else None,
            duration_seconds=time.monotonic() - started,
        )

    result = _payload_to_result(payload)
    result.duration_seconds = result.duration_seconds or (time.monotonic() - started)
    return result
