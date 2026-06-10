"""
Nyx — Structured JSON logger with session tracking, cost accounting,
and optional output to file or stderr.

Every agent interaction (LLM calls, tool executions, errors) can be
recorded as structured JSON for downstream analysis, debugging, and
cost monitoring.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

# Approximate per-token costs (USD) — used when the API does not return cost info
MODEL_COST_PER_1K_INPUT: dict[str, float] = {
    "deepseek/deepseek-v4-flash": 0.000_15,
    "deepseek/deepseek-chat": 0.000_15,
    "openai/gpt-4o": 0.0025,
    "openai/gpt-4o-mini": 0.000_15,
    "openai/gpt-4-turbo": 0.01,
    "anthropic/claude-3-opus": 0.015,
    "anthropic/claude-3-sonnet": 0.003,
    "anthropic/claude-3-haiku": 0.000_25,
    "anthropic/claude-3-5-sonnet": 0.003,
}

MODEL_COST_PER_1K_OUTPUT: dict[str, float] = {
    "deepseek/deepseek-v4-flash": 0.000_60,
    "deepseek/deepseek-chat": 0.000_60,
    "openai/gpt-4o": 0.01,
    "openai/gpt-4o-mini": 0.000_60,
    "openai/gpt-4-turbo": 0.03,
    "anthropic/claude-3-opus": 0.075,
    "anthropic/claude-3-sonnet": 0.015,
    "anthropic/claude-3-haiku": 0.001_25,
    "anthropic/claude-3-5-sonnet": 0.015,
}

DEFAULT_INPUT_COST = 0.000_2
DEFAULT_OUTPUT_COST = 0.000_8


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate USD cost for a model call based on token counts."""
    input_rate = MODEL_COST_PER_1K_INPUT.get(model, DEFAULT_INPUT_COST)
    output_rate = MODEL_COST_PER_1K_OUTPUT.get(model, DEFAULT_OUTPUT_COST)
    return (input_tokens / 1000 * input_rate) + (output_tokens / 1000 * output_rate)


# ---------------------------------------------------------------------------
# JSON log entry
# ---------------------------------------------------------------------------


@dataclass
class JSONLogEntry:
    """A single structured JSON log entry."""
    timestamp: float
    event_type: str          # "llm_call", "tool_call", "tool_result", "error", "session_start", "session_end"
    session_id: str
    data: dict[str, Any]     # Event-specific payload
    level: str = "info"      # "info", "warning", "error"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


# ---------------------------------------------------------------------------
# JSON Logger
# ---------------------------------------------------------------------------


from queue import Queue

class JSONLogger:
    """Optional structured JSON logger with session tracking.

    Writes newline-delimited JSON (NDJSON) to a file and/or stderr.
    Tracks cumulative costs across a session.
    """

    def __init__(
        self,
        output_path: str | Path | None = None,
        session_id: str | None = None,
        log_to_stderr: bool = False,
        enabled: bool = True,
        model: str = "",
    ) -> None:
        self._enabled = enabled
        self._session_id = session_id or uuid.uuid4().hex[:12]
        self._log_to_stderr = log_to_stderr
        self._model = model
        self._lock = threading.Lock()
        self._file: Any = None
        self._file_path: Path | None = None
        self._queue: Queue = Queue()
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Cumulative session stats
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cost: float = 0.0
        self.total_llm_calls: int = 0
        self.total_tool_calls: int = 0
        self.total_errors: int = 0

        if output_path:
            self._setup_file(output_path)

        if self._enabled:
            self._worker_thread = threading.Thread(target=self._worker, daemon=True)
            self._worker_thread.start()
            self._write_entry(JSONLogEntry(
                timestamp=time.time(),
                event_type="session_start",
                session_id=self._session_id,
                data={"model": model},
            ))

    def _setup_file(self, output_path: str | Path) -> None:
        """Set up the JSON log file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file_path = path
        self._file = path.open("a", encoding="utf-8")
        logger.info("JSON log initialized: %s", self._file_path)

    def _worker(self) -> None:
        """Background worker to write log entries to disk asynchronously."""
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                entry = self._queue.get(timeout=0.1)
            except Exception:
                continue
            try:
                line = entry.to_json() + "\n"
                with self._lock:
                    if self._file:
                        self._file.write(line)
                        self._file.flush()
            except Exception as e:
                logger.error("Failed to write JSON log entry: %s", e)
            finally:
                self._queue.task_done()

    def _write_entry(self, entry: JSONLogEntry) -> None:
        """Write a log entry to all configured outputs."""
        if not self._enabled:
            return
        if self._log_to_stderr:
            line = entry.to_json() + "\n"
            print(line, file=sys.stderr, end="", flush=True)
        self._queue.put(entry)

    # ------------------------------------------------------------------
    # Logging methods
    # ------------------------------------------------------------------

    def log_llm_call(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "",
        duration_ms: float = 0.0,
        error: str = "",
    ) -> None:
        """Log an LLM API call with token usage and cost."""
        if not self._enabled:
            return
        model = model or self._model
        cost = estimate_cost(model, input_tokens, output_tokens)

        with self._lock:
            self.total_llm_calls += 1
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost += cost
            if error:
                self.total_errors += 1

        level = "error" if error else "info"
        self._write_entry(JSONLogEntry(
            timestamp=time.time(),
            event_type="llm_call",
            session_id=self._session_id,
            level=level,
            data={
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": round(cost, 6),
                "duration_ms": round(duration_ms, 2),
                "error": error,
                "cumulative_cost": round(self.total_cost, 6),
            },
        ))

    def log_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        duration_ms: float = 0.0,
        error: str = "",
    ) -> None:
        """Log a tool call attempt (before execution)."""
        if not self._enabled:
            return
        with self._lock:
            self.total_tool_calls += 1
            if error:
                self.total_errors += 1

        level = "error" if error else "info"
        self._write_entry(JSONLogEntry(
            timestamp=time.time(),
            event_type="tool_call",
            session_id=self._session_id,
            level=level,
            data={
                "tool": tool_name,
                "arguments": arguments,
                "duration_ms": round(duration_ms, 2),
                "error": error,
            },
        ))

    def log_tool_result(
        self,
        tool_name: str,
        result: str,
        duration_ms: float = 0.0,
        error: str = "",
    ) -> None:
        """Log a tool execution result."""
        if not self._enabled:
            return
        level = "error" if error else "info"
        self._write_entry(JSONLogEntry(
            timestamp=time.time(),
            event_type="tool_result",
            session_id=self._session_id,
            level=level,
            data={
                "tool": tool_name,
                "result_preview": result[:1000],
                "duration_ms": round(duration_ms, 2),
                "error": error,
            },
        ))

    def log_error(
        self,
        source: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log an error event."""
        if not self._enabled:
            return
        with self._lock:
            self.total_errors += 1
        self._write_entry(JSONLogEntry(
            timestamp=time.time(),
            event_type="error",
            session_id=self._session_id,
            level="error",
            data={
                "source": source,
                "message": message,
                **(details or {}),
            },
        ))

    def log_event(self, event_type: str, data: dict[str, Any], level: str = "info") -> None:
        """Log an arbitrary event."""
        self._write_entry(JSONLogEntry(
            timestamp=time.time(),
            event_type=event_type,
            session_id=self._session_id,
            level=level,
            data=data,
        ))

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def get_session_summary(self) -> dict[str, Any]:
        """Return a summary of the current session."""
        return {
            "session_id": self._session_id,
            "total_llm_calls": self.total_llm_calls,
            "total_tool_calls": self.total_tool_calls,
            "total_errors": self.total_errors,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": round(self.total_cost, 6),
        }

    def close(self) -> None:
        """Close the logger and write a session_end event."""
        if self._enabled:
            self._write_entry(JSONLogEntry(
                timestamp=time.time(),
                event_type="session_end",
                session_id=self._session_id,
                data=self.get_session_summary(),
            ))
        
        # Stop worker thread and flush remaining logs
        self._stop_event.set()
        if self._worker_thread:
            try:
                self._queue.join()
                self._worker_thread.join(timeout=2.0)
            except Exception:
                pass
            
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def enabled(self) -> bool:
        return self._enabled