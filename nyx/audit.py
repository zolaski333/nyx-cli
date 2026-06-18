"""
Nyx — Structured audit trail for all agent actions.

Logs every tool execution, permission decision, and file operation
to a structured JSON-lines file for later analysis, debugging, and
security review.

Audit entries are written as newline-delimited JSON (NDJSON) for
easy streaming and parsing.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from queue import Queue
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audit entry types
# ---------------------------------------------------------------------------

@dataclass
class AuditEntry:
    """A single audit log entry."""

    timestamp: float           # Unix timestamp
    event_type: str            # e.g. "tool_call", "permission_check", "file_write", "approval"
    agent_id: str              # Agent/session identifier
    details: dict[str, Any]    # Event-specific data
    level: str = "info"        # "info", "warning", "error", "security"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------

class AuditTrail:
    """Thread-safe audit trail that writes structured JSON-lines to a file."""

    def __init__(
        self,
        output_dir: str | Path | None = None,
        agent_id: str = "nyx",
        max_file_size_mb: int = 50,
        enabled: bool = True,
    ) -> None:
        self._agent_id = agent_id
        self._enabled = enabled
        self._max_file_size = max_file_size_mb * 1024 * 1024
        self._lock = threading.Lock()
        self._file: Any = None  # Type: IO | None
        self._file_path: Path | None = None
        self._entry_count = 0
        self._queue: Queue[AuditEntry] = Queue()
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        if output_dir and self._enabled:
            self._setup_file(output_dir)
            self._worker_thread = threading.Thread(target=self._worker, daemon=True)
            self._worker_thread.start()

    def _setup_file(self, output_dir: str | Path) -> None:
        """Set up the audit log file."""
        dir_path = Path(output_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Create a new log file with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._file_path = dir_path / f"audit_{timestamp}.ndjson"
        self._file = self._file_path.open("a", encoding="utf-8")
        logger.info("Audit trail initialized: %s", self._file_path)

    def set_output_dir(self, output_dir: str | Path) -> None:
        """Set or change the output directory for audit logs."""
        with self._lock:
            # Stop existing worker
            if self._worker_thread:
                self._stop_event.set()
                try:
                    self._queue.join()
                    self._worker_thread.join(timeout=2.0)
                except Exception:
                    pass
                self._stop_event.clear()
                self._worker_thread = None

            if self._file:
                self._file.close()
            
            self._setup_file(output_dir)
            
            if self._enabled:
                self._worker_thread = threading.Thread(target=self._worker, daemon=True)
                self._worker_thread.start()

    def _worker(self) -> None:
        """Background worker to write audit logs asynchronously."""
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
                        self._check_rotation()
            except Exception as e:
                logger.error("Failed to write audit entry: %s", e)
            finally:
                self._queue.task_done()

    def _check_rotation(self) -> None:
        """Rotate the log file if it exceeds the maximum size."""
        if not self._file_path or not self._file:
            return
        try:
            if self._file_path.stat().st_size > self._max_file_size:
                # Close current file and open a new one
                self._file.close()
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                self._file_path = self._file_path.parent / f"audit_{timestamp}.ndjson"
                self._file = self._file_path.open("a", encoding="utf-8")
                logger.info("Audit log rotated: %s", self._file_path)
        except OSError:
            pass

    def log(self, entry: AuditEntry) -> None:
        """Log an audit entry."""
        if not self._enabled:
            return

        with self._lock:
            self._entry_count += 1
        
        self._queue.put(entry)

    def flush(self) -> None:
        """Force-flush all buffered entries to disk."""
        if self._enabled:
            try:
                self._queue.join()
            except Exception:
                pass

    def close(self) -> None:
        """Close the audit log file."""
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
            logger.info("Audit trail closed. Total entries: %d", self._entry_count)

    # ------------------------------------------------------------------
    # Convenience logging methods
    # ------------------------------------------------------------------

    def log_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: str,
        duration_ms: float,
        level: str = "info",
    ) -> None:
        """Log a tool execution."""
        self.log(AuditEntry(
            timestamp=time.time(),
            event_type="tool_call",
            agent_id=self._agent_id,
            level=level,
            details={
                "tool": tool_name,
                "arguments": arguments,
                "result_preview": result[:500],
                "duration_ms": round(duration_ms, 2),
            },
        ))

    def log_permission_check(
        self,
        category: str,
        target: str,
        level: str,
        approved: bool,
        reason: str = "",
    ) -> None:
        """Log a permission check result."""
        self.log(AuditEntry(
            timestamp=time.time(),
            event_type="permission_check",
            agent_id=self._agent_id,
            level="security" if not approved else "info",
            details={
                "category": category,
                "target": target,
                "permission_level": level,
                "approved": approved,
                "reason": reason,
            },
        ))

    def log_file_operation(
        self,
        operation: str,
        path: str,
        size_bytes: int = 0,
        success: bool = True,
        error: str = "",
    ) -> None:
        """Log a file system operation."""
        self.log(AuditEntry(
            timestamp=time.time(),
            event_type="file_operation",
            agent_id=self._agent_id,
            level="error" if not success else "info",
            details={
                "operation": operation,
                "path": path,
                "size_bytes": size_bytes,
                "success": success,
                "error": error,
            },
        ))

    def log_approval(
        self,
        category: str,
        description: str,
        target: str,
        approved: bool,
        reason: str = "",
    ) -> None:
        """Log an approval decision."""
        self.log(AuditEntry(
            timestamp=time.time(),
            event_type="approval",
            agent_id=self._agent_id,
            level="security",
            details={
                "category": category,
                "description": description,
                "target": target,
                "approved": approved,
                "reason": reason,
            },
        ))

    def log_error(self, source: str, message: str, details: dict[str, Any] | None = None) -> None:
        """Log an error event."""
        self.log(AuditEntry(
            timestamp=time.time(),
            event_type="error",
            agent_id=self._agent_id,
            level="error",
            details={
                "source": source,
                "message": message,
                **(details or {}),
            },
        ))

    def log_security_event(self, event: str, details: dict[str, Any]) -> None:
        """Log a security-related event."""
        self.log(AuditEntry(
            timestamp=time.time(),
            event_type="security",
            agent_id=self._agent_id,
            level="security",
            details={
                "event": event,
                **details,
            },
        ))

    # ------------------------------------------------------------------
    # Query / read-back
    # ------------------------------------------------------------------

    def read_entries(
        self,
        limit: int = 100,
        event_type: str | None = None,
        level: str | None = None,
    ) -> list[dict[str, Any]]:
        """Read back audit entries from the current log file."""
        if not self._file_path or not self._file_path.exists():
            return []

        entries: list[dict[str, Any]] = []
        with self._file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if event_type and entry.get("event_type") != event_type:
                    continue
                if level and entry.get("level") != level:
                    continue

                entries.append(entry)
                if len(entries) >= limit:
                    break

        return entries

    @property
    def entry_count(self) -> int:
        return self._entry_count

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
