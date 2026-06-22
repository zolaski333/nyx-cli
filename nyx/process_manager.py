"""Small persistent process manager for long-running tool commands."""
from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO


@dataclass
class ManagedProcess:
    process_id: str
    command: str
    cwd: str
    started_at: float
    process: subprocess.Popen[str]
    output: "queue.Queue[tuple[str, str]]" = field(default_factory=queue.Queue)


class ProcessManager:
    """Track long-running subprocesses and expose non-blocking output reads."""

    def __init__(self) -> None:
        self._processes: dict[str, ManagedProcess] = {}
        self._lock = threading.Lock()

    def start(
        self,
        command: str,
        *,
        cwd: str | Path,
        env: dict[str, str] | None = None,
        shell: bool = True,
    ) -> ManagedProcess:
        process_id = uuid.uuid4().hex[:12]
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            shell=shell,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        managed = ManagedProcess(
            process_id=process_id,
            command=command,
            cwd=str(cwd),
            started_at=time.time(),
            process=proc,
        )
        self._start_reader(managed, "stdout", proc.stdout)
        self._start_reader(managed, "stderr", proc.stderr)
        with self._lock:
            self._processes[process_id] = managed
        return managed

    def get(self, process_id: str) -> ManagedProcess | None:
        with self._lock:
            return self._processes.get(process_id)

    def read(self, process_id: str, *, max_lines: int = 200) -> tuple[ManagedProcess | None, list[tuple[str, str]]]:
        managed = self.get(process_id)
        if not managed:
            return None, []
        lines: list[tuple[str, str]] = []
        for _ in range(max(1, max_lines)):
            try:
                lines.append(managed.output.get_nowait())
            except queue.Empty:
                break
        return managed, lines

    def write(self, process_id: str, text: str) -> tuple[bool, str]:
        managed = self.get(process_id)
        if not managed:
            return False, f"Unknown process_id: {process_id}"
        if managed.process.poll() is not None:
            return False, f"Process {process_id} has already exited with code {managed.process.returncode}."
        if managed.process.stdin is None:
            return False, f"Process {process_id} has no stdin pipe."
        managed.process.stdin.write(text)
        managed.process.stdin.flush()
        return True, f"Wrote {len(text)} chars to process {process_id}."

    def stop(self, process_id: str) -> tuple[bool, str]:
        managed = self.get(process_id)
        if not managed:
            return False, f"Unknown process_id: {process_id}"
        proc = managed.process
        if proc.poll() is not None:
            self._forget(process_id)
            return True, f"Process {process_id} already exited with code {proc.returncode}."
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, text=True, timeout=10)
        else:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        self._forget(process_id)
        return True, f"Stopped process {process_id}."

    def _forget(self, process_id: str) -> None:
        with self._lock:
            self._processes.pop(process_id, None)

    def _start_reader(
        self,
        managed: ManagedProcess,
        stream_name: str,
        stream: IO[str] | None,
    ) -> None:
        if stream is None:
            return

        def _reader() -> None:
            try:
                for line in stream:
                    managed.output.put((stream_name, str(line).rstrip("\r\n")))
            finally:
                managed.output.put((stream_name, f"[{stream_name} closed]"))

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()


PROCESS_MANAGER = ProcessManager()
