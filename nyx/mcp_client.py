"""Robust stdio MCP client.

Nyx currently supports MCP over stdio. Servers are local trusted processes, but
the client still defends itself against hangs, noisy output, oversized
responses, stderr backpressure and dead child processes.
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from nyx.providers.base import ToolDefinition


SAFE_INHERITED_ENV = {
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "WINDIR",
    "HOME",
    "USERPROFILE",
    "TMP",
    "TEMP",
    "TMPDIR",
}

MCP_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
DEFAULT_REQUEST_TIMEOUT = 30.0
DEFAULT_CONNECT_TIMEOUT = 30.0
DEFAULT_MAX_RESPONSE_CHARS = 20000
DEFAULT_MCP_DOCKER_IMAGE = "python:3.11-slim-buster"


def _write_line(stream: Any, data: dict[str, Any]) -> None:
    line = json.dumps(data, ensure_ascii=False) + "\n"
    stream.write(line.encode("utf-8"))
    stream.flush()


def _decode_line(line: bytes | str) -> str:
    if isinstance(line, bytes):
        return line.decode("utf-8", errors="replace").strip()
    return str(line).strip()


def _posix_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def _shell_join(argv: list[str]) -> str:
    return " ".join(_posix_quote(part) for part in argv)


@dataclass
class MCPResult:
    """Structured result from an MCP tool call."""

    server_name: str
    tool_name: str
    output: str = ""
    status: str = "completed"
    error: str | None = None
    error_type: str | None = None
    duration_seconds: float = 0.0
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and self.status == "completed"

    def to_text(self) -> str:
        if self.ok:
            suffix = "\n[MCP output truncated]" if self.truncated else ""
            return f"{self.output}{suffix}"
        return f"[MCP:{self.server_name}/{self.tool_name}] {self.status} ({self.error_type}): {self.error}"


@dataclass
class MCPTool:
    """A tool exposed by an MCP server."""

    server_name: str
    name: str
    description: str
    input_schema: dict[str, Any]
    _call_fn: Callable[[dict[str, Any]], str] | None = None
    _call_result_fn: Callable[[dict[str, Any]], MCPResult] | None = None

    def to_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=f"mcp_{self.server_name}_{self.name}",
            description=self.description,
            parameters=self.input_schema,
        )

    def call_result(self, arguments: dict[str, Any]) -> MCPResult:
        if self._call_result_fn:
            return self._call_result_fn(arguments)
        if self._call_fn:
            started = time.monotonic()
            try:
                return MCPResult(
                    server_name=self.server_name,
                    tool_name=self.name,
                    output=self._call_fn(arguments),
                    duration_seconds=time.monotonic() - started,
                )
            except Exception as exc:
                return MCPResult(
                    server_name=self.server_name,
                    tool_name=self.name,
                    status="failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                    duration_seconds=time.monotonic() - started,
                )
        return MCPResult(
            server_name=self.server_name,
            tool_name=self.name,
            status="failed",
            error=f"tool '{self.name}' not connected.",
            error_type="not_connected",
        )

    def call(self, arguments: dict[str, Any]) -> str:
        return self.call_result(arguments).to_text()


class MCPSession:
    """Manages a single MCP server connection via stdio JSON-RPC."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name = name
        self.config = config
        self._proc: subprocess.Popen[bytes] | Any | None = None
        self._req_id = 0
        self.status = "disconnected"
        self.last_error: str | None = None
        self._request_lock = threading.RLock()
        self._io_lock = threading.Lock()
        self._stdout_queue: queue.Queue[str] | None = None
        self._stderr_tail: deque[str] = deque(maxlen=50)
        self._reader_threads: list[threading.Thread] = []
        self._tools: list[MCPTool] = []
        self.request_timeout = float(config.get("request_timeout", DEFAULT_REQUEST_TIMEOUT))
        self.connect_timeout = float(config.get("connect_timeout", self.request_timeout or DEFAULT_CONNECT_TIMEOUT))
        self.max_response_chars = int(config.get("max_response_chars", DEFAULT_MAX_RESPONSE_CHARS))
        self.restart_on_failure = bool(config.get("restart_on_failure", True))

    def connect(self) -> list[MCPTool]:
        """Start the server subprocess, initialize it and list tools."""
        self._validate_server_name()
        if not self._is_process_running():
            self._start_process()
        self._ensure_readers()
        self.status = "connecting"

        try:
            self._request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "nyx", "version": "0.2.0"},
                },
                timeout=self.connect_timeout,
            )
            self._notify("notifications/initialized", {})
            tools_resp = self._request("tools/list", {}, timeout=self.connect_timeout)
            tools = self._parse_tools(tools_resp.get("tools", []))
            self._tools = tools
            self.status = "connected"
            self.last_error = None
            return tools
        except Exception as exc:
            self.status = "failed"
            self.last_error = str(exc)
            self.close()
            raise

    def _validate_server_name(self) -> None:
        if not MCP_NAME_RE.match(self.name):
            raise RuntimeError(f"MCP server '{self.name}': invalid server name.")

    def _start_process(self) -> None:
        argv, merged_env, cwd = self._build_process_invocation()

        popen_kwargs: dict[str, Any] = {}
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=merged_env,
            cwd=cwd,
            **popen_kwargs,
        )
        if not self._proc.stdin or not self._proc.stdout:
            raise RuntimeError(f"MCP server '{self.name}': failed to open pipes.")

    def _build_process_invocation(self) -> tuple[list[str], dict[str, str], str | None]:
        cmd = self.config.get("command", "")
        args = self.config.get("args", [])
        env = self.config.get("env", {})
        cwd = self.config.get("cwd")
        if not cmd:
            raise RuntimeError(f"MCP server '{self.name}': no 'command' in config.")
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise RuntimeError(f"MCP server '{self.name}': 'args' must be a list of strings.")
        if env and not isinstance(env, dict):
            raise RuntimeError(f"MCP server '{self.name}': 'env' must be an object.")

        pass_env = set(self.config.get("pass_env", []))
        inherited = SAFE_INHERITED_ENV | pass_env
        merged_env = {k: v for k, v in os.environ.items() if k in inherited}
        if env:
            merged_env.update({str(k): str(v) for k, v in env.items()})

        local_cwd = self._resolve_cwd(cwd)
        sandbox_cfg = self.config.get("sandbox", {})
        if sandbox_cfg is True:
            sandbox_cfg = {"enabled": True, "use_docker": True}
        if not isinstance(sandbox_cfg, dict):
            sandbox_cfg = {}
        sandbox_enabled = bool(sandbox_cfg.get("enabled", False))
        use_docker = bool(sandbox_cfg.get("use_docker", sandbox_enabled))
        if sandbox_enabled and use_docker:
            return self._build_docker_invocation([str(cmd), *args], merged_env, local_cwd, sandbox_cfg)

        resolved_cmd = self._resolve_command(str(cmd), merged_env)
        return [resolved_cmd, *args], merged_env, local_cwd

    def _resolve_command(self, cmd: str, env: dict[str, str]) -> str:
        """Resolve command shims like npx.cmd on Windows without shell=True."""
        if os.name != "nt":
            return cmd
        if any(sep in cmd for sep in (os.sep, os.altsep) if sep):
            return cmd

        path = env.get("PATH") or os.environ.get("PATH")
        found = shutil.which(cmd, path=path)
        if found:
            return found

        base, ext = os.path.splitext(cmd)
        if ext:
            raise RuntimeError(f"MCP server '{self.name}': command not found: {cmd}")

        for suffix in (".cmd", ".bat", ".exe", ".ps1"):
            found = shutil.which(base + suffix, path=path)
            if found:
                return found
        raise RuntimeError(f"MCP server '{self.name}': command not found: {cmd}")

    def _resolve_cwd(self, cwd: Any) -> str | None:
        if not cwd:
            return None
        if not isinstance(cwd, str):
            raise RuntimeError(f"MCP server '{self.name}': 'cwd' must be a string.")
        resolved = Path(cwd).expanduser().resolve()
        if not resolved.is_dir():
            raise RuntimeError(f"MCP server '{self.name}': cwd does not exist: {resolved}")
        return str(resolved)

    def _build_docker_invocation(
        self,
        target_argv: list[str],
        merged_env: dict[str, str],
        local_cwd: str | None,
        sandbox_cfg: dict[str, Any],
    ) -> tuple[list[str], dict[str, str], str | None]:
        docker_bin = shutil.which("docker") or shutil.which("podman")
        if not docker_bin:
            raise RuntimeError(f"MCP server '{self.name}': sandbox.use_docker is enabled but docker/podman was not found.")

        workspace = sandbox_cfg.get("project_dir") or local_cwd or os.getcwd()
        workspace_path = Path(str(workspace)).expanduser().resolve()
        if not workspace_path.is_dir():
            raise RuntimeError(f"MCP server '{self.name}': sandbox project_dir does not exist: {workspace_path}")

        image = str(sandbox_cfg.get("docker_image") or DEFAULT_MCP_DOCKER_IMAGE)
        network = str(sandbox_cfg.get("network", "none"))
        read_only = bool(sandbox_cfg.get("read_only", False))
        workdir = str(sandbox_cfg.get("container_workdir", "/workspace"))
        command = _shell_join(target_argv)
        argv = [
            docker_bin,
            "run",
            "--rm",
            "-i",
            "--network",
            network,
            "-v",
            f"{workspace_path}:{workdir}",
            "-w",
            workdir,
        ]
        if read_only:
            argv.append("--read-only")
            argv.extend(["--tmpfs", "/tmp"])
        for key, value in merged_env.items():
            if key in self.config.get("env", {}) or key in set(self.config.get("pass_env", [])):
                argv.extend(["-e", f"{key}={value}"])
        argv.extend([image, "sh", "-lc", command])
        # Host secrets stay out of the container except explicit env/pass_env.
        container_env = {k: v for k, v in os.environ.items() if k in SAFE_INHERITED_ENV}
        return argv, container_env, None

    def _ensure_readers(self) -> None:
        if self._stdout_queue is not None:
            return
        if not self._proc or not getattr(self._proc, "stdout", None):
            raise RuntimeError(f"MCP server '{self.name}': stdout pipe is not available.")

        self._stdout_queue = queue.Queue()
        stdout_thread = threading.Thread(
            target=self._drain_stdout,
            name=f"nyx-mcp-{self.name}-stdout",
            daemon=True,
        )
        stdout_thread.start()
        self._reader_threads.append(stdout_thread)

        if getattr(self._proc, "stderr", None):
            stderr_thread = threading.Thread(
                target=self._drain_stderr,
                name=f"nyx-mcp-{self.name}-stderr",
                daemon=True,
            )
            stderr_thread.start()
            self._reader_threads.append(stderr_thread)

    def _drain_stdout(self) -> None:
        assert self._proc and self._proc.stdout and self._stdout_queue
        stdout = self._proc.stdout
        stdout_queue = self._stdout_queue
        try:
            while True:
                line = stdout.readline()
                if not line:
                    break
                stdout_queue.put(_decode_line(line))
        finally:
            stdout_queue.put("")

    def _drain_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        stderr = self._proc.stderr
        while True:
            chunk = stderr.read(4096)
            if not chunk:
                break
            self._stderr_tail.append(_decode_line(chunk))

    def _is_process_running(self) -> bool:
        if not self._proc:
            return False
        poll = getattr(self._proc, "poll", None)
        return True if poll is None else poll() is None

    def _request(self, method: str, params: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        with self._request_lock:
            return self._request_locked(method, params, timeout=timeout)

    def _request_locked(self, method: str, params: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        if not self._proc or not getattr(self._proc, "stdin", None):
            raise ConnectionError(f"MCP server '{self.name}' is not connected.")
        if not self._is_process_running():
            raise ConnectionError(f"MCP server '{self.name}' is not running.")
        self._ensure_readers()

        self._req_id += 1
        req_id = self._req_id
        req = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        try:
            with self._io_lock:
                _write_line(self._proc.stdin, req)
        except Exception as exc:
            self.status = "dead"
            self.last_error = str(exc)
            raise ConnectionError(f"MCP server '{self.name}' write failed: {exc}") from exc

        deadline = time.monotonic() + (timeout if timeout is not None else self.request_timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.status = "timeout"
                raise TimeoutError(f"MCP server '{self.name}' timed out waiting for '{method}'.")
            line = self._read_stdout_line(timeout=remaining)
            if not line or not line.strip().startswith("{"):
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if parsed.get("id") != req_id:
                continue
            if parsed.get("jsonrpc") not in (None, "2.0"):
                raise RuntimeError(f"MCP server '{self.name}' returned invalid JSON-RPC version.")
            if "error" in parsed and parsed["error"]:
                raise RuntimeError(f"MCP error: {parsed['error']}")
            result = parsed.get("result", {})
            if not isinstance(result, dict):
                raise RuntimeError(f"MCP server '{self.name}' returned non-object result.")
            return result

    def _read_stdout_line(self, timeout: float) -> str:
        assert self._stdout_queue is not None
        try:
            line = self._stdout_queue.get(timeout=max(0.001, timeout))
        except queue.Empty as exc:
            raise TimeoutError(f"MCP server '{self.name}' did not produce a response.") from exc
        if line == "":
            stderr = "\n".join(self._stderr_tail)
            suffix = f" Stderr tail:\n{stderr}" if stderr else ""
            raise ConnectionError(f"MCP server '{self.name}' closed stdout.{suffix}")
        return line

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        notif = {"jsonrpc": "2.0", "method": method, "params": params}
        if not self._proc or not getattr(self._proc, "stdin", None):
            raise ConnectionError(f"MCP server '{self.name}' is not connected.")
        with self._io_lock:
            _write_line(self._proc.stdin, notif)

    def _parse_tools(self, tools_raw: Any) -> list[MCPTool]:
        if not isinstance(tools_raw, list):
            raise RuntimeError(f"MCP server '{self.name}' returned invalid tools/list payload.")
        tools: list[MCPTool] = []
        seen: set[str] = set()
        for item in tools_raw:
            if not isinstance(item, dict):
                continue
            tool_name = item.get("name", "")
            if not isinstance(tool_name, str) or not MCP_NAME_RE.match(tool_name):
                print(f"  ! MCP '{self.name}': skipping invalid tool name {tool_name!r}")
                continue
            if tool_name in seen:
                print(f"  ! MCP '{self.name}': skipping duplicate tool '{tool_name}'")
                continue
            seen.add(tool_name)
            schema = item.get("inputSchema", {"type": "object", "properties": {}})
            if not isinstance(schema, dict) or schema.get("type", "object") != "object":
                schema = {"type": "object", "properties": {}}
            tool = MCPTool(
                server_name=self.name,
                name=tool_name,
                description=str(item.get("description", "")),
                input_schema=schema,
            )
            tool._call_result_fn = lambda args, t_name=tool_name: self._call_tool_result(t_name, args)
            tool._call_fn = lambda args, t_name=tool_name: self._call_tool_result(t_name, args).to_text()
            tools.append(tool)
        return tools

    def _call_tool_result(self, name: str, arguments: dict[str, Any]) -> MCPResult:
        started = time.monotonic()
        try:
            try:
                result = self._request("tools/call", {"name": name, "arguments": arguments})
            except (ConnectionError, TimeoutError):
                if not self.restart_on_failure:
                    raise
                self.restart()
                result = self._request("tools/call", {"name": name, "arguments": arguments})
            output, truncated = self._format_tool_result(result)
            return MCPResult(
                server_name=self.name,
                tool_name=name,
                output=output,
                status="completed",
                duration_seconds=time.monotonic() - started,
                truncated=truncated,
            )
        except Exception as exc:
            return MCPResult(
                server_name=self.name,
                tool_name=name,
                status="failed",
                error=str(exc),
                error_type=type(exc).__name__,
                duration_seconds=time.monotonic() - started,
            )

    def _format_tool_result(self, result: dict[str, Any]) -> tuple[str, bool]:
        content = result.get("content", [])
        parts: list[str] = []
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif item.get("type") == "resource":
                    parts.append(json.dumps(item.get("resource", {}), ensure_ascii=False))
        output = "\n".join(parts) if parts else json.dumps(result, ensure_ascii=False)
        if self.max_response_chars > 0 and len(output) > self.max_response_chars:
            return output[: self.max_response_chars], True
        return output, False

    def restart(self) -> None:
        """Restart this session and refresh tool call closures."""
        old_tools = list(self._tools)
        self.close()
        new_tools = self.connect()
        by_name = {t.name: t for t in new_tools}
        for old in old_tools:
            replacement = by_name.get(old.name)
            if replacement:
                old._call_fn = replacement._call_fn
                old._call_result_fn = replacement._call_result_fn
        self._tools = old_tools or new_tools

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        self._stdout_queue = None
        self.status = "disconnected"
        if not proc:
            return
        try:
            stdin = getattr(proc, "stdin", None)
            if stdin is not None:
                try:
                    stdin.close()
                except Exception:
                    pass
            poll = getattr(proc, "poll", None)
            if poll is None or poll() is None:
                self._terminate_process_tree(proc)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._kill_process_tree(proc)
                    proc.wait(timeout=3)
        except Exception:
            pass

    def _terminate_process_tree(self, proc: subprocess.Popen[bytes] | Any) -> None:
        if os.name == "nt":
            pid = getattr(proc, "pid", None)
            if pid:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                return
        try:
            os.killpg(proc.pid, signal.SIGTERM)  # type: ignore[attr-defined]
        except Exception:
            proc.terminate()

    def _kill_process_tree(self, proc: subprocess.Popen[bytes] | Any) -> None:
        if os.name == "nt":
            pid = getattr(proc, "pid", None)
            if pid:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                return
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # type: ignore[attr-defined]
        except Exception:
            proc.kill()


class MCPManager:
    """Manages connections to multiple MCP servers."""

    def __init__(
        self,
        *,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        max_response_chars: int = DEFAULT_MAX_RESPONSE_CHARS,
        restart_on_failure: bool = True,
        sandbox_enabled: bool = False,
        sandbox_docker_image: str = DEFAULT_MCP_DOCKER_IMAGE,
        sandbox_network: str = "none",
        sandbox_read_only: bool = False,
        sandbox_project_dir: str = "",
    ) -> None:
        self._sessions: dict[str, MCPSession] = {}
        self.tools: list[MCPTool] = []
        self._progress_callback: Callable[[str, int, int], None] | None = None
        self.request_timeout = request_timeout
        self.connect_timeout = connect_timeout
        self.max_response_chars = max_response_chars
        self.restart_on_failure = restart_on_failure
        self.sandbox_enabled = sandbox_enabled
        self.sandbox_docker_image = sandbox_docker_image
        self.sandbox_network = sandbox_network
        self.sandbox_read_only = sandbox_read_only
        self.sandbox_project_dir = sandbox_project_dir

    def set_progress_callback(self, callback: Callable[[str, int, int], None] | None) -> None:
        self._progress_callback = callback

    def connect_all(self, servers_config: dict[str, dict[str, Any]]) -> list[MCPTool]:
        all_tools: list[MCPTool] = []
        enabled_servers = [(name, cfg) for name, cfg in servers_config.items() if cfg.get("enabled", True)]
        total = len(enabled_servers)

        for i, (name, cfg) in enumerate(enabled_servers):
            if self._progress_callback:
                self._progress_callback(f"MCP '{name}'", i, total)
            try:
                merged_cfg = dict(cfg)
                merged_cfg.setdefault("request_timeout", self.request_timeout)
                merged_cfg.setdefault("connect_timeout", self.connect_timeout)
                merged_cfg.setdefault("max_response_chars", self.max_response_chars)
                merged_cfg.setdefault("restart_on_failure", self.restart_on_failure)
                if "sandbox" not in merged_cfg and self.sandbox_enabled:
                    merged_cfg["sandbox"] = {
                        "enabled": True,
                        "use_docker": True,
                        "docker_image": self.sandbox_docker_image,
                        "network": self.sandbox_network,
                        "read_only": self.sandbox_read_only,
                        "project_dir": self.sandbox_project_dir,
                    }
                session = MCPSession(name, merged_cfg)
                tools = session.connect()
                self._sessions[name] = session
                all_tools.extend(tools)
                print(f"  ok MCP '{name}': {len(tools)} tool(s) loaded")
            except Exception as e:
                print(f"  x MCP '{name}': {e}")
            if self._progress_callback:
                self._progress_callback(f"MCP '{name}'", i + 1, total)
        self.tools = all_tools
        return all_tools

    def get_tool_definitions(self) -> list[ToolDefinition]:
        return [t.to_definition() for t in self.tools]

    def call_tool_result(self, name: str, arguments: dict[str, Any]) -> MCPResult:
        for t in self.tools:
            fqn = f"mcp_{t.server_name}_{t.name}"
            if fqn == name:
                return t.call_result(arguments)
        return MCPResult(server_name="", tool_name=name, status="failed", error=f"Unknown tool: {name}", error_type="unknown_tool")

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return self.call_tool_result(name, arguments).to_text()

    def close_all(self) -> None:
        for session in list(self._sessions.values()):
            if session is not None:
                session.close()
        self._sessions.clear()
        self.tools.clear()
