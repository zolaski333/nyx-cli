"""
Nyx — MCP (Model Context Protocol) client.

Connects to MCP servers via stdio or SSE and exposes their tools
as callable functions the agent can use.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any, Callable

from nyx.providers.base import ToolDefinition

# ---------------------------------------------------------------------------
# Helpers: minimal JSON-RPC over stdio
# ---------------------------------------------------------------------------


def _read_line(stream) -> str:
    """Read a single JSON line from the subprocess stdout."""
    line = stream.readline()
    if not line:
        raise ConnectionError("MCP server closed the connection.")
    return line.decode("utf-8", errors="replace").strip()


def _write_line(stream, data: dict) -> None:
    line = json.dumps(data, ensure_ascii=False) + "\n"
    stream.write(line.encode("utf-8"))
    stream.flush()


# ---------------------------------------------------------------------------
# MCP tool wrapper
# ---------------------------------------------------------------------------


@dataclass
class MCPTool:
    """A tool exposed by an MCP server."""
    server_name: str
    name: str
    description: str
    input_schema: dict[str, Any]
    _call_fn: Callable[[dict[str, Any]], str] | None = None

    def to_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=f"mcp_{self.server_name}_{self.name}",
            description=self.description,
            parameters=self.input_schema,
        )

    def call(self, arguments: dict[str, Any]) -> str:
        if self._call_fn:
            return self._call_fn(arguments)
        return f"[MCP:{self.server_name}] tool '{self.name}' not connected."


# ---------------------------------------------------------------------------
# MCP session (single server)
# ---------------------------------------------------------------------------


class MCPSession:
    """Manages a single MCP server connection via stdio JSON-RPC."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name = name
        self.config = config
        self._proc: subprocess.Popen | None = None
        self._req_id = 0

    def connect(self) -> list[MCPTool]:
        """Start the server subprocess, run initialize & list_tools."""
        cmd = self.config.get("command", "")
        args = self.config.get("args", [])
        env = self.config.get("env", {})
        if not cmd:
            raise RuntimeError(f"MCP server '{self.name}': no 'command' in config.")

        # Inherit parent environment and overlay MCP-specific env vars
        import os as _os
        merged_env = dict(_os.environ)
        if env:
            merged_env.update(env)
        popen_kwargs = {}
        if _os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        self._proc = subprocess.Popen(
            [cmd, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=merged_env,
            **popen_kwargs,
        )
        if not self._proc.stdin or not self._proc.stdout:
            raise RuntimeError(f"MCP server '{self.name}': failed to open pipes.")

        # 1. Initialize
        self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "nyx", "version": "0.2.0"},
        })
        # Send initialized notification
        self._notify("notifications/initialized", {})

        # 2. List tools
        tools_resp = self._request("tools/list", {})
        tools_raw = tools_resp.get("tools", [])

        tools: list[MCPTool] = []
        for t in tools_raw:
            tool = MCPTool(
                server_name=self.name,
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {"type": "object", "properties": {}}),
            )
            tool._call_fn = lambda args, t_name=t["name"]: self._call_tool(t_name, args)
            tools.append(tool)

        return tools

    def _request(self, method: str, params: dict) -> dict:
        self._req_id += 1
        req = {
            "jsonrpc": "2.0",
            "id": self._req_id,
            "method": method,
            "params": params,
        }
        assert self._proc and self._proc.stdin and self._proc.stdout
        _write_line(self._proc.stdin, req)
        while True:
            resp = _read_line(self._proc.stdout)
            if not resp.strip().startswith("{"):
                continue
            try:
                parsed = json.loads(resp)
            except json.JSONDecodeError:
                continue
            if parsed.get("id") == self._req_id:
                if "error" in parsed and parsed["error"]:
                    raise RuntimeError(f"MCP error: {parsed['error']}")
                return parsed.get("result", {})

    def _notify(self, method: str, params: dict) -> None:
        notif = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        assert self._proc and self._proc.stdin
        _write_line(self._proc.stdin, notif)

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        content = result.get("content", [])
        parts = []
        for c in content:
            if c.get("type") == "text":
                parts.append(c["text"])
            elif c.get("type") == "resource":
                parts.append(str(c.get("resource", {})))
        return "\n".join(parts) if parts else json.dumps(result, ensure_ascii=False)

    def close(self) -> None:
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None


# ---------------------------------------------------------------------------
# MCP manager (multiple servers)
# ---------------------------------------------------------------------------


class MCPManager:
    """Manages connections to multiple MCP servers."""

    def __init__(self) -> None:
        self._sessions: dict[str, MCPSession] = {}
        self.tools: list[MCPTool] = []
        self._progress_callback: Callable[[str, int, int], None] | None = None

    def set_progress_callback(self, callback: Callable[[str, int, int], None] | None) -> None:
        """Set a callback for progress updates: (label, current, total)."""
        self._progress_callback = callback

    def connect_all(self, servers_config: dict[str, dict[str, Any]]) -> list[MCPTool]:
        """Connect to all configured MCP servers and collect their tools."""
        all_tools: list[MCPTool] = []
        enabled_servers = [(name, cfg) for name, cfg in servers_config.items() if cfg.get("enabled", True)]
        total = len(enabled_servers)

        for i, (name, cfg) in enumerate(enabled_servers):
            if self._progress_callback:
                self._progress_callback(f"MCP '{name}'", i, total)
            try:
                session = MCPSession(name, cfg)
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

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Find the right MCP tool and call it."""
        for t in self.tools:
            fqn = f"mcp_{t.server_name}_{t.name}"
            if fqn == name:
                return t.call(arguments)
        return f"[MCP] Unknown tool: {name}"

    def close_all(self) -> None:
        for session in self._sessions.values():
            if session is not None:
                session.close()
        self._sessions.clear()
        self.tools.clear()
