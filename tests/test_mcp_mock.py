"""Tests for MCP client with a dummy JSON-RPC server over stdio."""
from __future__ import annotations

import subprocess
import sys

import pytest

from nyx.mcp_client import MCPManager, MCPSession, MCPTool


# ---------------------------------------------------------------------------
# Dummy JSON-RPC MCP server script
# ---------------------------------------------------------------------------

DUMMY_MCP_SERVER = """
import json
import sys

def read_line():
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    return json.loads(line.strip())

def write_line(data):
    sys.stdout.write(json.dumps(data, ensure_ascii=False) + "\\n")
    sys.stdout.flush()

# Handle initialize
req = read_line()
assert req["method"] == "initialize"
write_line({"jsonrpc": "2.0", "id": req["id"], "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {"tools": {}},
    "serverInfo": {"name": "dummy-mcp", "version": "1.0.0"},
}})

# Handle initialized notification
req = read_line()
assert req["method"] == "notifications/initialized"

# Handle tools/list
req = read_line()
assert req["method"] == "tools/list"
write_line({"jsonrpc": "2.0", "id": req["id"], "result": {
    "tools": [
        {
            "name": "echo",
            "description": "Echo back the input",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Message to echo"}
                },
                "required": ["message"],
            },
        },
        {
            "name": "add",
            "description": "Add two numbers",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        },
    ]
}})

# Handle tool calls in a loop
while True:
    req = read_line()
    method = req["method"]
    params = req.get("params", {})

    if method == "tools/call":
        tool_name = params.get("name", "")
        args = params.get("arguments", {})

        if tool_name == "echo":
            result = {"content": [{"type": "text", "text": args.get("message", "")}]}
        elif tool_name == "add":
            a = args.get("a", 0)
            b = args.get("b", 0)
            result = {"content": [{"type": "text", "text": str(a + b)}]}
        else:
            result = {"content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}]}

        write_line({"jsonrpc": "2.0", "id": req["id"], "result": result})
    elif method == "notifications/initialized":
        pass  # ignore
    else:
        write_line({"jsonrpc": "2.0", "id": req["id"], "error": {
            "code": -32601, "message": f"Method not found: {method}"
        }})
"""

NO_RESPONSE_MCP_SERVER = """
import time
time.sleep(10)
"""

LOUD_MCP_SERVER = """
import json
import sys

def read_line():
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    return json.loads(line.strip())

def write_line(data):
    sys.stdout.write(json.dumps(data, ensure_ascii=False) + "\\n")
    sys.stdout.flush()

sys.stderr.write("x" * 200000)
sys.stderr.flush()
req = read_line()
write_line({"jsonrpc": "2.0", "id": req["id"], "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {"tools": {}},
}})
req = read_line()
req = read_line()
write_line({"jsonrpc": "2.0", "id": req["id"], "result": {"tools": []}})
"""

INVALID_TOOLS_MCP_SERVER = """
import json
import sys

def read_line():
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    return json.loads(line.strip())

def write_line(data):
    sys.stdout.write(json.dumps(data, ensure_ascii=False) + "\\n")
    sys.stdout.flush()

req = read_line()
write_line({"jsonrpc": "2.0", "id": req["id"], "result": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}}})
read_line()
req = read_line()
write_line({"jsonrpc": "2.0", "id": req["id"], "result": {"tools": [
    {"name": "../bad", "description": "bad", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "ok_tool", "description": "ok", "inputSchema": {"type": "array"}}
]}})
while True:
    req = read_line()
    write_line({"jsonrpc": "2.0", "id": req["id"], "result": {"content": [{"type": "text", "text": "y" * 100}]}})
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_mcp_server():
    """Start a dummy MCP server subprocess using the embedded script."""
    proc = subprocess.Popen(
        [sys.executable, "-c", DUMMY_MCP_SERVER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    yield proc
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()


# =========================================================================
# MCP Tool Tests
# =========================================================================


class TestMCPTool:
    """Test the MCPTool wrapper."""

    def test_to_definition(self):
        """Should convert to ToolDefinition with prefixed name."""
        tool = MCPTool(
            server_name="test-server",
            name="echo",
            description="Echo input",
            input_schema={"type": "object", "properties": {"msg": {"type": "string"}}},
        )
        td = tool.to_definition()
        assert td.name == "mcp_test-server_echo"
        assert td.description == "Echo input"
        assert td.parameters["properties"]["msg"]["type"] == "string"

    def test_call_without_fn(self):
        """Should return error message if no call_fn set."""
        tool = MCPTool(server_name="s", name="t", description="", input_schema={})
        result = tool.call({"arg": 1})
        assert "not connected" in result

    def test_call_with_fn(self):
        """Should call the provided call_fn."""
        def fake_call(args):
            return f"called with {args}"

        tool = MCPTool(server_name="s", name="t", description="", input_schema={})
        tool._call_fn = fake_call
        result = tool.call({"key": "value"})
        assert result == "called with {'key': 'value'}"


# =========================================================================
# MCP Session Tests (with dummy server)
# =========================================================================


class TestMCPSession:
    """Test MCPSession with a dummy JSON-RPC server."""

    def test_connect_and_list_tools(self, dummy_mcp_server):
        """Should connect and discover tools from the dummy server."""
        session = MCPSession("dummy", {
            "command": sys.executable,
            "args": ["-c", DUMMY_MCP_SERVER],
        })
        # Override the process with our fixture
        session._proc = dummy_mcp_server
        session._req_id = 0

        tools = session.connect()
        assert len(tools) == 2
        assert tools[0].name == "echo"
        assert tools[1].name == "add"

    def test_call_echo_tool(self, dummy_mcp_server):
        """Should call the echo tool and get the message back."""
        session = MCPSession("dummy", {
            "command": sys.executable,
            "args": ["-c", DUMMY_MCP_SERVER],
        })
        session._proc = dummy_mcp_server
        session._req_id = 0

        tools = session.connect()
        echo_tool = next(t for t in tools if t.name == "echo")

        result = echo_tool.call({"message": "Hello MCP!"})
        assert result == "Hello MCP!"

    def test_call_add_tool(self, dummy_mcp_server):
        """Should call the add tool and get the sum."""
        session = MCPSession("dummy", {
            "command": sys.executable,
            "args": ["-c", DUMMY_MCP_SERVER],
        })
        session._proc = dummy_mcp_server
        session._req_id = 0

        tools = session.connect()
        add_tool = next(t for t in tools if t.name == "add")

        result = add_tool.call({"a": 3, "b": 4})
        assert result == "7"

    def test_close_terminates_process(self, dummy_mcp_server):
        """Should terminate the subprocess on close."""
        session = MCPSession("dummy", {})
        session._proc = dummy_mcp_server
        session.close()
        assert session._proc is None
        assert dummy_mcp_server.poll() is not None

    def test_mcp_resilience_to_noise(self):
        """Should skip non-JSON debug lines and notifications to find the correct response."""
        import io
        session = MCPSession("dummy", {})
        session._req_id = 42
        
        # We simulate a stdout stream that contains:
        # 1. Non-JSON debug message
        # 2. Malformed JSON
        # 3. A JSON-RPC response with a different ID (e.g., 99)
        # 4. A JSON-RPC response with the correct ID (43)
        stdout_data = (
            "DEBUG: some random logging\n"
            "{invalid json\n"
            '{"jsonrpc": "2.0", "id": 99, "result": {"ignored": true}}\n'
            '{"jsonrpc": "2.0", "id": 43, "result": {"success": true}}\n'
        ).encode("utf-8")
        
        class MockProc:
            def __init__(self):
                self.stdin = io.BytesIO()
                self.stdout = io.BytesIO(stdout_data)
        
        session._proc = MockProc()
        
        # _request will increment self._req_id to 43, write request, then read responses
        res = session._request("test_method", {})
        assert res == {"success": True}

    def test_mcp_does_not_inherit_secret_env_by_default(self, monkeypatch):
        """MCP servers should not receive parent API keys unless explicitly passed."""
        import os
        from nyx.mcp_client import SAFE_INHERITED_ENV

        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
        monkeypatch.setenv("PATH", os.environ.get("PATH", ""))

        inherited = {k: v for k, v in os.environ.items() if k in SAFE_INHERITED_ENV}
        assert "OPENAI_API_KEY" not in inherited

    def test_connect_timeout_does_not_hang(self):
        """A server that never responds should fail within the configured timeout."""
        session = MCPSession("slow", {
            "command": sys.executable,
            "args": ["-c", NO_RESPONSE_MCP_SERVER],
            "connect_timeout": 0.1,
            "request_timeout": 0.1,
        })
        with pytest.raises(TimeoutError):
            session.connect()
        session.close()

    def test_stderr_is_drained_during_connect(self):
        """Verbose stderr should not block a server from completing initialization."""
        session = MCPSession("loud", {
            "command": sys.executable,
            "args": ["-c", LOUD_MCP_SERVER],
            "connect_timeout": 2,
        })
        tools = session.connect()
        assert tools == []
        assert session.status == "connected"
        session.close()

    def test_invalid_tools_are_skipped_and_output_is_truncated(self):
        """Invalid tool names/schemas should be handled defensively."""
        session = MCPSession("validated", {
            "command": sys.executable,
            "args": ["-c", INVALID_TOOLS_MCP_SERVER],
            "max_response_chars": 10,
        })
        tools = session.connect()
        assert [t.name for t in tools] == ["ok_tool"]
        assert tools[0].input_schema == {"type": "object", "properties": {}}
        result = tools[0].call_result({})
        assert result.ok
        assert result.output == "y" * 10
        assert result.truncated is True
        session.close()

    def test_invalid_cwd_is_rejected(self):
        """Invalid working directories should fail before process start."""
        session = MCPSession("badcwd", {
            "command": sys.executable,
            "cwd": "Z:/definitely/not/a/real/path",
        })
        with pytest.raises(RuntimeError):
            session._build_process_invocation()

    def test_windows_command_shim_is_resolved(self, monkeypatch):
        """Windows MCP commands like npx should resolve to command shims."""
        monkeypatch.setattr("nyx.mcp_client.os.name", "nt")
        monkeypatch.setattr(
            "nyx.mcp_client.shutil.which",
            lambda name, path=None: "C:\\node\\npx.cmd" if name == "npx.cmd" else None,
        )
        session = MCPSession("node", {
            "command": "npx",
            "args": ["-y", "some-server"],
        })

        argv, _env, _cwd = session._build_process_invocation()

        assert argv == ["C:\\node\\npx.cmd", "-y", "some-server"]

    def test_docker_sandbox_invocation_is_built_without_secret_env(self, monkeypatch, tmp_path):
        """Docker sandbox mode should build a stdio-friendly isolated invocation."""
        monkeypatch.setattr("nyx.mcp_client.shutil.which", lambda name: "docker" if name == "docker" else None)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
        session = MCPSession("boxed", {
            "command": "python",
            "args": ["server.py"],
            "env": {"TOKEN": "explicit"},
            "pass_env": ["PATH"],
            "sandbox": {
                "enabled": True,
                "use_docker": True,
                "project_dir": str(tmp_path),
                "docker_image": "python:3.11-slim",
                "network": "none",
                "read_only": True,
            },
        })

        argv, env, cwd = session._build_process_invocation()

        assert argv[:4] == ["docker", "run", "--rm", "-i"]
        assert "--network" in argv
        assert "none" in argv
        assert "--read-only" in argv
        assert "python:3.11-slim" in argv
        assert "python" in argv[-1]
        assert "server.py" in argv[-1]
        assert "OPENAI_API_KEY" not in env
        assert cwd is None



# =========================================================================
# MCP Manager Tests
# =========================================================================


class TestMCPManager:
    """Test the MCP manager."""

    def test_connect_all_empty(self):
        """Empty config should yield no tools."""
        manager = MCPManager()
        tools = manager.connect_all({})
        assert len(tools) == 0
        assert len(manager.tools) == 0

    def test_connect_all_disabled(self):
        """Disabled servers should be skipped."""
        manager = MCPManager()
        tools = manager.connect_all({
            "disabled-server": {
                "enabled": False,
                "command": "nonexistent",
            },
        })
        assert len(tools) == 0

    def test_manager_injects_global_sandbox_config(self, monkeypatch, tmp_path):
        """Global MCP sandbox settings should be applied to server configs."""
        captured = {}

        class FakeSession:
            def __init__(self, name, config):
                captured["name"] = name
                captured["config"] = config

            def connect(self):
                return []

        monkeypatch.setattr("nyx.mcp_client.MCPSession", FakeSession)
        manager = MCPManager(
            sandbox_enabled=True,
            sandbox_docker_image="python:3.11-slim",
            sandbox_project_dir=str(tmp_path),
            sandbox_read_only=True,
        )
        manager.connect_all({"boxed": {"command": "python"}})

        assert captured["name"] == "boxed"
        sandbox = captured["config"]["sandbox"]
        assert sandbox["enabled"] is True
        assert sandbox["use_docker"] is True
        assert sandbox["docker_image"] == "python:3.11-slim"
        assert sandbox["read_only"] is True
        assert sandbox["project_dir"] == str(tmp_path)

    def test_get_tool_definitions(self):
        """Should return ToolDefinition list."""
        manager = MCPManager()
        tool = MCPTool(
            server_name="s", name="t", description="desc",
            input_schema={"type": "object", "properties": {}},
        )
        manager.tools = [tool]
        defs = manager.get_tool_definitions()
        assert len(defs) == 1
        assert defs[0].name == "mcp_s_t"

    def test_call_tool(self):
        """Should find and call the right tool."""
        manager = MCPManager()
        results = []

        def fake_call(args):
            results.append(args)
            return "ok"

        tool = MCPTool(
            server_name="my-server", name="my-tool", description="",
            input_schema={},
        )
        tool._call_fn = fake_call
        manager.tools = [tool]

        result = manager.call_tool("mcp_my-server_my-tool", {"arg": 1})
        assert result == "ok"
        assert results == [{"arg": 1}]

    def test_call_unknown_tool(self):
        """Unknown tool should return error message."""
        manager = MCPManager()
        result = manager.call_tool("unknown_tool", {})
        assert "Unknown tool" in result

    def test_close_all(self):
        """Should close all sessions."""
        manager = MCPManager()
        manager._sessions = {"s1": None, "s2": None}  # Will be skipped gracefully
        manager.close_all()  # Should not raise
        assert len(manager._sessions) == 0
