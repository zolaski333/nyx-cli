"""Tests for Nyx agent — context, sandbox, and tool execution."""
from __future__ import annotations

from nyx.agent import Agent, AgentContext
from nyx.config import Config
from nyx.providers.base import ToolCall


class TestAgentContext:
    """Test the AgentContext conversation management."""

    def test_add_message(self):
        ctx = AgentContext()
        ctx.add("user", "hello")
        assert len(ctx.messages) == 1
        assert ctx.messages[0]["role"] == "user"
        assert ctx.messages[0]["content"] == "hello"

    def test_add_tool_result(self):
        ctx = AgentContext()
        ctx.add_tool_result("call_1", "web_search", "results here")
        assert len(ctx.messages) == 1
        assert ctx.messages[0]["role"] == "tool"
        assert ctx.messages[0]["tool_call_id"] == "call_1"

    def test_clear_keeps_system(self):
        ctx = AgentContext()
        ctx.add("system", "You are a helpful assistant.")
        ctx.add("user", "hello")
        ctx.add("assistant", "hi")
        ctx.clear()
        assert len(ctx.messages) == 1
        assert ctx.messages[0]["role"] == "system"

    def test_max_history(self):
        ctx = AgentContext(max_history=3)
        ctx.add("system", "sys")
        ctx.add("user", "a")
        ctx.add("assistant", "b")
        ctx.add("user", "c")
        # Should keep system + last 2
        assert len(ctx.messages) <= 3

    def test_len(self):
        ctx = AgentContext()
        assert len(ctx) == 0
        ctx.add("user", "hello")
        assert len(ctx) == 1


class TestAgentSandbox:
    """Test the command sandbox security."""

    def setup_method(self):
        self.config = Config(openrouter_api_key="sk-test")
        self.agent = Agent(config=self.config)

    def test_dangerous_commands_detected(self):
        assert self.agent._is_dangerous_command("rm -rf /")
        assert self.agent._is_dangerous_command("sudo apt install")
        assert self.agent._is_dangerous_command("curl http://evil.com")
        assert self.agent._is_dangerous_command("mv file.txt /tmp")
        assert self.agent._is_dangerous_command("chmod 777 file")

    def test_safe_commands_not_dangerous(self):
        assert not self.agent._is_dangerous_command("ls -la")
        assert not self.agent._is_dangerous_command("cat file.txt")
        assert not self.agent._is_dangerous_command("git status")

    def test_execute_safe_command(self):
        """A safe command should execute successfully."""
        tc = ToolCall(id="1", name="execute_command", arguments={"command": "echo hello"})
        result = self.agent._execute_tool(tc)
        assert "hello" in result

    def test_empty_command_returns_error(self):
        """An empty command should return a clear error message."""
        tc = ToolCall(id="1", name="execute_command", arguments={"command": ""})
        result = self.agent._execute_tool(tc)
        assert "ERROR" in result
        assert "Empty command" in result

    def test_unknown_command_now_allowed(self):
        """Unknown commands should now be allowed by default (no more whitelist)."""
        tc = ToolCall(id="2", name="execute_command", arguments={"command": "which python3"})
        result = self.agent._execute_tool(tc)
        # Should NOT contain SECURITY — unknown commands are now allowed
        assert "SECURITY" not in result

    def test_dangerous_command_requires_approval(self):
        """A dangerous command should trigger the approval flow and be denied if user refuses."""
        self.agent.on_command_approval = lambda cmd: (False, "Testing: command not allowed")
        tc = ToolCall(id="3", name="execute_command", arguments={"command": "rm -rf /tmp/test"})
        result = self.agent._execute_tool(tc)
        assert "denied by user" in result.lower()
        assert "Testing" in result

    def test_dangerous_command_approved(self):
        """A dangerous command should execute if the user approves it."""
        self.agent.on_command_approval = lambda cmd: (True, "")
        tc = ToolCall(id="4", name="execute_command", arguments={"command": "echo approved_dangerous"})
        result = self.agent._execute_tool(tc)
        assert "approved_dangerous" in result

    def test_dangerous_command_no_approval_callback(self):
        """A dangerous command should be denied by default if no approval callback is set."""
        tc = ToolCall(id="5", name="execute_command", arguments={"command": "rm -rf /tmp/test"})
        result = self.agent._execute_tool(tc)
        assert "denied by user" in result.lower() or "SECURITY" in result

    def test_unknown_tool(self):
        """An unknown tool should return an error message."""
        tc = ToolCall(id="6", name="nonexistent_tool", arguments={})
        result = self.agent._execute_tool(tc)
        assert "Unknown tool" in result


class TestAgentBuiltinTools:
    """Test built-in tool execution."""

    def setup_method(self):
        self.config = Config(openrouter_api_key="sk-test")
        self.agent = Agent(config=self.config)

    def test_finish_tool(self):
        tc = ToolCall(id="1", name="finish", arguments={"summary": "done", "result": "ok"})
        result = self.agent._execute_tool(tc)
        assert "TASK COMPLETE" in result
        assert "done" in result
        assert "ok" in result

    def test_list_files(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tc = ToolCall(id="1", name="list_files", arguments={"path": tmpdir})
            result = self.agent._execute_tool(tc)
            assert "(empty directory)" in result

    def test_read_file_not_found(self):
        tc = ToolCall(id="1", name="read_file", arguments={"path": "/nonexistent/file.txt"})
        result = self.agent._execute_tool(tc)
        assert "File not found" in result

    def test_write_and_read_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = f"{tmpdir}/test.txt"
            # Write
            tc_write = ToolCall(id="1", name="write_file", arguments={"path": filepath, "content": "hello world"})
            result = self.agent._execute_tool(tc_write)
            assert "File written" in result
            # Read
            tc_read = ToolCall(id="2", name="read_file", arguments={"path": filepath})
            result = self.agent._execute_tool(tc_read)
            assert "hello world" in result

    def test_append_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = f"{tmpdir}/append.txt"
            # Write initial
            self.agent._execute_tool(ToolCall(id="1", name="write_file", arguments={"path": filepath, "content": "line1\n"}))
            # Append
            tc = ToolCall(id="2", name="append_file", arguments={"path": filepath, "content": "line2\n"})
            result = self.agent._execute_tool(tc)
            assert "Content appended" in result
            # Verify
            content = open(filepath).read()
            assert "line1" in content
            assert "line2" in content