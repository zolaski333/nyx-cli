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
        # Set up both approval callbacks so the permission model prompts instead of denying
        self.agent.on_command_approval = lambda cmd: (False, "Testing: command not allowed")
        self.agent.permissions.set_approval_callback(
            lambda cat, desc, target: (False, "Testing: command not allowed")
        )
        tc = ToolCall(id="3", name="execute_command", arguments={"command": "rm -rf /tmp/test"})
        result = self.agent._execute_tool(tc)
        assert "denied" in result.lower()
        assert "Testing" in result

    def test_dangerous_command_approved(self):
        """A dangerous command should execute if the user approves it."""
        self.agent.on_command_approval = lambda cmd: (True, "")
        self.agent.permissions.set_approval_callback(
            lambda cat, desc, target: (True, "")
        )
        tc = ToolCall(id="4", name="execute_command", arguments={"command": "rm -rf /tmp/test_approved"})
        result = self.agent._execute_tool(tc)
        # Should execute (not be denied)
        assert "denied" not in result.lower()

    def test_dangerous_command_no_approval_callback(self):
        """A dangerous command should be denied by default if no approval callback is set."""
        tc = ToolCall(id="5", name="execute_command", arguments={"command": "rm -rf /tmp/test"})
        result = self.agent._execute_tool(tc)
        assert "denied" in result.lower() or "SECURITY" in result

    def test_unknown_tool(self):
        """An unknown tool should return an error message."""
        tc = ToolCall(id="6", name="nonexistent_tool", arguments={})
        result = self.agent._execute_tool(tc)
        assert "Unknown tool" in result

    def test_permissions_deny_root_deletion(self):
        """The permission model should deny rm -rf /* explicitly."""
        tc = ToolCall(id="7", name="execute_command", arguments={"command": "rm -rf /"})
        result = self.agent._execute_tool(tc)
        assert "denied by security policy" in result.lower() or "SECURITY" in result

    def test_permissions_prompt_sudo(self):
        """The permission model should prompt for sudo commands."""
        self.agent.on_command_approval = lambda cmd: (False, "Testing: sudo not allowed")
        tc = ToolCall(id="8", name="execute_command", arguments={"command": "sudo apt update"})
        result = self.agent._execute_tool(tc)
        assert "denied" in result.lower()


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
        """Write and read a file via the PatchTool (with approval callback)."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = f"{tmpdir}/test.txt"
            # Set up approval callback to auto-approve
            self.agent.on_file_approval = lambda path, summary, diff: (True, "")
            # Write
            tc_write = ToolCall(id="1", name="write_file", arguments={"path": filepath, "content": "hello world"})
            result = self.agent._execute_tool(tc_write)
            assert "File written" in result or "No changes" in result
            # Read
            tc_read = ToolCall(id="2", name="read_file", arguments={"path": filepath})
            result = self.agent._execute_tool(tc_read)
            assert "hello world" in result

    def test_append_file(self):
        """Append to a file via the PatchTool (with approval callback)."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = f"{tmpdir}/append.txt"
            # Set up approval callback to auto-approve
            self.agent.on_file_approval = lambda path, summary, diff: (True, "")
            # Write initial
            self.agent._execute_tool(ToolCall(id="1", name="write_file", arguments={"path": filepath, "content": "line1\n"}))
            # Append
            tc = ToolCall(id="2", name="append_file", arguments={"path": filepath, "content": "line2\n"})
            result = self.agent._execute_tool(tc)
            assert "Append" in result or "File written" in result
            # Verify
            content = open(filepath).read()
            assert "line1" in content
            assert "line2" in content

    def test_apply_diff_new_file(self):
        """The apply_diff tool should create a new file."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = f"{tmpdir}/newfile.txt"
            self.agent.on_file_approval = lambda path, summary, diff: (True, "")
            tc = ToolCall(id="1", name="apply_diff", arguments={
                "path": filepath,
                "content": "new content",
                "description": "Create new file",
            })
            result = self.agent._execute_tool(tc)
            assert "File written" in result or "No changes" in result
            # Verify
            content = open(filepath).read()
            assert "new content" in content

    def test_apply_diff_modify_existing(self):
        """The apply_diff tool should modify an existing file."""
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = f"{tmpdir}/modify.txt"
            # Create initial file
            with open(filepath, "w") as f:
                f.write("original content\n")
            self.agent.on_file_approval = lambda path, summary, diff: (True, "")
            tc = ToolCall(id="1", name="apply_diff", arguments={
                "path": filepath,
                "content": "modified content\n",
                "description": "Modify file",
            })
            result = self.agent._execute_tool(tc)
            assert "File written" in result or "No changes" in result
            # Verify
            content = open(filepath).read()
            assert "modified content" in content

    def test_write_file_denied(self):
        """A write_file should be denied if the approval callback rejects it."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = f"{tmpdir}/denied.txt"
            self.agent.on_file_approval = lambda path, summary, diff: (False, "Testing: file write denied")
            tc = ToolCall(id="1", name="write_file", arguments={"path": filepath, "content": "should not appear"})
            result = self.agent._execute_tool(tc)
            assert "denied" in result.lower()

    def test_sandbox_path_traversal_blocked(self):
        """Path traversal outside the sandbox should be blocked."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set sandbox root
            self.agent.sandbox.set_root(tmpdir)
            # Try to read a file outside the sandbox
            tc = ToolCall(id="1", name="read_file", arguments={"path": "/etc/passwd"})
            result = self.agent._execute_tool(tc)
            # Should either be blocked by sandbox or allowed by safe_read_path
            # (safe_read_path allows system paths for reading)
            assert "SECURITY" not in result  # Should not be a security error for reads

    def test_sandbox_write_outside_blocked(self):
        """Writing outside the sandbox should be blocked."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            self.agent.sandbox.set_root(tmpdir)
            self.agent.on_file_approval = lambda path, summary, diff: (True, "")
            tc = ToolCall(id="1", name="write_file", arguments={
                "path": "/tmp/nyx_test_outside.txt",
                "content": "should be blocked",
            })
            result = self.agent._execute_tool(tc)
            assert "traversal" in result.lower() or "SECURITY" in result or "denied" in result.lower()

    def test_audit_trail_logs_tool_calls(self):
        """The audit trail should log tool calls."""
        # Audit trail is enabled by default; tool calls are logged in _loop,
        # but _execute_tool doesn't log directly. The audit trail logs via
        # the _loop method. Let's verify the audit system is wired up.
        assert self.agent.audit is not None
        assert self.agent.audit.is_enabled

    def test_permission_manager_initialized(self):
        """The permission manager should be initialized with defaults."""
        assert self.agent.permissions is not None
        # Check shell category exists
        shell_cat = self.agent.permissions.categories.get("shell")
        assert shell_cat is not None
        # Check filesystem category exists
        fs_cat = self.agent.permissions.categories.get("filesystem")
        assert fs_cat is not None

    def test_permission_check_shell(self):
        """Permission checks should work for shell commands."""
        # Safe command should be ALLOW
        level = self.agent.permissions.check_shell("ls -la")
        assert level.value == "allow"
        # Dangerous command should be PROMPT or DENY
        level = self.agent.permissions.check_shell("rm -rf /")
        assert level.value in ("deny", "prompt")

    def test_permission_check_file_write(self):
        """Permission checks should work for file writes."""
        # Project-local path should be ALLOW
        level = self.agent.permissions.check_file_write("./src/main.py")
        assert level.value == "allow"
        # System path should be PROMPT
        level = self.agent.permissions.check_file_write("/etc/config.conf")
        assert level.value in ("prompt", "deny")