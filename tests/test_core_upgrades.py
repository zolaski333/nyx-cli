"""Tests for Nyx core upgrades — fuzzy diffing, token truncation, and subagent security."""
from __future__ import annotations

import tempfile
from pathlib import Path

from nyx.agent import Agent, AgentContext
from nyx.config import Config
from nyx.providers.base import ToolCall
from nyx.diff_tool import _apply_search_replace_to_content
from nyx.subagent import Subagent


class TestFuzzyDiff:
    """Test that the fuzzy diff matcher tolerates minor spacing/content deviations."""

    def test_fuzzy_search_replace_match(self):
        original = (
            "def foo(x):\n"
            "    # This is a comment\n"
            "    print(x)\n"
            "    return x + 1\n"
        )
        # SEARCH block has slightly different comment and spaces
        patch = (
            "<<<<<<< SEARCH\n"
            "    # This comment is slightly different\n"
            "    print(x)\n"
            "=======\n"
            "    print('debug:', x)\n"
            ">>>>>>> REPLACE\n"
        )
        result = _apply_search_replace_to_content(original, patch)
        assert result is not None
        assert "debug:" in result
        assert "return x + 1" in result


class TestTokenTruncation:
    """Test that long inputs are truncated to prevent context saturation."""

    def test_context_truncation(self):
        ctx = AgentContext()
        huge_content = "A" * 20000
        ctx.add("user", huge_content)
        
        # Should be truncated
        content = ctx.messages[0]["content"]
        assert len(content) < 20000
        assert "TRUNCATED" in content


class TestSubagentSecurity:
    """Test that subagents route execution through the parent's context and permission guards."""

    def test_subagent_uses_parent_context_and_guards(self):
        config = Config(openrouter_api_key="sk-test", sandbox_enabled=True)
        agent = Agent(config=config)
        agent.setup()

        # Create a temp directory for sandbox
        with tempfile.TemporaryDirectory() as tmpdir:
            agent.sandbox.set_root(tmpdir)
            # Override command approval to deny
            agent.on_command_approval = lambda cmd: (False, "Denied by user test")
            agent.permissions.set_approval_callback(lambda cat, desc, target: (False, "Denied by user test"))

            # Spawn a subagent
            sub = agent.subagents.spawn("test-subagent")
            assert sub.context is agent.tool_context

            # Let subagent call a dangerous command
            tc = ToolCall(id="1", name="execute_command", arguments={"command": "rm -rf /"})
            from nyx.tools import execute_tool
            res = execute_tool(tc, sub.context)
            
            # Should be denied because it routed through the parent's guard!
            assert "denied" in res.lower() or "security" in res.lower()
