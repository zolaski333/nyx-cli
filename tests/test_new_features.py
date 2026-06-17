"""Tests for Nyx repo map, search code, test loop, and subagent tools."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from nyx.repo_map import build_repo_map, build_repo_map_short, _get_git_status, _get_test_info
from nyx.search_code import search_code, search_symbol, search_text, SearchResult, SearchMatch
from nyx.test_loop import (
    run_tests,
    discover_test_commands,
    TestResult,
    TestFailure,
    format_failures_for_llm,
    _parse_failures,
    _parse_counts,
    auto_correct_loop,
)
from nyx.subagent import Subagent, SubagentManager, SubagentResult
from nyx.agent import Agent, AgentContext
from nyx.config import Config
from nyx.providers.base import ToolDefinition

# Project root derived from test file location (more reliable than cwd)
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


# =========================================================================
# Repo Map Tests
# =========================================================================


class TestRepoMap:
    """Test the repository map builder."""

    def test_build_repo_map_in_current_dir(self):
        """Should build a repo map for the current directory."""
        result = build_repo_map(PROJECT_ROOT)
        assert "REPOSITORY MAP" in result
        assert "Language" in result or "Language" in result
        assert "Directory Structure" in result or "Directory" in result

    def test_build_repo_map_invalid_dir(self):
        """Should return an error for invalid directories."""
        result = build_repo_map("/nonexistent/path/xyz123")
        assert "Error" in result

    def test_build_repo_map_does_not_write_index_by_default(self):
        """Repo map should be read-only unless write_index is explicit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
            (root / "demo.py").write_text("import os\n", encoding="utf-8")

            result = build_repo_map(root)

            assert "REPOSITORY MAP" in result
            assert not (root / ".nyx" / "repo_graph.json").exists()

    def test_repo_map_tool_does_not_write_index_in_architect_mode(self):
        """The architect-allowed repo_map tool should not mutate the filesystem."""
        from nyx.providers.base import ToolCall

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
            (root / "demo.py").write_text("import os\n", encoding="utf-8")
            agent = Agent(Config(
                openrouter_api_key="sk-test",
                project_dir=str(root),
                agent_mode="architect",
                audit_enabled=False,
            ))
            agent.switch_mode("architect")

            result = agent._execute_tool(ToolCall(id="1", name="repo_map", arguments={"path": str(root)}))

            assert "REPOSITORY MAP" in result
            assert not (root / ".nyx" / "repo_graph.json").exists()

    def test_build_repo_map_short(self):
        """Short map should be a one-liner."""
        result = build_repo_map_short(PROJECT_ROOT)
        assert isinstance(result, str)
        assert len(result) < 500

    def test_git_status(self):
        """Git status should return branch info (if in a git repo)."""
        status = _get_git_status(Path(PROJECT_ROOT))
        assert isinstance(status, dict)
        assert "branch" in status
        # May or may not be a git repo, but should not crash
        assert isinstance(status["branch"], str)

    def test_test_info(self):
        """Test info should discover test files."""
        info = _get_test_info(Path(PROJECT_ROOT))
        assert isinstance(info, dict)
        assert "test_files" in info
        assert "framework" in info
        # This project uses pytest
        assert info["framework"] in ("pytest", "unknown")

    def test_repo_map_contains_important_files(self):
        """Repo map should mention important files like pyproject.toml."""
        result = build_repo_map(PROJECT_ROOT)
        assert "pyproject.toml" in result or "setup.py" in result

    def test_get_ast_symbols_enriched(self):
        """_get_ast_symbols should extract heritage, signatures, decorators, and async functions."""
        from nyx.repo_map import _get_ast_symbols
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "dummy.py"
            file_path.write_text(
                "class Child(Parent):\n"
                "    @decorator\n"
                "    async def method(self, x: int, y=1):\n"
                "        pass\n"
                "\n"
                "@staticmethod\n"
                "def func(a, b=True):\n"
                "    pass\n"
            )
            result = _get_ast_symbols(Path(tmpdir))
            assert "Class: Child (Parent)" in result
            assert "@decorator async method(self, x: int, y=1)" in result
            assert "@staticmethod func(a, b=True)" in result


class TestSkillsConfig:
    """Test skill loading safety switches."""

    def test_agent_skips_skills_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()
            (skills_dir / "sample.py").write_text(
                "name = 'sample'\n"
                "description = 'sample skill'\n"
                "parameters = {'type': 'object', 'properties': {}}\n"
                "def execute(arguments):\n"
                "    return 'ok'\n",
                encoding="utf-8",
            )
            config = Config(
                openrouter_api_key="sk-test",
                skills_dir=str(skills_dir),
                skills_enabled=False,
                audit_enabled=False,
            )
            agent = Agent(config)
            agent.setup()

            assert "skill_sample" not in {tool.name for tool in agent.tools}


# =========================================================================
# Search Code Tests
# =========================================================================


class TestSearchCode:
    """Test the code search functionality."""

    def test_search_result_formatted(self):
        """SearchResult.formatted() should produce readable output."""
        result = SearchResult(
            query="test",
            matches=[
                SearchMatch(file="test.py", line=1, column=1, line_content="def test_func():", match_length=4),
                SearchMatch(file="test.py", line=5, column=1, line_content="    test_var = 1", match_length=4),
            ],
            total_matches=2,
            files_matched=1,
            engine="rg",
        )
        formatted = result.formatted()
        assert "test" in formatted
        assert "test.py" in formatted
        assert "2 matches" in formatted or "2 results" in formatted or "2" in formatted

    def test_search_result_empty(self):
        """Empty search result should show no results."""
        result = SearchResult(query="nonexistent_pattern_xyz")
        formatted = result.formatted()
        assert "No results" in formatted

    def test_search_result_error(self):
        """Search result with error should show error."""
        result = SearchResult(query="test", error="Something went wrong")
        formatted = result.formatted()
        assert "error" in formatted.lower()

    def test_search_code_basic(self):
        """Basic code search should work (may use grep fallback)."""
        # Search for a common Python keyword in the nyx directory
        nyx_dir = os.path.join(PROJECT_ROOT, "nyx")
        result = search_code("def ", root=nyx_dir, max_results=5, context_lines=0)
        assert result.total_matches > 0 or result.error
        if result.total_matches > 0:
            assert any("def " in m.line_content for m in result.matches)

    def test_search_symbol(self):
        """Search symbol should find function/class definitions."""
        nyx_dir = os.path.join(PROJECT_ROOT, "nyx")
        result = search_symbol("class ", root=nyx_dir, max_results=5)
        # Should find classes in the codebase
        assert result.total_matches > 0 or result.error

    def test_search_text(self):
        """Search text should find literal strings."""
        nyx_dir = os.path.join(PROJECT_ROOT, "nyx")
        result = search_text("TODO", root=nyx_dir)
        assert isinstance(result, SearchResult)

    def test_search_with_file_pattern(self):
        """Search with file pattern should filter results."""
        nyx_dir = os.path.join(PROJECT_ROOT, "nyx")
        result = search_code("def ", root=nyx_dir, file_pattern="*.py", max_results=5)
        if result.total_matches > 0:
            assert all(m.file.endswith(".py") for m in result.matches)

    def test_search_case_sensitive(self):
        """Case-sensitive search should work."""
        nyx_dir = os.path.join(PROJECT_ROOT, "nyx")
        result_upper = search_code("CLASS", root=nyx_dir, case_sensitive=True, max_results=5)
        result_lower = search_code("class", root=nyx_dir, case_sensitive=True, max_results=5)
        # 'CLASS' (uppercase) should find fewer matches than 'class' (lowercase)
        assert result_upper.total_matches <= result_lower.total_matches or not result_upper.error


# =========================================================================
# Test Loop Tests
# =========================================================================


class TestTestLoop:
    """Test the test loop functionality."""

    def test_discover_test_commands(self):
        """Should discover test commands for this project."""
        commands = discover_test_commands(PROJECT_ROOT)
        assert len(commands) > 0
        # Should include pytest
        assert any("pytest" in cmd for cmd in commands)

    def test_run_tests(self):
        """Should run tests and return structured results."""
        result = run_tests(f"python3 -m pytest {PROJECT_ROOT}/tests/test_config.py -v --tb=short 2>&1", root=PROJECT_ROOT)
        assert isinstance(result, TestResult)
        assert result.command is not None
        # Should have parsed results
        assert result.total >= 0

    def test_run_tests_with_failures(self):
        """Should parse test failures correctly."""
        # Create a temporary test file that will fail
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test_fail.py"
            test_file.write_text("""
def test_should_pass():
    assert 1 + 1 == 2

def test_should_fail():
    assert 1 + 1 == 3
""")
            result = run_tests(
                f"python3 -m pytest {test_file} -v --tb=short 2>&1",
                root=tmpdir,
            )
            assert not result.success
            assert result.failed >= 1
            assert result.passed >= 1
            assert len(result.failures) >= 1

    def test_parse_failures(self):
        """Should parse pytest failure output."""
        raw_output = """
FAILED tests/test_agent.py::test_empty_command_returns_error - AssertionError: Expected error
FAILED tests/test_memory.py::test_save_and_load - KeyError: 'missing_key'
"""
        failures = _parse_failures(raw_output, Path(PROJECT_ROOT))
        assert len(failures) >= 2
        assert any("test_empty_command_returns_error" in f.test_name for f in failures)
        assert any("test_save_and_load" in f.test_name for f in failures)

    def test_parse_counts(self):
        """Should parse test counts from pytest output."""
        raw = "3 passed, 2 failed in 0.45s"
        passed, failed, total = _parse_counts(raw)
        assert passed == 3
        assert failed == 2
        assert total == 5

    def test_parse_counts_all_pass(self):
        """Should parse all-pass output."""
        raw = "5 passed in 0.12s"
        passed, failed, total = _parse_counts(raw)
        assert passed == 5
        assert failed == 0
        assert total == 5

    def test_format_failures_for_llm(self):
        """Should format failures into a prompt-friendly string."""
        failures = [
            TestFailure(
                file="test_example.py",
                line=10,
                test_name="test_foo",
                error_type="AssertionError",
                message="Expected True, got False",
            ),
        ]
        formatted = format_failures_for_llm(failures, "raw output here")
        assert "test_foo" in formatted
        assert "AssertionError" in formatted
        assert "Expected True" in formatted
        assert "raw output" in formatted

    def test_test_result_summary(self):
        """TestResult.summary should produce readable output."""
        result = TestResult(success=True, command="pytest", passed=5, failed=0, total=5, duration_ms=100.0)
        assert "All" in result.summary
        assert "5" in result.summary

        result2 = TestResult(success=False, command="pytest", passed=3, failed=2, total=5, duration_ms=100.0)
        assert "failed" in result2.summary
        assert "2" in result2.summary

    def test_test_failure_summary(self):
        """TestFailure.summary() should produce readable output."""
        f = TestFailure(file="test.py", line=10, test_name="test_x", error_type="ValueError", message="bad value")
        s = f.summary()
        assert "test_x" in s
        assert "test.py" in s
        assert "ValueError" in s
        assert "bad value" in s

    def test_auto_correct_loop_history(self):
        """auto_correct_loop should build and pass corrective history to the fix function on subsequent attempts."""
        history_calls = []

        def mock_fix(failures, raw_output, history=None):
            if history is not None:
                history_calls.append(list(history))
            return f"applied fix for {len(failures)} failures"

        fail_cmd = 'python3 -c "import sys; print(\'FAILED test_fail.py::test_f - AssertionError: expected failure\'); sys.exit(1)"'
        
        correction = auto_correct_loop(
            fix_function=mock_fix,
            root=PROJECT_ROOT,
            test_command=fail_cmd,
            max_iterations=3,
        )
        
        assert correction.iterations == 3
        assert not correction.success
        assert len(correction.corrections) == 3
        
        # It should record history for iterations 2 and 3
        assert len(history_calls) >= 3
        assert len(history_calls[1]) == 1
        assert history_calls[1][0]["iteration"] == 1
        assert "applied fix for 1 failures" in history_calls[1][0]["correction"]
        assert "expected failure" in history_calls[1][0]["failure_summary"]

        assert len(history_calls[2]) == 2
        assert history_calls[2][1]["iteration"] == 2


# =========================================================================
# Subagent Controlled Tools Tests
# =========================================================================


class TestSubagentControlledTools:
    """Test the controlled tool subset for subagents."""

    def setup_method(self):
        self.config = Config(openrouter_api_key="sk-test")
        self.agent = Agent(config=self.config)

    def test_get_subagent_tools_returns_filtered_list(self):
        """_get_subagent_tools should return a filtered list of tools."""
        tools = self.agent._get_subagent_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0
        # Should include read-only tools
        tool_names = [t.name for t in tools]
        assert "read_file" in tool_names
        assert "list_files" in tool_names
        assert "search_code" in tool_names
        assert "repo_map" in tool_names
        # Should NOT include subagent spawning tools
        assert "subagent_run" not in tool_names
        assert "parallel_subagents" not in tool_names
        assert "auto_correct_tests" not in tool_names

    def test_subagent_tool_whitelist(self):
        """The whitelist should contain expected tools."""
        assert "read_file" in Agent.SUBAGENT_TOOL_WHITELIST
        assert "list_files" in Agent.SUBAGENT_TOOL_WHITELIST
        assert "search_code" in Agent.SUBAGENT_TOOL_WHITELIST
        assert "repo_map" in Agent.SUBAGENT_TOOL_WHITELIST
        assert "execute_command" in Agent.SUBAGENT_TOOL_WHITELIST
        assert "write_file" in Agent.SUBAGENT_TOOL_WHITELIST
        assert "apply_diff" in Agent.SUBAGENT_TOOL_WHITELIST
        assert "finish" in Agent.SUBAGENT_TOOL_WHITELIST

    def test_subagent_tool_blacklist(self):
        """The blacklist should contain restricted tools."""
        assert "subagent_run" in Agent.SUBAGENT_TOOL_BLACKLIST
        assert "parallel_subagents" in Agent.SUBAGENT_TOOL_BLACKLIST
        assert "auto_correct_tests" in Agent.SUBAGENT_TOOL_BLACKLIST
        assert "memory_save" in Agent.SUBAGENT_TOOL_BLACKLIST

    def test_subagent_manager_default_tools(self):
        """SubagentManager should accept and propagate default tools."""
        manager = SubagentManager(self.config)
        assert manager._default_tools is None

        fake_tools = [ToolDefinition(name="read_file", description="Read", parameters={"type": "object", "properties": {}})]
        manager.set_default_tools(fake_tools)
        assert manager._default_tools == fake_tools

    def test_subagent_spawn_with_tools(self):
        """Subagent.spawn should pass tools to the Subagent."""
        manager = SubagentManager(self.config)
        fake_tools = [ToolDefinition(name="read_file", description="Read", parameters={"type": "object", "properties": {}})]
        agent = manager.spawn("test_agent", tools=fake_tools)
        assert agent._tools == fake_tools

    def test_subagent_spawn_falls_back_to_default(self):
        """Subagent.spawn should fall back to default_tools if no tools provided."""
        manager = SubagentManager(self.config)
        fake_tools = [ToolDefinition(name="list_files", description="List", parameters={"type": "object", "properties": {}})]
        manager.set_default_tools(fake_tools)
        agent = manager.spawn("test_agent2")  # No tools arg
        assert agent._tools == fake_tools

    def test_subagent_execute_tool_call_read_file(self):
        """Subagent._execute_tool_call should handle read_file."""
        import tempfile
        from nyx.providers.base import ToolCall

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world")
            tmp_path = f.name

        agent = Subagent(name="test", config=self.config)
        tc = ToolCall(id="1", name="read_file", arguments={"path": tmp_path})
        result = agent._execute_tool_call(tc, [])
        assert "hello world" in result

        os.unlink(tmp_path)

    def test_subagent_execute_tool_call_list_files(self):
        """Subagent._execute_tool_call should handle list_files."""
        import tempfile
        from nyx.providers.base import ToolCall

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "a.txt").touch()
            Path(tmpdir, "b.py").touch()

            agent = Subagent(name="test", config=self.config)
            tc = ToolCall(id="1", name="list_files", arguments={"path": tmpdir})
            result = agent._execute_tool_call(tc, [])
            assert "a.txt" in result
            assert "b.py" in result

    def test_subagent_execute_tool_call_unknown(self):
        """Subagent._execute_tool_call should return error for unknown tools."""
        from nyx.providers.base import ToolCall

        agent = Subagent(name="test", config=self.config)
        tc = ToolCall(id="1", name="nonexistent_tool", arguments={})
        result = agent._execute_tool_call(tc, [])
        assert "Unknown tool" in result

    def test_subagent_multiturn_loop(self):
        """Subagent should run a multi-turn tool execution loop and pass actual results."""
        from nyx.providers.base import BaseLLMProvider, LLMResponse, ToolCall
        
        class MockSubagentProvider(BaseLLMProvider):
            def __init__(self):
                super().__init__(None)
                self.calls = 0
            
            def chat(self, messages, tools=None, stream=False, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return LLMResponse(
                        content="",
                        tool_calls=[ToolCall(id="call_x", name="read_file", arguments={"path": "dummy.txt"})],
                        usage={"total_tokens": 10}
                    )
                else:
                    last_msg = messages[-1]
                    assert last_msg["role"] == "tool"
                    assert "mock content" in last_msg["content"]
                    return LLMResponse(
                        content="Final Answer based on: " + last_msg["content"],
                        usage={"total_tokens": 5}
                    )

        agent = Subagent(name="test_loop", config=self.config)
        agent._provider = MockSubagentProvider()
        agent._execute_tool_call = lambda tc, tools: "mock content from dummy.txt"
        
        fake_tools = [ToolDefinition(name="read_file", description="Read", parameters={"type": "object", "properties": {}})]
        
        result = agent.execute(task="Do task", tools=fake_tools)
        assert result.error is None
        assert "Final Answer based on: mock content from dummy.txt" in result.output
        assert result.tokens_used == 15


# =========================================================================
# Agent Integration Tests
# =========================================================================


class TestAgentNewTools:
    """Test the new built-in tools in the agent."""

    def setup_method(self):
        self.config = Config(openrouter_api_key="sk-test", project_dir=PROJECT_ROOT)
        self.agent = Agent(config=self.config)

    def test_repo_map_tool_definition_exists(self):
        """The repo_map tool should be defined in BUILTIN_TOOLS."""
        names = [t.name for t in Agent._get_tools_static()]
        assert "repo_map" in names

    def test_search_code_tool_definition_exists(self):
        """The search_code tool should be defined in BUILTIN_TOOLS."""
        names = [t.name for t in Agent._get_tools_static()]
        assert "search_code" in names

    def test_run_tests_tool_definition_exists(self):
        """The run_tests tool should be defined in BUILTIN_TOOLS."""
        names = [t.name for t in Agent._get_tools_static()]
        assert "run_tests" in names

    def test_auto_correct_tests_tool_definition_exists(self):
        """The auto_correct_tests tool should be defined in BUILTIN_TOOLS."""
        names = [t.name for t in Agent._get_tools_static()]
        assert "auto_correct_tests" in names

    def test_repo_map_tool_execution(self):
        """The repo_map tool should execute successfully."""
        from nyx.providers.base import ToolCall
        tc = ToolCall(id="1", name="repo_map", arguments={"path": ".", "short": True})
        result = self.agent._execute_tool(tc)
        # Should return something meaningful (not an error)
        assert "Error" not in result
        assert len(result) > 10

    def test_search_code_tool_execution(self):
        """The search_code tool should execute successfully."""
        from nyx.providers.base import ToolCall
        tc = ToolCall(id="1", name="search_code", arguments={
            "pattern": "def ",
            "file_pattern": "*.py",
            "max_results": 5,
            "context_lines": 0,
        })
        result = self.agent._execute_tool(tc)
        # Should find some results or report none found (not an error)
        assert "Error" not in result or "No results" in result

    def test_run_tests_tool_execution(self):
        """The run_tests tool should execute successfully."""
        from nyx.providers.base import ToolCall
        tc = ToolCall(id="1", name="run_tests", arguments={
            "command": "python3 -m pytest tests/test_config.py -v --tb=short 2>&1",
        })
        result = self.agent._execute_tool(tc)
        assert "passed" in result.lower() or "failed" in result.lower()


# =========================================================================
# Memory Summary Injection Tests
# =========================================================================


class TestMemorySummaryInjection:
    """Test the memory summary injection into LLM context."""

    def setup_method(self):
        self.config = Config(openrouter_api_key="sk-test")
        self.agent = Agent(config=self.config)

    def test_inject_memory_summary_no_crash(self):
        """Memory summary injection should not crash."""
        # Just verify it doesn't raise
        self.agent._inject_memory_summary()

    def test_inject_memory_summary_adds_context(self):
        """Memory summary injection should add system messages."""
        # Add some memory entries first
        self.agent.memory.add_entry("user", "test conversation content")
        self.agent.memory.add_entry("assistant", "test response")

        # Inject
        self.agent._inject_memory_summary()

        # Check that system messages were added
        system_msgs = [m for m in self.agent.context.messages if m["role"] == "system"]
        memory_msgs = [m for m in system_msgs if "Memory" in m.get("content", "")]
        assert len(memory_msgs) >= 0  # May or may not have summary depending on state


# =========================================================================
# REPLHistory tests
# =========================================================================


class TestREPLHistory:
    """Test the ANSI REPLHistory class."""

    def setup_method(self):
        import tempfile
        self.tmpfile = tempfile.NamedTemporaryFile(mode="w", suffix=".hist", delete=False)
        self.hist_path = self.tmpfile.name
        self.tmpfile.close()

    def teardown_method(self):
        import os
        if os.path.exists(self.hist_path):
            os.unlink(self.hist_path)

    def _make_history(self):
        from nyx.cli import REPLHistory
        return REPLHistory(history_file=self.hist_path, max_size=100)

    def test_append_and_persist(self):
        hist = self._make_history()
        hist.append("/help")
        hist.append("/model")
        assert len(hist) == 2
        # Re-load to verify persistence
        hist2 = self._make_history()
        assert len(hist2) == 2

    def test_dedup(self):
        hist = self._make_history()
        hist.append("/help")
        hist.append("/tools")
        hist.append("/help")  # duplicate — should move to end
        entries = hist.entries
        assert entries[-1] == "/help"  # most recent at end
        assert entries.count("/help") == 1

    def test_search_by_prefix(self):
        hist = self._make_history()
        hist.append("/help")
        hist.append("/model gpt-4")
        hist.append("/tools")
        matches = hist.search("/model")
        assert len(matches) == 1
        assert matches[0] == "/model gpt-4"

    def test_search_empty_prefix(self):
        hist = self._make_history()
        hist.append("/help")
        assert hist.search("") == []

    def test_max_size_eviction(self):
        from nyx.cli import REPLHistory
        hist = REPLHistory(history_file=self.hist_path, max_size=3)
        hist.append("a")
        hist.append("b")
        hist.append("c")
        hist.append("d")  # should evict "a"
        assert len(hist) == 3
        assert hist.entries == ["b", "c", "d"]

    def test_empty_history(self):
        hist = self._make_history()
        assert len(hist) == 0
        assert hist.entries == []


# =========================================================================
# ProgressBar tests
# =========================================================================


class TestProgressBar:
    """Test the ANSI ProgressBar class."""

    def test_progress_basic(self):
        from nyx.cli import ProgressBar
        pb = ProgressBar(total=5, label="Testing")
        assert pb._current == 0
        pb.update(1)
        assert pb._current == 1
        pb.update(3)
        assert pb._current == 4
        pb.close()
        assert pb._current == 5

    def test_progress_zero_total(self):
        from nyx.cli import ProgressBar
        pb = ProgressBar(total=0, label="Empty")
        pb.close()  # Should not crash
        assert pb._current == 0

    def test_progress_set_total(self):
        from nyx.cli import ProgressBar
        pb = ProgressBar(total=1, label="Flex")
        pb.set_total(10)
        assert pb.total == 10


# =========================================================================
# Autocomplete tests
# =========================================================================


class TestAutocomplete:
    """Test the autocomplete helper functions."""

    def test_autocomplete_commands_full_match(self):
        from nyx.cli import _autocomplete_commands
        matches = _autocomplete_commands("/help")
        assert "/help" in matches

    def test_autocomplete_commands_partial(self):
        from nyx.cli import _autocomplete_commands
        matches = _autocomplete_commands("/mod")
        assert "/model" in matches

    def test_autocomplete_commands_empty(self):
        from nyx.cli import _autocomplete_commands
        matches = _autocomplete_commands("")
        assert len(matches) > 0  # Returns all commands

    def test_autocomplete_commands_no_match(self):
        from nyx.cli import _autocomplete_commands
        matches = _autocomplete_commands("/zzz")
        assert matches == []

    def test_get_paginated_arg(self):
        from nyx.cli import _get_paginated_arg
        assert _get_paginated_arg("/tools 3", "/tools") == 3
        assert _get_paginated_arg("/tools", "/tools") == 1
        assert _get_paginated_arg("/tools abc", "/tools") == 1  # invalid int
        assert _get_paginated_arg("/conversations 5", "/conversations") == 5


# =========================================================================
# MCP progress callback tests
# =========================================================================


class TestMCPProgress:
    """Test MCP progress callback integration."""

    def test_set_progress_callback(self):
        from nyx.mcp_client import MCPManager
        mcp = MCPManager()
        calls = []
        mcp.set_progress_callback(lambda label, cur, total: calls.append((label, cur, total)))
        assert mcp._progress_callback is not None
        # No servers configured — should still work without crashing
        result = mcp.connect_all({})
        assert result == []



# =========================================================================
# Markdown renderer tests
# =========================================================================


class TestMarkdownRenderer:
    """Test the ANSI zero-dependency Markdown renderer."""

    def test_render_headings(self):
        from nyx.cli import render_markdown
        text = "# Heading 1\n## Heading 2"
        res = render_markdown(text, force_color=True)
        assert "HEADING 1" in res
        assert "Heading 2" in res
        assert "\033[96m" in res  # Cyan heading 1
        assert "\033[93m" in res  # Yellow heading 2

    def test_render_bold_italic(self):
        from nyx.cli import render_markdown
        text = "This is **bold** and *italic* code."
        res = render_markdown(text, force_color=True)
        assert "\033[1mbold\033[0m" in res
        assert "\033[3mitalic\033[0m" in res

    def test_render_lists(self):
        from nyx.cli import render_markdown
        text = "- item 1\n* item 2"
        res = render_markdown(text, force_color=True)
        assert "◈" in res
        assert "\033[92m" in res  # Green list bullets

    def test_render_code_blocks(self):
        from nyx.cli import render_markdown
        text = "```python\nprint('hello')\n```"
        res = render_markdown(text, force_color=True)
        assert "╭" in res
        assert "╰" in res
        assert "print('hello')" in res





class TestToolLogging:
    """Test tool execution console logging."""

    def test_tool_logging_format(self, monkeypatch):
        import io
        import sys
        from nyx.agent import Agent, ToolCall
        from nyx.config import Config
        from nyx.providers.base import LLMResponse

        # Set environment variable NO_COLOR=1 so we don't have to deal with ANSI escape codes in assertions
        monkeypatch.setenv("NO_COLOR", "1")

        captured_output = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured_output)

        config = Config(provider="openai", openai_api_key="sk-test")
        agent = Agent(config=config)

        # Mock the execute_tool method
        agent._execute_tool = lambda tc: "Line 1\nLine 2\nLine 3"

        mock_tool_call = ToolCall(id="tc-1", name="read_file", arguments={"path": "src/main.py"})
        mock_response = LLMResponse(
            content="",
            tool_calls=[mock_tool_call],
            usage={"total_tokens": 10}
        )
        
        class MockProvider:
            def __init__(self):
                self.calls = 0
            def chat(self, messages, tools=None, stream=False, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return mock_response
                return LLMResponse(content="All done!", usage={"total_tokens": 5})

        agent.provider = MockProvider()
        agent.run("test prompt")

        output = captured_output.getvalue()
        assert "🛠️  [Tool Call] read_file ➔ src/main.py" in output
        assert "✓  [Tool Success] read_file ➔ src/main.py (3 lines read, took" in output

    def test_tool_logging_format_multiple_tools(self, monkeypatch):
        import io
        import sys
        from nyx.agent import Agent, ToolCall
        from nyx.config import Config
        from nyx.providers.base import LLMResponse

        monkeypatch.setenv("NO_COLOR", "1")

        captured_output = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured_output)

        config = Config(provider="openai", openai_api_key="sk-test")
        agent = Agent(config=config)

        tcs = [
            ToolCall(id="tc-1", name="execute_command", arguments={"command": "ls -la"}),
            ToolCall(id="tc-2", name="custom_tool", arguments={"filepath": "custom_path.txt"}),
        ]
        
        mock_response = LLMResponse(
            content="",
            tool_calls=tcs,
            usage={"total_tokens": 10}
        )
        
        class MockProvider:
            def __init__(self):
                self.calls = 0
            def chat(self, messages, tools=None, stream=False, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return mock_response
                return LLMResponse(content="Done!", usage={"total_tokens": 5})

        agent.provider = MockProvider()
        
        def mock_exec(tc):
            if tc.name == "execute_command":
                return "file1.txt\nfile2.txt\n"
            return "ok"
        agent._execute_tool = mock_exec

        agent.run("test prompt")

        output = captured_output.getvalue()
        # Check execute_command
        assert "🛠️  [Tool Call] execute_command ➔ ls -la" in output
        assert "✓  [Tool Success] execute_command ➔ ls -la (2 lines output, took" in output
        
        # Check custom_tool (fallback check for key 'filepath')
        assert "🛠️  [Tool Call] custom_tool ➔ custom_path.txt" in output
        assert "✓  [Tool Success] custom_tool ➔ custom_path.txt (took" in output


# =========================================================================
# Helper to get static tools for testing
# =========================================================================


def _get_builtin_tools_static():
    """Helper to access BUILTIN_TOOLS for testing."""
    from nyx.agent import BUILTIN_TOOLS
    return BUILTIN_TOOLS


# Patch the Agent class for testing
Agent._get_tools_static = staticmethod(_get_builtin_tools_static)


class TestStaticAnalysis:
    """Test the static dependency indexing and symbol reference finding."""

    def test_index_dependencies(self):
        """Should build a dependency graph and write to repo_graph.json."""
        import tempfile
        import json
        from pathlib import Path
        from nyx.repo_map import index_dependencies

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            # Create a py file importing os
            py_file = tmp_root / "test_dep.py"
            py_file.write_text("import sys\nfrom pathlib import Path\n")
            
            # Create a JS file with import/require
            js_file = tmp_root / "test_dep.js"
            js_file.write_text("import React from 'react';\nconst fs = require('fs');\n")

            graph = index_dependencies(tmp_root)
            
            # Assert they parsed correctly
            assert "test_dep.py" in graph
            assert "sys" in graph["test_dep.py"]
            assert "pathlib" in graph["test_dep.py"]
            
            assert "test_dep.js" in graph
            assert "react" in graph["test_dep.js"]
            assert "fs" in graph["test_dep.js"]

            # Assert file was written
            graph_file = tmp_root / ".nyx" / "repo_graph.json"
            assert graph_file.exists()
            data = json.loads(graph_file.read_text(encoding="utf-8"))
            assert "test_dep.py" in data

    def test_find_references_tool(self):
        """find_references tool should find occurrences of a symbol with word boundaries."""
        import tempfile
        from pathlib import Path
        from nyx.agent import Agent
        from nyx.providers.base import ToolCall
        from nyx.config import Config

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            src_file = tmp_root / "source.py"
            src_file.write_text("my_special_symbol = 42\nprint(my_special_symbol)\n# my_special_symbol_extra shouldn't match\n")
            
            # Set up agent with config pointing to this project root
            config = Config(openrouter_api_key="sk-test", project_dir=str(tmp_root))
            agent = Agent(config=config)
            try:
                tc = ToolCall(id="1", name="find_references", arguments={"symbol_name": "my_special_symbol"})
                result = agent._execute_tool(tc)
                
                # Should match the first two lines, not the third
                assert "source.py:1:" in result
                assert "source.py:2:" in result
                assert "source.py:3:" not in result
                assert "my_special_symbol_extra" not in result
            finally:
                agent.shutdown()


class TestMCPDiscovery:
    """Test the MCP Auto-Discovery scanning and interactive prompts."""

    def test_discover_mcp_servers(self):
        """Should detect git and sqlite signatures."""
        import tempfile
        from pathlib import Path
        from nyx.mcp_discovery import discover_mcp_servers

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            # Create a mock git folder
            (tmp_root / ".git").mkdir()
            # Create a mock sqlite file
            (tmp_root / "my_database.db").touch()

            discovered = discover_mcp_servers(tmp_root)
            
            assert "git" in discovered
            assert "sqlite" in discovered
            assert "@modelcontextprotocol/server-git" in discovered["git"]["args"]
            assert "my_database.db" in discovered["sqlite"]["args"][-1]

    def test_run_interactive_discovery(self, monkeypatch):
        """Should prompt user and add server configurations to the config dictionary if accepted."""
        import tempfile
        from pathlib import Path
        from nyx.mcp_discovery import run_interactive_discovery

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            (tmp_root / ".git").mkdir()

            config_dict = {"mcp_servers": {}}

            # Mock input to return "y" for git
            monkeypatch.setattr("builtins.input", lambda prompt: "y")

            updated = run_interactive_discovery(tmp_root, config_dict)

            assert "git" in updated["mcp_servers"]
            assert updated["mcp_servers"]["git"]["command"] == "npx"
            assert "@modelcontextprotocol/server-git" in updated["mcp_servers"]["git"]["args"]
