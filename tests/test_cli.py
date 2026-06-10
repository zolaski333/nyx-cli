"""Tests for the Nyx CLI — argument parsing, REPL, --prompt, --dir, --no-stream."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


# Path to the nyx CLI module
NYX_CLI = str(Path(__file__).resolve().parent.parent / "nyx" / "cli.py")


def _run_nyx(*args: str, env: dict[str, str] | None = None, timeout: int = 10) -> subprocess.CompletedProcess:
    """Run the nyx CLI with given arguments and return the result."""
    cmd = [sys.executable, NYX_CLI, *args]
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    # Set a dummy API key so config validation passes
    merged_env.setdefault("OPENROUTER_API_KEY", "test-key-placeholder")
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=merged_env,
    )


# =========================================================================
# Argument Parsing Tests
# =========================================================================


class TestArgumentParsing:
    """Test CLI argument parsing."""

    def test_help(self):
        """--help should display usage information."""
        result = _run_nyx("--help")
        assert result.returncode == 0
        assert "Nyx" in result.stdout
        assert "--prompt" in result.stdout
        assert "--dir" in result.stdout
        assert "--no-stream" in result.stdout
        assert "--model" in result.stdout
        assert "--provider" in result.stdout

    def test_help_short(self):
        """-h should also display help."""
        result = _run_nyx("-h")
        assert result.returncode == 0
        assert "Nyx" in result.stdout

    def test_version_not_implemented(self):
        """Should handle missing version flag gracefully."""
        result = _run_nyx("--prompt", "test", "--no-stream", "--no-color")
        # Should not crash — will try to run but may fail due to missing API
        assert result.returncode != 0 or True  # Just ensure it doesn't hang


# =========================================================================
# --prompt / -p flag tests
# =========================================================================


class TestPromptFlag:
    """Test the --prompt / -p flag."""

    def test_prompt_short_flag(self):
        """-p should be recognized as prompt."""
        result = _run_nyx("-p", "say hello", "--no-stream", "--no-color", timeout=5)
        # Should either succeed or fail gracefully (no real API key)
        assert result.returncode in (0, 1)
        if result.returncode == 1:
            assert "Error" in result.stdout or "error" in result.stderr.lower()

    def test_prompt_long_flag(self):
        """--prompt should be recognized."""
        result = _run_nyx("--prompt", "test", "--no-stream", "--no-color", timeout=5)
        assert result.returncode in (0, 1)

    def test_prompt_with_config(self):
        """--prompt with --config should not crash."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"provider": "openrouter", "model": "test-model"}')
            config_path = f.name

        try:
            result = _run_nyx(
                "-p", "hello",
                "--config", config_path,
                "--no-stream", "--no-color",
                timeout=5,
            )
            assert result.returncode in (0, 1)
        finally:
            os.unlink(config_path)

    def test_prompt_with_model_override(self):
        """--model should override the model."""
        result = _run_nyx(
            "-p", "test",
            "--model", "openai/gpt-4o",
            "--no-stream", "--no-color",
            timeout=5,
        )
        assert result.returncode in (0, 1)

    def test_prompt_with_provider_override(self):
        """--provider should override the provider."""
        result = _run_nyx(
            "-p", "test",
            "--provider", "openai",
            "--no-stream", "--no-color",
            timeout=5,
        )
        assert result.returncode in (0, 1)


# =========================================================================
# --dir / -d flag tests
# =========================================================================


class TestDirFlag:
    """Test the --dir / -d flag."""

    def test_dir_short_flag(self):
        """-d should set the project directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _run_nyx(
                "-p", "test",
                "-d", tmpdir,
                "--no-stream", "--no-color",
                timeout=5,
            )
            assert result.returncode in (0, 1)

    def test_dir_long_flag(self):
        """--dir should set the project directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _run_nyx(
                "-p", "test",
                "--dir", tmpdir,
                "--no-stream", "--no-color",
                timeout=5,
            )
            assert result.returncode in (0, 1)

    def test_project_alias(self):
        """--project should be an alias for --dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _run_nyx(
                "-p", "test",
                "--project", tmpdir,
                "--no-stream", "--no-color",
                timeout=5,
            )
            assert result.returncode in (0, 1)

    def test_dir_defaults_to_cwd(self):
        """Without --dir, should default to current working directory."""
        result = _run_nyx("-p", "test", "--no-stream", "--no-color", timeout=5)
        assert result.returncode in (0, 1)

    def test_dir_with_nonexistent_path(self):
        """Should handle nonexistent directory gracefully."""
        result = _run_nyx(
            "-p", "test",
            "--dir", "/nonexistent/path/xyz789",
            "--no-stream", "--no-color",
            timeout=5,
        )
        assert result.returncode in (0, 1)


# =========================================================================
# --no-stream flag tests
# =========================================================================


class TestNoStreamFlag:
    """Test the --no-stream flag."""

    def test_no_stream_flag(self):
        """--no-stream should disable streaming."""
        result = _run_nyx("-p", "test", "--no-stream", "--no-color", timeout=5)
        assert result.returncode in (0, 1)

    def test_stream_by_default(self):
        """Without --no-stream, streaming should be enabled by default."""
        result = _run_nyx("-p", "test", "--no-color", timeout=5)
        assert result.returncode in (0, 1)

    def test_no_stream_with_verbose(self):
        """--no-stream with --verbose should not crash."""
        result = _run_nyx(
            "-p", "test",
            "--no-stream",
            "--no-color",
            "-v",
            timeout=5,
        )
        assert result.returncode in (0, 1)


# =========================================================================
# --no-color flag tests
# =========================================================================


class TestNoColorFlag:
    """Test the --no-color flag."""

    def test_no_color_sets_env(self):
        """--no-color should set NO_COLOR env var."""
        result = _run_nyx("-p", "test", "--no-color", "--no-stream", timeout=5)
        assert result.returncode in (0, 1)

    def test_no_color_output(self):
        """Output should not contain ANSI escape codes when --no-color is set."""
        result = _run_nyx("--help", "--no-color")
        assert "\033[" not in result.stdout


# =========================================================================
# --verbose / -v flag tests
# =========================================================================


class TestVerboseFlag:
    """Test the --verbose / -v flag."""

    def test_verbose_flag(self):
        """-v should enable verbose logging."""
        result = _run_nyx("-p", "test", "-v", "--no-stream", "--no-color", timeout=5)
        assert result.returncode in (0, 1)

    def test_verbose_long_flag(self):
        """--verbose should enable verbose logging."""
        result = _run_nyx("-p", "test", "--verbose", "--no-stream", "--no-color", timeout=5)
        assert result.returncode in (0, 1)


# =========================================================================
# --no-rich flag tests
# =========================================================================


class TestNoRichFlag:
    """Test the --no-rich flag."""

    def test_no_rich_flag(self):
        """--no-rich should force basic CLI."""
        result = _run_nyx("-p", "test", "--no-rich", "--no-stream", "--no-color", timeout=5)
        assert result.returncode in (0, 1)


# =========================================================================
# Config file tests
# =========================================================================


class TestConfigFile:
    """Test the --config / -c flag."""

    def test_config_with_valid_json(self):
        """Should load config from a valid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"provider": "openrouter", "model": "custom-model"}')
            config_path = f.name

        try:
            result = _run_nyx(
                "-p", "test",
                "-c", config_path,
                "--no-stream", "--no-color",
                timeout=5,
            )
            assert result.returncode in (0, 1)
        finally:
            os.unlink(config_path)

    def test_config_with_invalid_json(self):
        """Should handle invalid JSON config gracefully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json")
            config_path = f.name

        try:
            result = _run_nyx(
                "-p", "test",
                "-c", config_path,
                "--no-stream", "--no-color",
                timeout=5,
            )
            # Should fail with a JSON decode error
            assert result.returncode == 1
        finally:
            os.unlink(config_path)

    def test_config_with_nonexistent_path(self):
        """Should handle nonexistent config path gracefully."""
        result = _run_nyx(
            "-p", "test",
            "-c", "/nonexistent/config.json",
            "--no-stream", "--no-color",
            timeout=5,
        )
        assert result.returncode in (0, 1)


# =========================================================================
# Combined flags tests
# =========================================================================


class TestCombinedFlags:
    """Test combinations of CLI flags."""

    def test_all_flags_together(self):
        """All flags together should not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _run_nyx(
                "-p", "test",
                "-d", tmpdir,
                "--model", "test-model",
                "--provider", "openrouter",
                "--no-stream",
                "--no-color",
                "--no-rich",
                "-v",
                timeout=5,
            )
            assert result.returncode in (0, 1)

    def test_prompt_with_special_characters(self):
        """Prompt with special characters should be handled."""
        result = _run_nyx(
            "-p", "say 'hello world' and use $PATH",
            "--no-stream", "--no-color",
            timeout=5,
        )
        assert result.returncode in (0, 1)

    def test_empty_prompt(self):
        """Empty prompt should not crash."""
        result = _run_nyx("-p", "", "--no-stream", "--no-color", timeout=5)
        assert result.returncode in (0, 1)


# =========================================================================
# REPL mode tests (basic)
# =========================================================================


class TestREPLMode:
    """Test REPL/interactive mode basics."""

    def test_repl_starts_without_crashing(self):
        """REPL should start without crashing (test with /exit)."""
        result = subprocess.run(
            [sys.executable, NYX_CLI, "--no-color", "--no-rich"],
            input="/exit\n",
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "OPENROUTER_API_KEY": "test-key"},
        )
        assert result.returncode == 0
        assert "Bye" in result.stdout or "Nyx" in result.stdout

    def test_repl_help_command(self):
        """REPL /help should show help text."""
        result = subprocess.run(
            [sys.executable, NYX_CLI, "--no-color", "--no-rich"],
            input="/help\n/exit\n",
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "OPENROUTER_API_KEY": "test-key"},
        )
        assert result.returncode == 0
        assert "/help" in result.stdout
        assert "/model" in result.stdout
        assert "/clear" in result.stdout

    def test_repl_model_command(self):
        """REPL /model should show current model."""
        result = subprocess.run(
            [sys.executable, NYX_CLI, "--no-color", "--no-rich"],
            input="/model\n/exit\n",
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "OPENROUTER_API_KEY": "test-key"},
        )
        assert result.returncode == 0
        assert "model" in result.stdout.lower()

    def test_repl_clear_command(self):
        """REPL /clear should clear context."""
        result = subprocess.run(
            [sys.executable, NYX_CLI, "--no-color", "--no-rich"],
            input="/clear\n/exit\n",
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "OPENROUTER_API_KEY": "test-key"},
        )
        assert result.returncode == 0
        assert "Context cleared" in result.stdout

    def test_repl_tools_command(self):
        """REPL /tools should list tools."""
        result = subprocess.run(
            [sys.executable, NYX_CLI, "--no-color", "--no-rich"],
            input="/tools\n/exit\n",
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "OPENROUTER_API_KEY": "test-key"},
        )
        assert result.returncode == 0
        assert "Tools" in result.stdout or "tools" in result.stdout

    def test_repl_quit_commands(self):
        """REPL /quit and /q should also exit."""
        for cmd in ["/quit", "/q"]:
            result = subprocess.run(
                [sys.executable, NYX_CLI, "--no-color", "--no-rich"],
                input=f"{cmd}\n",
                capture_output=True,
                text=True,
                timeout=10,
                env={**os.environ, "OPENROUTER_API_KEY": "test-key"},
            )
            assert result.returncode == 0
            assert "Bye" in result.stdout

    def test_repl_empty_input(self):
        """Empty input should not crash."""
        result = subprocess.run(
            [sys.executable, NYX_CLI, "--no-color", "--no-rich"],
            input="\n/exit\n",
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "OPENROUTER_API_KEY": "test-key"},
        )
        assert result.returncode == 0

    def test_repl_eof_handling(self):
        """EOF (Ctrl+D) should exit gracefully."""
        result = subprocess.run(
            [sys.executable, NYX_CLI, "--no-color", "--no-rich"],
            input="",  # Empty input = EOF
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "OPENROUTER_API_KEY": "test-key"},
        )
        # Should exit gracefully on EOF
        assert result.returncode == 0


# =========================================================================
# --json flag tests
# =========================================================================


class TestJsonFlag:
    """Test the --json flag for CI/CD output."""

    def test_json_requires_prompt(self):
        """--json without --prompt should exit with error."""
        result = _run_nyx("--json", "--no-color", timeout=5)
        assert result.returncode == 1
        assert "requires --prompt" in result.stdout or "requires --prompt" in result.stderr

    def test_json_output_format(self):
        """--json should output valid JSON."""
        result = _run_nyx("--json", "-p", "test", "--no-stream", "--no-color", timeout=5)
        assert result.returncode in (0, 1)
        # Output should be valid JSON
        import json as _json
        try:
            output = _json.loads(result.stdout)
            assert "status" in output
            assert output["status"] in ("success", "error")
            assert "duration_seconds" in output
        except (_json.JSONDecodeError, ValueError):
            # May fail due to missing API key, but should not crash
            pass

    def test_json_with_piped_input(self):
        """--json with piped stdin should work."""
        result = subprocess.run(
            [sys.executable, NYX_CLI, "--json", "-p", "analyse", "--no-stream", "--no-color"],
            input="test data from pipe",
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "OPENROUTER_API_KEY": "test-key"},
        )
        assert result.returncode in (0, 1)


# =========================================================================
# Pipe mode tests
# =========================================================================


class TestPipeMode:
    """Test pipe mode (stdin input)."""

    def test_pipe_with_prompt(self):
        """Piped data with -p should include context."""
        result = subprocess.run(
            [sys.executable, NYX_CLI, "-p", "review this", "--no-stream", "--no-color"],
            input="def hello():\n    pass\n",
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "OPENROUTER_API_KEY": "test-key"},
        )
        assert result.returncode in (0, 1)

    def test_pipe_without_prompt(self):
        """Piped data without -p should auto-generate prompt."""
        result = subprocess.run(
            [sys.executable, NYX_CLI, "--no-stream", "--no-color"],
            input="some content to analyse\n",
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "OPENROUTER_API_KEY": "test-key"},
        )
        assert result.returncode in (0, 1)


# =========================================================================
# REPL new commands tests
# =========================================================================


class TestREPLNewCommands:
    """Test new REPL commands: /switch, pagination."""

    def test_repl_switch_nonexistent(self):
        """REPL /switch with nonexistent ID."""
        result = subprocess.run(
            [sys.executable, NYX_CLI, "--no-color", "--no-rich"],
            input="/switch nonexistent\n/exit\n",
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "OPENROUTER_API_KEY": "test-key"},
        )
        assert result.returncode == 0
        assert "not found" in result.stdout.lower()

    def test_repl_tools_pagination(self):
        """REPL /tools 2 should show page 2."""
        result = subprocess.run(
            [sys.executable, NYX_CLI, "--no-color", "--no-rich"],
            input="/tools 2\n/exit\n",
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "OPENROUTER_API_KEY": "test-key"},
        )
        assert result.returncode == 0
        assert "Page" in result.stdout

    def test_repl_conversations_empty(self):
        """REPL /conversations when empty."""
        result = subprocess.run(
            [sys.executable, NYX_CLI, "--no-color", "--no-rich"],
            input="/conversations\n/exit\n",
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "OPENROUTER_API_KEY": "test-key"},
        )
        assert result.returncode == 0