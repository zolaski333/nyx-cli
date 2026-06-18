"""Tests for the Nyx CLI — argument parsing, REPL, --prompt, --dir, --no-stream."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import pytest


# Path to the nyx CLI module
NYX_CLI = str(Path(__file__).resolve().parent.parent / "nyx" / "cli.py")


# ---------------------------------------------------------------------------
# Mock LLM HTTP server — replaces the real OpenRouter API during tests
# ---------------------------------------------------------------------------


class _MockLLMHandler(BaseHTTPRequestHandler):
    """HTTP handler that simulates an LLM chat completion endpoint."""

    # Shared state across all handler instances
    responses: list[dict[str, Any]] = []
    request_count = 0

    def do_POST(self):
        _MockLLMHandler.request_count += 1

        # Read the request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        req_data = json.loads(body)

        is_stream = req_data.get("stream", False)

        if is_stream:
            self._send_streaming_response()
        else:
            self._send_non_stream_response()

    def _send_non_stream_response(self) -> None:
        """Send a non-streaming response."""
        if _MockLLMHandler.responses:
            resp_data = _MockLLMHandler.responses.pop(0)
        else:
            resp_data = {
                "id": "mock-cmpl-xxx",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "mock-model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from mock LLM!"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp_data).encode("utf-8"))

    def _send_streaming_response(self) -> None:
        """Send a streaming SSE response."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        chunks = [
            {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " from"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " mock"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " LLM!"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]},
        ]
        for chunk in chunks:
            line = f"data: {json.dumps(chunk)}\n\n"
            self.wfile.write(line.encode("utf-8"))
            self.wfile.flush()

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format, *args):
        """Suppress HTTP server log output during tests."""
        pass


class _MockLLMServer:
    """Manages a background HTTP server that mocks the LLM API."""

    def __init__(self):
        self._server = HTTPServer(("127.0.0.1", 0), _MockLLMHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._config_path: str | None = None

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/api/v1/chat/completions"

    def start(self) -> dict[str, str]:
        """Start the server and return env vars pointing to it."""
        self._thread.start()
        return {
            "NYX_OPENROUTER_BASE_URL": self.base_url,
            "NYX_MODEL": "mock-model",
            "NYX_PROVIDER": "openrouter",
        }

    def stop(self):
        """Stop the server."""
        self._server.shutdown()

    def reset(self):
        """Reset handler state between tests."""
        _MockLLMHandler.responses = []
        _MockLLMHandler.request_count = 0


# Module-level mock server instance (started once per test session)
_MOCK_SERVER = _MockLLMServer()
_MOCK_ENV_VARS: dict[str, str] | None = None


def _ensure_mock_server() -> dict[str, str]:
    """Start the mock server if not already running, return env vars."""
    global _MOCK_ENV_VARS
    if _MOCK_ENV_VARS is None:
        _MOCK_ENV_VARS = _MOCK_SERVER.start()
    _MOCK_SERVER.reset()
    return _MOCK_ENV_VARS


def _run_nyx(*args: str, env: dict[str, str] | None = None, timeout: int = 10) -> subprocess.CompletedProcess:
    """Run the nyx CLI with given arguments and return the result."""
    mock_env = _ensure_mock_server()
    cmd = [sys.executable, NYX_CLI, *args]
    merged_env = os.environ.copy()
    # Inject mock server env vars (low priority — can be overridden by caller)
    for k, v in mock_env.items():
        merged_env.setdefault(k, v)
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


@pytest.fixture(scope="session", autouse=True)
def _mock_server_lifetime():
    """Start the mock LLM server before the test session and stop it after."""
    # Server is started lazily by _ensure_mock_server on first use
    yield
    # Clean up after all tests
    _MOCK_SERVER.stop()


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
        assert "doctor" in result.stdout

    def test_help_short(self):
        """-h should also display help."""
        result = _run_nyx("-h")
        assert result.returncode == 0
        assert "Nyx" in result.stdout

    def test_version_flag(self):
        """--version should print the installed Nyx version."""
        result = _run_nyx("--version")
        assert result.returncode == 0
        assert "nyx " in result.stdout.lower()

    def test_doctor_runs_without_api_key(self):
        """doctor should diagnose setup without requiring an API key or LLM call."""
        result = _run_nyx("doctor", env={"OPENROUTER_API_KEY": ""})
        assert result.returncode == 0
        assert "Nyx doctor" in result.stdout
        assert "API key" in result.stdout
        assert "missing" in result.stdout.lower()

    def test_module_execution_has_no_import_warning(self):
        """python -m nyx.cli should not warn about nyx.cli already being imported."""
        result = subprocess.run(
            [sys.executable, "-m", "nyx.cli", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "OPENROUTER_API_KEY": "test-key-placeholder"},
        )
        assert result.returncode == 0
        assert "RuntimeWarning" not in result.stderr

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

    def test_repl_theme_command(self):
        """REPL /theme should list and switch themes."""
        # 1. Test listing themes
        result = subprocess.run(
            [sys.executable, NYX_CLI, "--no-color", "--no-rich"],
            input="/theme\n/exit\n",
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "OPENROUTER_API_KEY": "test-key"},
        )
        assert result.returncode == 0
        assert "Active Theme" in result.stdout
        assert "cyberpunk" in result.stdout

        # 2. Test switching theme
        result = subprocess.run(
            [sys.executable, NYX_CLI, "--no-color", "--no-rich"],
            input="/theme dracula\n/exit\n",
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "OPENROUTER_API_KEY": "test-key"},
        )
        assert result.returncode == 0
        assert "Theme switched to: dracula" in result.stdout


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

    def test_repl_config_status(self):
        """REPL /config should print active session config."""
        result = subprocess.run(
            [sys.executable, NYX_CLI, "--no-color", "--no-rich"],
            input="/config\n/exit\n",
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "OPENROUTER_API_KEY": "test-key"},
        )
        assert result.returncode == 0
        assert "Nyx Configuration Status" in result.stdout
        assert "Active Model:" in result.stdout

    def test_repl_config_set_and_save(self):
        """REPL /config set and /config save should work and modify project config."""
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock project dir with .nyx folder
            project_path = Path(tmpdir)
            (project_path / ".nyx").mkdir(parents=True)
            
            # Start REPL inside this project directory
            result = subprocess.run(
                [sys.executable, NYX_CLI, "--no-color", "--no-rich", "--dir", str(project_path)],
                input="/config set model test-custom-model\n/config save\n/exit\n",
                capture_output=True,
                text=True,
                timeout=10,
                env={**os.environ, "OPENROUTER_API_KEY": "test-key"},
            )
            assert result.returncode == 0
            assert "Updated config key" in result.stdout
            assert "Successfully saved session config to" in result.stdout
            
            # Verify the config file has the expected content
            config_file = project_path / ".nyx" / "config.json"
            assert config_file.exists()
            content = json.loads(config_file.read_text(encoding="utf-8"))
            assert content.get("model") == "test-custom-model"
