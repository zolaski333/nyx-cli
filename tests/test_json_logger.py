"""Tests for the JSON logger — session tracking, cost estimation, tool/LLM logging."""
from __future__ import annotations

import json
import os
import tempfile
import time


from nyx.json_logger import JSONLogger, estimate_cost, JSONLogEntry


class TestEstimateCost:
    """Test cost estimation for different models."""

    def test_known_model_cost(self):
        """Should estimate cost for a known model."""
        cost = estimate_cost("openai/gpt-4o", input_tokens=1000, output_tokens=500)
        assert cost > 0
        # 1000 input * 0.0025/1k + 500 output * 0.01/1k = 0.0025 + 0.005 = 0.0075
        assert abs(cost - 0.0075) < 0.001

    def test_unknown_model_cost(self):
        """Should use default rates for unknown models."""
        cost = estimate_cost("unknown/model", input_tokens=1000, output_tokens=1000)
        assert cost > 0
        # Uses DEFAULT_INPUT_COST and DEFAULT_OUTPUT_COST
        assert cost == (1000 / 1000 * 0.0002) + (1000 / 1000 * 0.0008)

    def test_zero_tokens(self):
        """Zero tokens should cost zero."""
        cost = estimate_cost("openai/gpt-4o", input_tokens=0, output_tokens=0)
        assert cost == 0.0

    def test_deepseek_cost(self):
        """DeepSeek model cost calculation."""
        cost = estimate_cost("deepseek/deepseek-v4-flash", input_tokens=2000, output_tokens=1000)
        # 2000 * 0.00015/1k + 1000 * 0.00060/1k = 0.0003 + 0.0006 = 0.0009
        assert abs(cost - 0.0009) < 0.0001


class TestJSONLogEntry:
    """Test JSON log entry creation and serialization."""

    def test_to_dict(self):
        """Should convert entry to dict."""
        entry = JSONLogEntry(
            timestamp=1234567890.0,
            event_type="llm_call",
            session_id="test-session",
            data={"model": "gpt-4o", "tokens": 100},
        )
        d = entry.to_dict()
        assert d["timestamp"] == 1234567890.0
        assert d["event_type"] == "llm_call"
        assert d["session_id"] == "test-session"
        assert d["data"]["model"] == "gpt-4o"
        assert d["level"] == "info"

    def test_to_json(self):
        """Should serialize entry to JSON string."""
        entry = JSONLogEntry(
            timestamp=1234567890.0,
            event_type="tool_call",
            session_id="sess-1",
            data={"tool": "read_file"},
        )
        parsed = json.loads(entry.to_json())
        assert parsed["event_type"] == "tool_call"
        assert parsed["session_id"] == "sess-1"

    def test_error_level(self):
        """Should support error level entries."""
        entry = JSONLogEntry(
            timestamp=time.time(),
            event_type="error",
            session_id="sess-1",
            data={"message": "something broke"},
            level="error",
        )
        assert entry.level == "error"
        parsed = json.loads(entry.to_json())
        assert parsed["level"] == "error"


class TestJSONLogger:
    """Test the JSON logger with file output."""

    def test_logger_disabled(self):
        """Disabled logger should not write anything."""
        logger = JSONLogger(enabled=False)
        logger.log_llm_call(input_tokens=100, output_tokens=50)
        assert logger.total_llm_calls == 0
        assert logger.total_cost == 0.0
        assert logger.total_input_tokens == 0
        assert logger.total_output_tokens == 0

    def test_logger_session_id(self):
        """Should auto-generate a session id if not provided."""
        logger = JSONLogger(enabled=True)
        assert len(logger.session_id) == 12
        logger.close()

    def test_logger_custom_session_id(self):
        """Should use provided session id."""
        logger = JSONLogger(enabled=True, session_id="my-custom-session")
        assert logger.session_id == "my-custom-session"
        logger.close()

    def test_log_llm_call_tracks_cumulative(self):
        """Should track cumulative tokens and cost across calls."""
        logger = JSONLogger(enabled=True, model="openai/gpt-4o")
        logger.log_llm_call(input_tokens=1000, output_tokens=500)
        assert logger.total_llm_calls == 1
        assert logger.total_input_tokens == 1000
        assert logger.total_output_tokens == 500
        assert logger.total_cost > 0

        logger.log_llm_call(input_tokens=500, output_tokens=250)
        assert logger.total_llm_calls == 2
        assert logger.total_input_tokens == 1500
        assert logger.total_output_tokens == 750
        logger.close()

    def test_log_tool_call(self):
        """Should log tool calls and track count."""
        logger = JSONLogger(enabled=True)
        logger.log_tool_call(tool_name="read_file", arguments={"path": "/tmp/test"})
        assert logger.total_tool_calls == 1
        logger.close()

    def test_log_tool_result(self):
        """Should log tool results."""
        logger = JSONLogger(enabled=True)
        logger.log_tool_result(tool_name="read_file", result="file content", duration_ms=10.5)
        assert logger.total_tool_calls == 0  # tool_result doesn't increment tool call count
        logger.close()

    def test_log_error(self):
        """Should log errors and track count."""
        logger = JSONLogger(enabled=True)
        logger.log_error(source="test", message="something went wrong")
        assert logger.total_errors == 1
        logger.close()

    def test_log_event(self):
        """Should log arbitrary events."""
        logger = JSONLogger(enabled=True)
        logger.log_event("custom_event", {"key": "value"})
        logger.close()

    def test_writes_to_file(self):
        """Should write NDJSON to the specified file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ndjson", delete=False) as f:
            tmp_path = f.name

        try:
            logger = JSONLogger(output_path=tmp_path, enabled=True, model="test-model")
            logger.log_llm_call(input_tokens=100, output_tokens=50)
            logger.log_tool_call(tool_name="test_tool", arguments={"arg": 1})
            logger.log_error(source="test", message="error msg")
            logger.close()

            with open(tmp_path, "r") as f:
                lines = f.readlines()

            # session_start + 3 events + session_end = 5 lines
            assert len(lines) == 5

            # Verify session_start
            first = json.loads(lines[0])
            assert first["event_type"] == "session_start"
            assert first["data"]["model"] == "test-model"

            # Verify llm_call
            second = json.loads(lines[1])
            assert second["event_type"] == "llm_call"
            assert second["data"]["input_tokens"] == 100

            # Verify session_end
            last = json.loads(lines[-1])
            assert last["event_type"] == "session_end"
            assert "total_cost" in last["data"]
        finally:
            os.unlink(tmp_path)

    def test_session_summary(self):
        """Should return accurate session summary."""
        logger = JSONLogger(enabled=True, model="openai/gpt-4o")
        logger.log_llm_call(input_tokens=1000, output_tokens=500)
        logger.log_tool_call(tool_name="tool1", arguments={})
        logger.log_error(source="test", message="err")

        summary = logger.get_session_summary()
        assert summary["total_llm_calls"] == 1
        assert summary["total_tool_calls"] == 1
        assert summary["total_errors"] == 1
        assert summary["total_input_tokens"] == 1000
        assert summary["total_output_tokens"] == 500
        assert summary["total_cost"] > 0
        assert summary["session_id"] == logger.session_id
        logger.close()

    def test_log_to_stderr_capsys(self, capsys):
        """Should write to stderr when log_to_stderr is True."""
        logger = JSONLogger(enabled=True, log_to_stderr=True)
        logger.log_llm_call(input_tokens=10, output_tokens=5)
        logger.close()

        captured = capsys.readouterr()
        # Should have written to stderr
        assert "llm_call" in captured.err
        assert "session_start" in captured.err
        assert "session_end" in captured.err

    def test_logger_reentrant(self):
        """Multiple log calls should not deadlock."""
        logger = JSONLogger(enabled=True)
        for _ in range(100):
            logger.log_llm_call(input_tokens=10, output_tokens=5)
            logger.log_tool_call(tool_name="t", arguments={})
        assert logger.total_llm_calls == 100
        assert logger.total_tool_calls == 100
        logger.close()