"""Tests for Nyx configuration system."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from nyx.config import Config, ConfigError, DEFAULT_CONFIG


class TestConfig:
    """Test configuration loading and validation."""

    def test_defaults(self):
        """Config with no file should use defaults (and fail on missing API key)."""
        # Temporarily remove API key from env to test the error
        old_key = os.environ.pop("OPENROUTER_API_KEY", None)
        try:
            with pytest.raises(ConfigError, match="API key"):
                Config.load()
        finally:
            if old_key is not None:
                os.environ["OPENROUTER_API_KEY"] = old_key

    def test_load_from_file(self):
        """Load config from a JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"provider": "openai", "model": "gpt-4o"}, f)
            tmp_path = f.name

        try:
            with pytest.raises(ConfigError, match="API key"):
                Config.load(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_env_var_override(self):
        """Environment variables should override file config."""
        os.environ["OPENROUTER_API_KEY"] = "sk-test-key"
        os.environ["NYX_MODEL"] = "test-model"

        try:
            config = Config.load()
            assert config.model == "test-model"
            assert config.openrouter_api_key == "sk-test-key"
        finally:
            del os.environ["OPENROUTER_API_KEY"]
            del os.environ["NYX_MODEL"]

    def test_get_api_key(self):
        """get_api_key should return the key for the configured provider."""
        config = Config(provider="openai", openai_api_key="sk-test")
        assert config.get_api_key() == "sk-test"

    def test_get_base_url(self):
        """get_base_url should return the URL for the configured provider."""
        config = Config(provider="openrouter")
        assert "openrouter" in config.get_base_url()

    def test_to_dict(self):
        """to_dict should not include 'raw' field."""
        config = Config(openrouter_api_key="sk-test")
        d = config.to_dict()
        assert "raw" not in d
        assert d["provider"] == "openrouter"

    def test_raw_copy_not_self_referencing(self):
        """raw should be a copy, not a self-reference (when loaded via Config.load)."""
        os.environ["OPENROUTER_API_KEY"] = "sk-test-raw"
        try:
            config = Config.load()
            assert config.raw is not config.__dict__
            # raw should contain the config keys but not be self-referencing
            assert config.raw.get("provider") == "openrouter"
            assert "raw" not in config.raw  # no circular reference
        finally:
            del os.environ["OPENROUTER_API_KEY"]

    def test_missing_api_key_raises(self):
        """Missing API key should raise ConfigError."""
        with pytest.raises(ConfigError, match="API key"):
            Config.load()

    def test_invalid_provider(self):
        """An unknown provider should still load but fail at provider factory."""
        config = Config(provider="unknown_provider", openrouter_api_key="sk-test")
        assert config.provider == "unknown_provider"