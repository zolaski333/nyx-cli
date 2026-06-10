"""
Nyx — Configuration system.

Loads config from multiple sources (config.json, env vars, CLI args)
with a strict priority chain.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.json"
EXAMPLE_CONFIG_PATH = PROJECT_DIR / "config.example.json"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "provider": "openrouter",
    "model": "deepseek/deepseek-v4-flash",
    "system_prompt": "You are a powerful agentic CLI assistant. You have access to tools, skills, MCP servers, subagents, and web search. Be concise, precise, and helpful. IMPORTANT: Do NOT use Markdown formatting in your responses. The terminal does not support Markdown rendering. Use plain text only (no **bold**, no `code`, no |tables|, no ## headings, no ```code blocks).",
    "site_url": "",
    "site_name": "Nyx",
    "request_timeout": 120,
    "stream": True,
    "max_tokens": 4096,
    "temperature": 0.7,
    "mcp_servers": {},
    "skills_dir": "skills",
    "subagents_dir": "subagents",
    "web_search_enabled": True,
    "web_search_provider": "duckduckgo",
    "openrouter_base_url": "https://openrouter.ai/api/v1/chat/completions",
    "openai_base_url": "https://api.openai.com/v1/chat/completions",
    "anthropic_base_url": "https://api.anthropic.com/v1/messages",
    # -- Security / sandbox --
    "project_dir": "",
    "sandbox": {
        "enabled": True,
        "auto_chdir": True,
        "allow_paths": [],
        "deny_paths": [],
    },
    # -- Permissions --
    "permissions": {
        "shell": {
            "default": "allow",
            "rules": [],
        },
        "filesystem": {
            "default": "allow",
            "rules": [],
        },
    },
    # -- Audit trail --
    "audit": {
        "enabled": True,
        "output_dir": "",
        "max_file_size_mb": 50,
    },
    # -- JSON logging --
    "json_logging": {
        "enabled": False,
        "output_path": "",
        "log_to_stderr": False,
    },
    # -- Rate limiting --
    # Tuned for fast models (DeepSeek V4 Flash). The local token bucket
    # should not be a bottleneck — real rate limits are enforced server-side.
    "rate_limiting": {
        "enabled": True,
        "rate": 100.0,       # was 10.0 — allow up to 100 req/s locally
        "burst": 50,         # was 20  — allow bursts of 50
        "max_retries": 1,    # was 3   — only 1 retry, no exponential backoff spiral
        "base_delay": 0.5,   # was 1.0 — start retry after 0.5s
        "max_delay": 10.0,   # was 60.0 — never wait more than 10s for retry
        "request_timeout": 120,
    },
    # -- Diff/patch tool --
    "diff_tool": {
        "require_approval": True,
        "show_full_diff": True,
    },
}


# ---------------------------------------------------------------------------
# ConfigError
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """Raised when configuration is invalid or missing."""


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Immutable-like configuration object."""

    provider: str = "openrouter"
    model: str = "deepseek/deepseek-v4-flash"
    system_prompt: str = DEFAULT_CONFIG["system_prompt"]
    site_url: str = ""
    site_name: str = "Nyx"
    request_timeout: int = 120
    stream: bool = True
    max_tokens: int = 4096
    temperature: float = 0.7
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    skills_dir: str = ""
    subagents_dir: str = ""
    project_dir: str = ""
    web_search_enabled: bool = True
    web_search_provider: str = "duckduckgo"
    openrouter_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    openrouter_base_url: str = DEFAULT_CONFIG["openrouter_base_url"]
    openai_base_url: str = DEFAULT_CONFIG["openai_base_url"]
    anthropic_base_url: str = DEFAULT_CONFIG["anthropic_base_url"]
    # -- Security / sandbox --
    sandbox_enabled: bool = True
    sandbox_auto_chdir: bool = True
    sandbox_allow_paths: list[str] = field(default_factory=list)
    sandbox_deny_paths: list[str] = field(default_factory=list)
    # -- Permissions --
    permissions_config: dict[str, Any] = field(default_factory=dict)
    # -- Audit trail --
    audit_enabled: bool = True
    audit_output_dir: str = ""
    audit_max_file_size_mb: int = 50
    # -- JSON logging --
    json_logging_enabled: bool = False
    json_logging_output_path: str = ""
    json_logging_log_to_stderr: bool = False
    # -- Rate limiting --
    rate_limiting_enabled: bool = True
    rate_limiting_rate: float = 10.0
    rate_limiting_burst: int = 20
    rate_limiting_max_retries: int = 3
    rate_limiting_base_delay: float = 1.0
    rate_limiting_max_delay: float = 60.0
    # -- Diff/patch tool --
    diff_tool_require_approval: bool = True
    diff_tool_show_full_diff: bool = True
    # -- Raw --
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Load configuration with priority: env vars > config.json > defaults."""
        path = Path(path) if path else DEFAULT_CONFIG_PATH

        # 1. Start with defaults
        raw: dict[str, Any] = dict(DEFAULT_CONFIG)

        # 2. Overlay config.json (or config.example.json as fallback)
        if not path.exists():
            # Try config.example.json as a fallback
            if EXAMPLE_CONFIG_PATH.exists():
                logger.info("No config.json found, using config.example.json as template")
                path = EXAMPLE_CONFIG_PATH

        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                file_config = json.load(f)
            raw.update(file_config)
            logger.info("Loaded config from %s", path)
        else:
            logger.info("No config file at %s, using defaults", path)

        # 3. Environment variables override (highest priority)
        env_map: dict[str, str] = {
            "OPENROUTER_API_KEY": "openrouter_api_key",
            "OPENAI_API_KEY": "openai_api_key",
            "ANTHROPIC_API_KEY": "anthropic_api_key",
            "NYX_MODEL": "model",
            "NYX_PROVIDER": "provider",
            "NYX_OPENROUTER_BASE_URL": "openrouter_base_url",
        }
        for env_key, config_key in env_map.items():
            val = os.environ.get(env_key, "").strip()
            if val:
                raw[config_key] = val
                logger.debug("Override %s from env var %s", config_key, env_key)

        # 4. Validate required keys
        provider = raw.get("provider", "openrouter")
        api_key = raw.get(f"{provider}_api_key", "")
        if not api_key:
            raise ConfigError(
                f"No API key found for provider '{provider}'.\n"
                f"Set {provider.upper()}_API_KEY env var or add it to config.json."
            )

        # 5. Flatten nested config sections into top-level fields
        cls._flatten_nested(raw, "sandbox", ["enabled", "auto_chdir", "allow_paths", "deny_paths"])
        cls._flatten_nested(raw, "permissions", ["shell", "filesystem"])
        cls._flatten_nested(raw, "audit", ["enabled", "output_dir", "max_file_size_mb"])
        cls._flatten_nested(raw, "json_logging", ["enabled", "output_path", "log_to_stderr"])
        cls._flatten_nested(raw, "rate_limiting", ["enabled", "rate", "burst", "max_retries", "base_delay", "max_delay"])
        cls._flatten_nested(raw, "diff_tool", ["require_approval", "show_full_diff"])

        # Store a copy of raw config (not self-referencing)
        raw_copy = dict(raw)
        config = cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
        config.raw = raw_copy
        return config

    @staticmethod
    def _flatten_nested(raw: dict, prefix: str, keys: list[str]) -> None:
        """Flatten nested config sections into top-level prefixed fields."""
        section = raw.get(prefix, {})
        if isinstance(section, dict):
            for key in keys:
                field_name = f"{prefix}_{key}"
                if field_name not in raw and key in section:
                    raw[field_name] = section[key]
            # Keep the original nested config for reference
            if prefix == "permissions":
                raw["permissions_config"] = section

    def get_api_key(self) -> str:
        return getattr(self, f"{self.provider}_api_key", "")

    def get_base_url(self) -> str:
        return getattr(self, f"{self.provider}_base_url", "")

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "raw"}