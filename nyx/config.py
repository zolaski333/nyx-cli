"""
Nyx — Configuration system.

Loads config from multiple sources (config.json, env vars, CLI args)
with a strict priority chain.
"""
from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.json"
EXAMPLE_CONFIG_PATH = PROJECT_DIR / "config.example.json"
DEFAULT_PROJECT_CONFIG_PATH = Path.cwd() / ".nyx" / "config.json"
DEFAULT_USER_CONFIG_PATH = (
    Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "nyx" / "config.json"
    if os.name == "nt"
    else Path.home() / ".config" / "nyx" / "config.json"
)
TRUST_REGISTRY_PATH = Path.home() / ".nyx" / "trusted_workspaces.json"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Mode system prompts
# ---------------------------------------------------------------------------

_CHAT_SUFFIX = ""

_CODE_SUFFIX = """\n\n[MODE: CODE]
You are an expert software developer. Your primary focus is writing, modifying and refactoring code.
Guidelines:
- Prefer apply_diff for modifying existing files (shows a clean diff).
- Use write_file only for new files or full rewrites.
- Always read a file before modifying it.
- Think step-by-step: understand the structure, then make minimal targeted changes.
- Run tests after each significant change using run_tests.
- Use search_code to navigate large codebases efficiently."""

_ARCHITECT_SUFFIX = """\n\n[MODE: ARCHITECT — READ ONLY]
You are a senior software architect. You analyse codebases and produce clear, actionable plans.
IMPORTANT CONSTRAINTS:
- You CANNOT modify, create, or delete any file. You have no write tools.
- Your sole job is to READ, UNDERSTAND and PLAN.
- Use repo_map to get an overview, then read_file and search_code for details.
- Produce a structured plan (problem statement, proposed architecture, files to change, risks).
- Be explicit about which files would need to change and why.
- Do NOT attempt to apply any changes."""

_DEBUG_SUFFIX = """\n\n[MODE: DEBUG]
You are an expert debugger. Your focus is finding and fixing bugs, errors, and test failures.
Guidelines:
- Start by reading error messages and stack traces carefully.
- Use search_code to find relevant code paths.
- Use run_tests to reproduce failures before and after fixes.
- Apply minimal, surgical fixes — avoid refactoring unrelated code.
- Explain the root cause of each bug before fixing it.
- Verify the fix with run_tests after applying it."""

# Maps mode name -> system prompt suffix injected into context
MODE_SYSTEM_PROMPTS: dict[str, str] = {
    "chat": _CHAT_SUFFIX,
    "code": _CODE_SUFFIX,
    "architect": _ARCHITECT_SUFFIX,
    "debug": _DEBUG_SUFFIX,
}

# Tools available in architect mode (read-only)
ARCHITECT_TOOLS: set[str] = {
    "read_file", "list_files", "search_code", "repo_map",
    "web_search", "web_fetch", "memory_recall", "finish",
}

# ---------------------------------------------------------------------------
# Autonomy configs
# ---------------------------------------------------------------------------

# Maps autonomy level -> dict of behaviour overrides
AUTONOMY_CONFIGS: dict[str, dict[str, Any]] = {
    "ask": {
        # Default: prompt for file writes outside sandbox, prompt for dangerous commands
        "auto_approve_files": False,
        "auto_approve_commands": False,
        "max_depth_multiplier": 1,   # × agent_max_depth
    },
    "auto": {
        # Auto-approve file writes, still prompt for dangerous shell commands
        "auto_approve_files": True,
        "auto_approve_commands": False,
        "max_depth_multiplier": 2,
    },
    "yolo": {
        # Auto-approve everything (except hard-DENY rules)
        "auto_approve_files": True,
        "auto_approve_commands": True,
        "max_depth_multiplier": 4,
    },
}


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "provider": "openrouter",
    "model": "deepseek/deepseek-v4-flash",
    "system_prompt": "You are a powerful agentic CLI assistant. You have access to tools, skills, MCP servers, subagents, and web search. Be concise, precise, and helpful. Feel free to use Markdown formatting in your responses (bold, lists, code blocks) as the terminal is capable of rendering it.",
    "theme": "cyberpunk",
    "site_url": "",
    "site_name": "Nyx",
    "request_timeout": 120,
    "stream": True,
    "max_tokens": 4096,
    "temperature": 0.7,
    "mcp_servers": {},
    "mcp": {
        "request_timeout": 30,
        "connect_timeout": 30,
        "max_response_chars": 20000,
        "restart_on_failure": True,
        "sandbox_enabled": False,
        "sandbox_docker_image": "python:3.11-slim-buster",
        "sandbox_network": "none",
        "sandbox_read_only": False,
    },
    "skills_dir": "skills",
    "skills_enabled": True,
    "skills": {
        "process_isolation": True,
        "default_timeout_seconds": 30,
        "max_output_chars": 20000,
    },
    "subagents_dir": "subagents",
    "subagents": {
        "process_isolation": True,
        "default_timeout_seconds": 120,
    },
    "web_search_enabled": True,
    "web_search_provider": "searxng",
    "searxng_base_url": "https://searx.be/search",
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
        "use_docker": False,
        "docker_image": "python:3.11-slim-buster",
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
        "enabled": False,
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
        "enable_rollback": True,
        "enable_history": True,
        "use_git": True,
        "max_rollback_entries": 50,
    },
    # -- Agent behaviour --
    "agent": {
        "mode": "chat",        # chat | code | architect | debug
        "autonomy": "ask",     # ask | auto | yolo
        "max_depth": 50,       # maximum reasoning steps
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
    mcp_request_timeout: float = 30
    mcp_connect_timeout: float = 30
    mcp_max_response_chars: int = 20000
    mcp_restart_on_failure: bool = True
    mcp_sandbox_enabled: bool = False
    mcp_sandbox_docker_image: str = "python:3.11-slim-buster"
    mcp_sandbox_network: str = "none"
    mcp_sandbox_read_only: bool = False
    skills_dir: str = ""
    skills_enabled: bool = True
    skills_process_isolation: bool = True
    skills_default_timeout_seconds: float | None = 30
    skills_max_output_chars: int = 20000
    subagents_dir: str = ""
    subagents_process_isolation: bool = True
    subagents_default_timeout_seconds: float | None = 120
    project_dir: str = ""
    web_search_enabled: bool = True
    web_search_provider: str = "searxng"
    searxng_base_url: str = DEFAULT_CONFIG["searxng_base_url"]
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
    sandbox_use_docker: bool = False
    sandbox_docker_image: str = "python:3.11-slim-buster"
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
    rate_limiting_enabled: bool = False
    rate_limiting_rate: float = 100.0
    rate_limiting_burst: int = 50
    rate_limiting_max_retries: int = 1
    rate_limiting_base_delay: float = 0.5
    rate_limiting_max_delay: float = 10.0
    # -- Diff/patch tool --
    diff_tool_require_approval: bool = True
    diff_tool_show_full_diff: bool = True
    diff_tool_enable_rollback: bool = True
    diff_tool_enable_history: bool = True
    diff_tool_use_git: bool = True
    diff_tool_max_rollback_entries: int = 50
    # -- Agent behaviour --
    agent_mode: str = "chat"       # chat | code | architect | debug
    agent_autonomy: str = "ask"    # ask | auto | yolo
    agent_max_depth: int = 50
    theme: str = "cyberpunk"
    # -- Raw --
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Load configuration with priority: env vars > config.json > defaults."""
        # 1. Start with defaults
        raw: dict[str, Any] = deepcopy(DEFAULT_CONFIG)

        # 2. Overlay config files from least-specific to most-specific.
        config_paths = cls._config_paths(path)
        loaded_paths: list[Path] = []
        trusted_local_configs = cls._trusted_local_config_paths(config_paths)
        for cfg_path in config_paths:
            if cfg_path.exists():
                if cfg_path in trusted_local_configs and not cls._is_workspace_trusted(Path.cwd()):
                    if not cls._confirm_workspace_trust(Path.cwd(), cfg_path):
                        logger.warning(
                            "Skipping untrusted local config: %s. Using safer global/default config.",
                            cfg_path,
                        )
                        continue
                    cls.trust_workspace(Path.cwd())
                with cfg_path.open("r", encoding="utf-8") as f:
                    file_config = json.load(f)
                raw = cls._deep_merge(raw, file_config)
                loaded_paths.append(cfg_path)

        if loaded_paths:
            logger.info("Loaded config from %s", ", ".join(str(p) for p in loaded_paths))
        else:
            logger.info("No config file found, using defaults")

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
        cls._flatten_nested(raw, "sandbox", ["enabled", "auto_chdir", "allow_paths", "deny_paths", "use_docker", "docker_image"])
        cls._flatten_nested(raw, "mcp", [
            "request_timeout",
            "connect_timeout",
            "max_response_chars",
            "restart_on_failure",
            "sandbox_enabled",
            "sandbox_docker_image",
            "sandbox_network",
            "sandbox_read_only",
        ])
        cls._flatten_nested(raw, "permissions", ["shell", "filesystem"])
        cls._flatten_nested(raw, "audit", ["enabled", "output_dir", "max_file_size_mb"])
        cls._flatten_nested(raw, "json_logging", ["enabled", "output_path", "log_to_stderr"])
        cls._flatten_nested(raw, "rate_limiting", ["enabled", "rate", "burst", "max_retries", "base_delay", "max_delay"])
        cls._flatten_nested(raw, "diff_tool", ["require_approval", "show_full_diff", "enable_rollback", "enable_history", "use_git", "max_rollback_entries"])
        cls._flatten_nested(raw, "agent", ["mode", "autonomy", "max_depth"])
        cls._flatten_nested(raw, "skills", ["process_isolation", "default_timeout_seconds", "max_output_chars"])
        cls._flatten_nested(raw, "subagents", ["process_isolation", "default_timeout_seconds"])

        # Store a copy of raw config (not self-referencing)
        raw_copy = dict(raw)
        config = cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
        config.raw = raw_copy
        return config

    @staticmethod
    def _config_paths(path: str | Path | None = None) -> list[Path]:
        """Return config paths in merge order."""
        if path:
            return [Path(path)]
        candidates = [
            DEFAULT_CONFIG_PATH,
            DEFAULT_USER_CONFIG_PATH,
            Path.cwd() / ".nyx" / "config.json",
            Path.cwd() / "config.json",
        ]
        seen: set[Path] = set()
        ordered: list[Path] = []
        for candidate in candidates:
            resolved = candidate.expanduser()
            if resolved not in seen:
                ordered.append(resolved)
                seen.add(resolved)
        return ordered

    @staticmethod
    def _trusted_local_config_paths(config_paths: list[Path]) -> set[Path]:
        """Return default workspace-local config paths that need workspace trust."""
        local_candidates = {
            (Path.cwd() / ".nyx" / "config.json").expanduser().resolve(),
            (Path.cwd() / "config.json").expanduser().resolve(),
        }
        trusted: set[Path] = set()
        for path in config_paths:
            try:
                resolved = path.expanduser().resolve()
            except OSError:
                resolved = path.expanduser()
            if resolved in local_candidates:
                trusted.add(path)
        return trusted

    @staticmethod
    def _normalise_workspace(path: str | Path) -> str:
        return str(Path(path).expanduser().resolve())

    @classmethod
    def _load_trusted_workspaces(cls) -> list[str]:
        try:
            data = json.loads(TRUST_REGISTRY_PATH.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return []
        if isinstance(data, dict):
            items = data.get("workspaces", [])
        else:
            items = data
        return [str(item) for item in items if isinstance(item, str)]

    @classmethod
    def _is_workspace_trusted(cls, workspace: str | Path) -> bool:
        workspace_key = cls._normalise_workspace(workspace)
        return workspace_key in set(cls._load_trusted_workspaces())

    @classmethod
    def trust_workspace(cls, workspace: str | Path) -> None:
        """Persist workspace trust in the user's Nyx registry."""
        workspace_key = cls._normalise_workspace(workspace)
        trusted = cls._load_trusted_workspaces()
        if workspace_key in trusted:
            return
        trusted.append(workspace_key)
        TRUST_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        TRUST_REGISTRY_PATH.write_text(
            json.dumps({"workspaces": sorted(trusted)}, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _confirm_workspace_trust(workspace: Path, config_path: Path) -> bool:
        """Ask before loading workspace-local config, with safe non-interactive default."""
        if os.environ.get("NYX_TRUST_WORKSPACE", "").strip().lower() in {"1", "true", "yes", "y"}:
            return True
        if os.environ.get("NYX_SKIP_WORKSPACE_TRUST", "").strip().lower() in {"1", "true", "yes", "y"}:
            return False
        if not os.isatty(0):
            return False
        try:
            answer = input(
                f"Trust Nyx workspace '{workspace}' and load local config '{config_path}'? [y/N] "
            ).strip().lower()
        except OSError:
            return False
        return answer in {"y", "yes", "o", "oui"}

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Merge nested dictionaries without losing sibling defaults."""
        merged = deepcopy(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = Config._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _flatten_nested(raw: dict[str, Any], prefix: str, keys: list[str]) -> None:
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
