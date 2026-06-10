"""
Nyx — the core agentic loop.

Orchestrates: LLM calls → tool execution → context management.
Supports: skills, MCP tools, web search, subagents (sync + parallel),
          persistent memory, summarisation, code execution,
          repo mapping, code search, test loop, auto-correction.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from nyx.config import Config, MODE_SYSTEM_PROMPTS, AUTONOMY_CONFIGS, ARCHITECT_TOOLS
from nyx.providers.base import BaseLLMProvider, LLMResponse, ToolCall, ToolDefinition
from nyx.providers import get_provider
from nyx.mcp_client import MCPManager
from nyx.skill_manager import SkillManager
from nyx.subagent import SubagentManager
from nyx.async_subagent import AsyncSubagentManager, ParallelTask
from nyx.memory import MemoryManager
from nyx.web_search import search_web, format_search_results, fetch_page
from nyx.permissions import PermissionManager, PermissionLevel
from nyx.sandbox import Sandbox, PathTraversalError
from nyx.audit import AuditTrail
from nyx.json_logger import JSONLogger
from nyx.diff_tool import (
    PatchTool, compute_diff, compute_diff_from_path,
    ChangeType, PatchInfo, PatchRecord, RollbackEntry,
    parse_unified_diff, parse_search_replace,
    validate_patch_syntax, detect_conflicts,
    format_diff_for_display, get_patch_history,
)
from nyx.repo_map import build_repo_map, build_repo_map_short
from nyx.search_code import search_code as _search_code
from nyx.test_loop import run_tests as _run_tests
from nyx.test_loop import auto_correct_loop, format_failures_for_llm, TestFailure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

BUILTIN_TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="web_search",
        description="Search the internet for current information. Returns titles, URLs and snippets.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query string"},
                "max_results": {"type": "integer", "description": "Maximum number of results (1-10)", "default": 5},
            },
            "required": ["query"],
        },
    ),
    ToolDefinition(
        name="web_fetch",
        description="Fetch the text content of a web page. Useful for reading articles, docs, etc.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The full URL to fetch"},
            },
            "required": ["url"],
        },
    ),
    ToolDefinition(
        name="subagent_run",
        description="Spawn a subagent to handle a complex subtask. Use for research, code gen, analysis.",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "A unique name for this subagent"},
                "task": {"type": "string", "description": "The task to delegate (detailed instructions)"},
                "system_prompt": {"type": "string", "description": "Optional custom system prompt for the subagent"},
            },
            "required": ["name", "task"],
        },
    ),
    ToolDefinition(
        name="parallel_subagents",
        description="Execute multiple subagent tasks in PARALLEL using a thread pool. Use for independent research, code generation, or analysis tasks. Much faster than sequential execution.",
        parameters={
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "description": "List of tasks to run in parallel",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Unique name for this subagent"},
                            "task": {"type": "string", "description": "The detailed task description"},
                            "context": {"type": "string", "description": "Optional shared context for this subagent"},
                            "system_prompt": {"type": "string", "description": "Optional custom system prompt"},
                        },
                        "required": ["name", "task"],
                    },
                },
            },
            "required": ["tasks"],
        },
    ),
    ToolDefinition(
        name="memory_save",
        description="Save an important note to persistent memory for future reference across conversations.",
        parameters={
            "type": "object",
            "properties": {
                "note": {"type": "string", "description": "The note or information to remember"},
                "tags": {"type": "string", "description": "Optional comma-separated tags for categorisation"},
            },
            "required": ["note"],
        },
    ),
    ToolDefinition(
        name="memory_recall",
        description="Recall past conversations and saved notes from persistent memory.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for in memory"},
            },
            "required": ["query"],
        },
    ),
    ToolDefinition(
        name="execute_command",
        description="Execute a shell command on the local system. Most commands are allowed. Destructive commands (rm, sudo, chmod, curl, etc.) require user approval before execution.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default: 30)", "default": 30},
            },
            "required": ["command"],
        },
    ),
    ToolDefinition(
        name="read_file",
        description="Read the contents of a file from the filesystem.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file (absolute or relative to project root)"},
            },
            "required": ["path"],
        },
    ),
    ToolDefinition(
        name="write_file",
        description="Write content to a file on the filesystem. Uses a diff/patch workflow with user approval for changes outside the project sandbox.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "content": {"type": "string", "description": "The content to write"},
            },
            "required": ["path", "content"],
        },
    ),
    ToolDefinition(
        name="append_file",
        description="Append content to an existing file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "content": {"type": "string", "description": "Content to append"},
            },
            "required": ["path", "content"],
        },
    ),
    ToolDefinition(
        name="list_files",
        description="List files in a directory on the filesystem.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path (default: current dir)", "default": "."},
                "recursive": {"type": "boolean", "description": "List recursively", "default": False},
            },
            "required": [],
        },
    ),
    ToolDefinition(
        name="apply_diff",
        description="Apply a unified diff or SEARCH/REPLACE patch to a file. Supports standard unified diff format and SEARCH/REPLACE blocks. Validates syntax, detects conflicts, categorizes changes (CREATE/MODIFY/DELETE), and shows a clear summary before requiring user approval. Preferred over write_file for modifying existing files.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to modify"},
                "diff": {"type": "string", "description": "The unified diff or SEARCH/REPLACE block to apply. For unified diff, use standard format with @@ hunk headers. For SEARCH/REPLACE, use \\<<<<<<< SEARCH / ======= / \\>>>>>>> REPLACE blocks."},
                "description": {"type": "string", "description": "Brief description of what this change does"},
            },
            "required": ["path", "diff"],
        },
    ),
    ToolDefinition(
        name="rollback_file",
        description="Rollback the most recent change to a file. Restores the file to its state before the last patch was applied.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to rollback"},
            },
            "required": ["path"],
        },
    ),
    ToolDefinition(
        name="patch_history",
        description="Show the history of patches applied to files. Returns a list of recent file modifications with timestamps, change types, and summaries.",
        parameters={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Maximum number of history entries to return (default: 20)", "default": 20},
            },
            "required": [],
        },
    ),
    ToolDefinition(
        name="finish",
        description="Call this when the user's task is complete. Provide a summary of what was done.",
        parameters={
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "A summary of what was accomplished"},
                "result": {"type": "string", "description": "The final result or output"},
            },
            "required": ["summary", "result"],
        },
    ),
    # ------------------------------------------------------------------
    # Repo map tool
    # ------------------------------------------------------------------
    ToolDefinition(
        name="repo_map",
        description="Get a structured overview of the current repository: directory tree, important files, git status (branch, changes, last commit), and available test suites. Use this at the start of a task to understand the project layout.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Optional path to the repository root (default: current directory)"},
                "short": {"type": "boolean", "description": "If true, return a one-line summary instead of full map", "default": False},
            },
            "required": [],
        },
    ),
    # ------------------------------------------------------------------
    # Code search tool
    # ------------------------------------------------------------------
    ToolDefinition(
        name="search_code",
        description="Search the codebase using ripgrep (or grep fallback). Returns matching lines with surrounding context. Use for finding function definitions, variable references, patterns, or any code navigation.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "The search pattern (plain text or regex)"},
                "file_pattern": {"type": "string", "description": "Optional glob filter (e.g., '*.py', '*.rs', '*.ts')"},
                "max_results": {"type": "integer", "description": "Maximum number of matches to return (default: 30)", "default": 30},
                "context_lines": {"type": "integer", "description": "Lines of context before/after each match (default: 2)", "default": 2},
                "case_sensitive": {"type": "boolean", "description": "Case-sensitive search (default: false)", "default": False},
                "regex": {"type": "boolean", "description": "Treat pattern as regex (default: auto-detect)", "default": False},
                "fixed_strings": {"type": "boolean", "description": "Treat pattern as literal text (default: false)", "default": False},
            },
            "required": ["pattern"],
        },
    ),
    # ------------------------------------------------------------------
    # Test tools
    # ------------------------------------------------------------------
    ToolDefinition(
        name="run_tests",
        description="Run the project's test suite and return structured results with parsed failures. Auto-discovers the test framework (pytest, unittest, npm, etc.).",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Optional specific test command (e.g., 'pytest tests/test_agent.py -v'). If omitted, auto-discovers."},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default: 120)", "default": 120},
            },
            "required": [],
        },
    ),
    ToolDefinition(
        name="auto_correct_tests",
        description="Run tests, parse failures, and automatically fix them using a subagent. Iterates up to max_iterations times until all tests pass. Use this to automatically fix broken tests.",
        parameters={
            "type": "object",
            "properties": {
                "test_command": {"type": "string", "description": "Optional specific test command. If omitted, auto-discovers."},
                "max_iterations": {"type": "integer", "description": "Maximum fix iterations (default: 5)", "default": 5},
                "timeout": {"type": "integer", "description": "Timeout per test run in seconds (default: 120)", "default": 120},
            },
            "required": [],
        },
    ),
]


# ---------------------------------------------------------------------------
# AgentContext
# ---------------------------------------------------------------------------


@dataclass
class AgentContext:
    """Conversation context for the agent."""
    messages: list[dict[str, Any]] = field(default_factory=list)
    max_history: int = 50

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_history:
            keep = [self.messages[0]] if self.messages[0]["role"] == "system" else []
            keep.extend(self.messages[-(self.max_history - len(keep)):])
            self.messages = keep

    def add_tool_result(self, tool_call_id: str, name: str, content: str, use_anthropic_format: bool = False) -> None:
        if use_anthropic_format:
            # Anthropic uses "tool_use_id" and content blocks with role "user" for tool results
            self.messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": [{"type": "text", "text": content}],
                    }
                ],
            })
        else:
            # OpenAI-compatible format
            self.messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": name,
                "content": content,
            })

    def clear(self) -> None:
        system_msgs = [m for m in self.messages if m["role"] == "system"]
        self.messages = system_msgs

    def __len__(self) -> int:
        return len(self.messages)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent:
    """The main agent that orchestrates everything."""

    # Commands that are explicitly blocked (destructive)
    # These will trigger an interactive user approval prompt.
    # Uses word-boundary matching to avoid false positives
    # (e.g., "cp " won't match "scp ", "mv " won't match "tmux ").
    DANGEROUS_PATTERNS: list[str] = [
        "rm", "rmdir", "mv", "cp", "chmod", "chown", "dd",
        "sudo", "su", "passwd", "kill",
        "mkfs", "fdisk", "mount", "umount", "iptables",
        "wget", "curl", "apt", "yum", "dnf", "pacman",
        "pip install", "npm install", "git push", "git reset",
        "git rebase", "git merge", "git cherry-pick",
        "docker", "systemctl", "journalctl",
    ]

    # Operators that are dangerous in shell context (checked separately)
    DANGEROUS_OPERATORS: list[str] = [
        ">", ">>", "|",
    ]

    @staticmethod
    def _is_dangerous_command(command: str) -> bool:
        """Check if a command matches dangerous patterns using word-boundary matching."""
        cmd_lower = command.strip().lower()

        # Check dangerous operators (only when used as standalone tokens)
        # Avoid false positives: "||" should not match "|", "=>" should not match ">"
        for op in Agent.DANGEROUS_OPERATORS:
            # Match operator as a standalone token (not part of ||, &&, =>, etc.)
            if re.search(rf'(?:^|\s){re.escape(op)}(?:\s|$)', cmd_lower):
                return True

        # Check dangerous commands using word boundaries
        for pattern in Agent.DANGEROUS_PATTERNS:
            if re.search(rf'\b{re.escape(pattern)}\b', cmd_lower):
                return True

        return False

    def _request_command_approval(self, command: str) -> tuple[bool, str]:
        """Request user approval for a potentially dangerous command.

        Returns (approved: bool, reason: str).
        If no approval callback is configured, the command is denied by default.
        """
        if self._on_command_approval:
            return self._on_command_approval(command)
        # Default: deny if no approval mechanism is configured
        logger.warning("No approval callback configured, denying dangerous command: %s", command)
        return False, "No approval mechanism configured. This command requires manual approval."

    def __init__(
        self,
        config: Config,
        provider: BaseLLMProvider | None = None,
        mcp_manager: MCPManager | None = None,
        skill_manager: SkillManager | None = None,
        subagent_manager: SubagentManager | None = None,
        memory_manager: MemoryManager | None = None,
        async_subagent_manager: AsyncSubagentManager | None = None,
        on_token: Callable[[str], None] | None = None,
        on_command_approval: Callable[[str], tuple[bool, str]] | None = None,
        on_file_approval: Callable[[str, str, str], tuple[bool, str]] | None = None,
    ) -> None:
        self.config = config
        self.provider = provider or get_provider(config)
        self.mcp = mcp_manager or MCPManager()
        self.skills = skill_manager or SkillManager(config.skills_dir)
        self.subagents = subagent_manager or SubagentManager(config)
        self.memory = memory_manager or MemoryManager(provider=self.provider)
        self.async_subagents = async_subagent_manager or AsyncSubagentManager(
            config=config,
            provider_factory=lambda: get_provider(config),
        )
        self.on_token = on_token
        self._on_command_approval = on_command_approval
        self._on_file_approval = on_file_approval
        self.context = AgentContext()
        self.call_depth = 0
        self.max_depth = config.agent_max_depth
        # Controlled tool subset for subagents (set after setup)
        self._subagent_tools: list[ToolDefinition] = []

        # -- Security subsystems --
        # Permission manager
        self.permissions = PermissionManager(config.permissions_config if config.permissions_config else None)

        # Sandbox (project root)
        self.sandbox = Sandbox(
            project_root=config.project_dir if config.sandbox_enabled else None,
            allow_paths=config.sandbox_allow_paths,
            deny_paths=config.sandbox_deny_paths,
            auto_chdir=config.sandbox_auto_chdir,
        )

        # Audit trail
        audit_dir = config.audit_output_dir or (config.project_dir + "/.nyx/audit" if config.project_dir else "")
        self.audit = AuditTrail(
            output_dir=audit_dir if audit_dir else None,
            agent_id="nyx",
            enabled=config.audit_enabled,
            max_file_size_mb=config.audit_max_file_size_mb,
        )

        # JSON logger (optional structured logging with session id, costs, etc.)
        json_log_path = config.json_logging_output_path or (
            config.project_dir + "/.nyx/nyx_log.ndjson" if config.project_dir else ""
        )
        self.json_logger = JSONLogger(
            output_path=json_log_path if json_log_path and config.json_logging_enabled else None,
            log_to_stderr=config.json_logging_log_to_stderr,
            enabled=config.json_logging_enabled,
            model=config.model,
        )

        # Patch/diff tool
        self.patch_tool = PatchTool(
            approval_callback=self._file_approval_handler,
            project_dir=config.project_dir if config.project_dir else None,
            enable_rollback=True,
            enable_history=True,
            use_git=True,
        )

        # Collect all tools
        self._all_tools: list[ToolDefinition] = list(BUILTIN_TOOLS)

    # ------------------------------------------------------------------
    # Approval callback properties (keep PermissionManager in sync)
    # ------------------------------------------------------------------

    @property
    def on_command_approval(self) -> Callable[[str], tuple[bool, str]] | None:
        return self._on_command_approval

    @on_command_approval.setter
    def on_command_approval(self, callback: Callable[[str], tuple[bool, str]] | None) -> None:
        self._on_command_approval = callback
        # Sync to PermissionManager so PROMPT-level shell commands can be approved
        if hasattr(self, "permissions"):
            if callback:
                def _pm_shell_callback(cat: str, desc: str, target: str) -> tuple[bool, str]:
                    if cat == "shell":
                        return callback(target)
                    return True, ""
                self.permissions.set_approval_callback(_pm_shell_callback)
            else:
                self.permissions.set_approval_callback(None)

    @property
    def on_file_approval(self) -> Callable[[str, str, str], tuple[bool, str]] | None:
        return self._on_file_approval

    @on_file_approval.setter
    def on_file_approval(self, callback: Callable[[str, str, str], tuple[bool, str]] | None) -> None:
        self._on_file_approval = callback
        # Sync to PatchTool so file writes go through the interactive prompt
        if hasattr(self, "patch_tool"):
            self.patch_tool.set_approval_callback(
                self._file_approval_handler
            )

    def _file_approval_handler(self, path: str, summary: str, diff: str) -> tuple[bool, str]:
        """Handle file operation approval requests from the PatchTool."""
        if self._on_file_approval:
            return self._on_file_approval(path, summary, diff)
        # If no interactive callback, fall back to permission rules (allow/deny only)
        level = self.permissions.check_file_write(path)
        if level == PermissionLevel.DENY:
            return False, f"Writing to this path is explicitly denied by security policy: {path}"
        # ALLOW or PROMPT without a callback → allow (sandbox already validated the path)
        return True, ""

    def setup(self) -> None:
        """Connect MCP servers, discover skills, inject memory summary, repo map, and mode config."""
        # MCP
        if self.config.mcp_servers:
            logger.info("Connecting MCP servers...")
            print("\n🔌 Connecting MCP servers...")

            # Wire up progress callback if available
            if self.json_logger and self.json_logger.enabled:
                self.mcp.set_progress_callback(
                    lambda label, cur, total: self.json_logger.log_event(
                        "mcp_progress", {"server": label, "current": cur, "total": total}
                    )
                )

            self.mcp.connect_all(self.config.mcp_servers)
            self._all_tools.extend(self.mcp.get_tool_definitions())

        # Wire up subagent progress callback
        if self.json_logger and self.json_logger.enabled:
            self.subagents.set_progress_callback(
                lambda label, cur, total: self.json_logger.log_event(
                    "subagent_progress", {"name": label, "current": cur, "total": total}
                )
            )

        # Skills
        skills_dir = self.config.skills_dir
        if skills_dir:
            logger.info("Loading skills from %s", skills_dir)
            print("🧠 Loading skills...")
            skills_found = self.skills.discover(skills_dir)
            if skills_found:
                self._all_tools.extend(self.skills.get_tool_definitions())

        # Project directory / sandbox
        if self.config.project_dir:
            root_info = self.config.project_dir
            if self.sandbox.root:
                root_info = self.sandbox.root_str
            self.context.add("system", f"The current working directory is: {root_info}")
            self.context.add("system", (
                "You are operating in a sandboxed environment. "
                "All file operations are restricted to the project directory. "
                "Use apply_diff for modifying existing files — it shows a diff and requires approval."
            ))

        # -- Memory summary injection --
        # Automatically inject a summary of past conversations into the LLM context
        self._inject_memory_summary()

        # -- Repo map injection --
        # Automatically inject a short repo map for context
        if self.config.project_dir:
            try:
                repo_summary = build_repo_map_short(self.config.project_dir)
                self.context.add("system", f"[Repository context]\n{repo_summary}")
                logger.info("Injected repo map: %s", repo_summary)
            except Exception as e:
                logger.debug("Could not build repo map: %s", e)

        # -- Mode & autonomy configuration --
        self._apply_mode_config()

        # -- Controlled tool subset for subagents --
        self._subagent_tools = self._get_subagent_tools()
        self.subagents.set_default_tools(self._subagent_tools)
        logger.info("Subagent tools: %d (filtered from %d)", len(self._subagent_tools), len(self._all_tools))

        logger.info("Total tools available: %d", len(self._all_tools))
        print(f"\n📦 Total tools available: {len(self._all_tools)}")

    def _inject_memory_summary(self) -> None:
        """Inject a summary of past conversations into the LLM context."""
        try:
            convs = self.memory.list_conversations()
            if not convs:
                return

            # Build a concise summary of past conversations
            parts: list[str] = []
            parts.append("[Memory: Past conversations]")

            for conv in convs[:5]:  # Last 5 conversations
                title = conv.get("title", "Untitled")
                summary = conv.get("summary", "")
                entry_count = conv.get("entry_count", 0)
                if summary:
                    parts.append(f"- {title} ({entry_count} msgs): {summary[:200]}")
                else:
                    parts.append(f"- {title} ({entry_count} msgs)")

            # Also include saved notes
            notes = self.memory._load_notes()
            if notes:
                parts.append(f"\n[Saved notes: {len(notes)}]")
                for note in notes[-3:]:  # Last 3 notes
                    parts.append(f"- {note['content'][:150]}")

            if len(parts) > 1:
                self.context.add("system", "\n".join(parts))
                logger.info("Injected memory summary (%d conversations)", len(convs))
        except Exception as e:
            logger.debug("Could not inject memory summary: %s", e)

    # ------------------------------------------------------------------
    # Mode & autonomy configuration
    # ------------------------------------------------------------------

    def _apply_mode_config(self) -> None:
        """Apply mode and autonomy settings to the agent.

        Called at the end of setup():
        - Injects the mode-specific system prompt suffix.
        - Filters tools for architect mode (read-only).
        - Applies autonomy-level approval callbacks.
        - Adjusts max_depth based on autonomy multiplier.
        """
        mode = self.config.agent_mode.lower()
        autonomy = self.config.agent_autonomy.lower()

        # Validate
        if mode not in MODE_SYSTEM_PROMPTS:
            logger.warning("Unknown agent mode '%s', defaulting to 'chat'", mode)
            mode = "chat"
        if autonomy not in AUTONOMY_CONFIGS:
            logger.warning("Unknown autonomy level '%s', defaulting to 'ask'", autonomy)
            autonomy = "ask"

        # 1. Inject mode system prompt suffix
        suffix = MODE_SYSTEM_PROMPTS[mode]
        if suffix:
            self.context.add("system", suffix)
            logger.info("Mode '%s' system prompt injected", mode)

        # 2. Filter tools for architect mode (read-only)
        if mode == "architect":
            self._all_tools = [t for t in self._all_tools if t.name in ARCHITECT_TOOLS]
            logger.info("Architect mode: tools filtered to %d read-only tools", len(self._all_tools))

        # 3. Apply autonomy-level behaviour
        aut = AUTONOMY_CONFIGS[autonomy]
        multiplier = aut["max_depth_multiplier"]
        self.max_depth = self.config.agent_max_depth * multiplier
        logger.info("Autonomy '%s': max_depth=%d", autonomy, self.max_depth)

        if aut["auto_approve_files"]:
            # Bypass interactive file-approval prompt — accept all writes silently
            self._on_file_approval = lambda path, summary, diff: (True, "")
            logger.info("Autonomy '%s': file writes auto-approved", autonomy)

        if aut["auto_approve_commands"]:
            # Bypass interactive command-approval prompt — accept all commands silently
            self._on_command_approval = lambda cmd: (True, "")
            # Also update PermissionManager callback so PROMPT-level rules pass through
            self.permissions.set_approval_callback(
                lambda cat, desc, target: (True, "")
            )
            logger.info("Autonomy '%s': commands auto-approved", autonomy)
        elif aut["auto_approve_files"] and not aut["auto_approve_commands"]:
            # Sync PermissionManager for file category only
            orig_pm_cb = self.permissions._approval_callback

            def _selective_pm_callback(cat: str, desc: str, target: str) -> tuple[bool, str]:
                if cat == "filesystem":
                    return True, ""
                if orig_pm_cb:
                    return orig_pm_cb(cat, desc, target)
                return False, "No approval mechanism configured."

            self.permissions.set_approval_callback(_selective_pm_callback)

        # Print mode banner
        mode_labels = {
            "chat": "💬 Chat", "code": "💻 Code",
            "architect": "🏛  Architect", "debug": "🐛 Debug",
        }
        aut_labels = {"ask": "🙋 Ask", "auto": "⚡ Auto", "yolo": "🔥 Yolo"}
        print(f"  Mode: {mode_labels.get(mode, mode)}  |  Autonomy: {aut_labels.get(autonomy, autonomy)}")

    def switch_mode(self, mode: str) -> str:
        """Switch the agent mode at runtime. Returns a status message."""
        mode = mode.lower().strip()
        if mode not in MODE_SYSTEM_PROMPTS:
            return f"Unknown mode '{mode}'. Valid modes: chat, code, architect, debug"
        self.config.agent_mode = mode
        # Re-inject mode prompt and re-filter tools
        self._apply_mode_config()
        return f"Mode switched to: {mode}"

    def switch_autonomy(self, autonomy: str) -> str:
        """Switch the autonomy level at runtime. Returns a status message."""
        autonomy = autonomy.lower().strip()
        if autonomy not in AUTONOMY_CONFIGS:
            return f"Unknown autonomy level '{autonomy}'. Valid levels: ask, auto, yolo"
        self.config.agent_autonomy = autonomy
        self._apply_mode_config()
        return f"Autonomy switched to: {autonomy}"

    # ------------------------------------------------------------------
    # Controlled tool subsets for subagents
    # ------------------------------------------------------------------

    SUBAGENT_TOOL_WHITELIST: set[str] = {
        # Read-only tools (safe for subagents)
        "read_file",
        "list_files",
        "search_code",
        "repo_map",
        "web_search",
        "web_fetch",
        "memory_recall",
        # Write tools (subagents can propose changes)
        "write_file",
        "apply_diff",
        "append_file",
        # Execution (subagents can run safe commands)
        "execute_command",
        "run_tests",
        # Meta
        "finish",
    }

    SUBAGENT_TOOL_BLACKLIST: set[str] = {
        # Prevent subagents from spawning their own subagents (infinite recursion)
        "subagent_run",
        "parallel_subagents",
        "auto_correct_tests",
        # Memory save should be controlled by main agent
        "memory_save",
    }

    def _get_subagent_tools(self) -> list[ToolDefinition]:
        """Return a filtered list of tools safe for subagent use.

        Subagents get a read-only subset plus controlled write/execution tools.
        They cannot spawn sub-subagents or auto-correct tests.
        """
        return [
            t for t in self._all_tools
            if t.name in self.SUBAGENT_TOOL_WHITELIST
        ]

    @property
    def tools(self) -> list[ToolDefinition]:
        return self._all_tools

    def run(self, user_input: str, on_token: Callable[[str], None] | None = None) -> str:
        """Process a user input through the agentic loop."""
        self.context.add("user", user_input)
        # Also save to persistent memory
        self.memory.add_entry("user", user_input)
        return self._loop(on_token=on_token)

    def _loop(self, on_token: Callable[[str], None] | None = None) -> str:
        """The main agentic reasoning loop."""
        self.call_depth += 1
        if self.call_depth > self.max_depth:
            # Reset depth so subsequent calls work normally
            self.call_depth = 0
            logger.warning("Max reasoning depth reached (%d)", self.max_depth)
            return (
                f"I've reached the maximum number of reasoning steps ({self.max_depth}). "
                "This usually means I'm stuck in a loop — for example, a tool keeps failing "
                "and I'm retrying with different approaches. "
                "Please tell me what you'd like me to do next, or simplify the request."
            )

        logger.debug("LLM call (depth=%d, messages=%d)", self.call_depth, len(self.context.messages))
        llm_start = time.time()
        llm_error = ""
        try:
            response = self.provider.chat(
                messages=self.context.messages,
                tools=self.tools if self.tools else None,
                stream=self.config.stream,
                on_token=on_token or self.on_token,
            )
        except Exception as e:
            self.call_depth -= 1
            logger.error("LLM call failed: %s", e)
            llm_error = str(e)
            # Log the failed LLM call
            self.json_logger.log_llm_call(
                input_tokens=0,
                output_tokens=0,
                duration_ms=(time.time() - llm_start) * 1000,
                error=llm_error,
            )
            return f"Error calling LLM: {e}"

        # Log successful LLM call
        usage = response.usage or {}
        self.json_logger.log_llm_call(
            input_tokens=usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0),
            output_tokens=usage.get("output_tokens", 0) or usage.get("completion_tokens", 0),
            duration_ms=(time.time() - llm_start) * 1000,
        )

        # Handle tool calls
        if response.tool_calls:
            tool_names = [tc.name for tc in response.tool_calls]
            logger.info("Tool calls: %s", tool_names)
            # content must be a string (not None/null) for API compatibility
            content_str = response.content if response.content else ""
            self.context.messages.append({
                "role": "assistant",
                "content": content_str,
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    for tc in response.tool_calls
                ],
            })

            for tc in response.tool_calls:
                t_start = time.time()
                
                # Format a friendly console display for tool call
                target = ""
                if tc.name in ("read_file", "write_file", "apply_diff", "append_file", "rollback_file"):
                    target = tc.arguments.get("path", "")
                elif tc.name == "list_files":
                    target = tc.arguments.get("path", ".")
                elif tc.name == "execute_command":
                    cmd = tc.arguments.get("command", "").strip()
                    target = (cmd[:60] + "...") if len(cmd) > 60 else cmd
                elif tc.name == "web_search":
                    target = tc.arguments.get("query", "")
                elif tc.name == "web_fetch":
                    target = tc.arguments.get("url", "")
                elif tc.name == "subagent_run":
                    target = tc.arguments.get("name", "")
                elif tc.name == "parallel_subagents":
                    tasks = tc.arguments.get("tasks", [])
                    target = f"{len(tasks)} tasks"
                elif tc.name == "memory_save":
                    note = tc.arguments.get("note", "").strip()
                    target = (note[:40] + "...") if len(note) > 40 else note
                elif tc.name == "memory_recall":
                    target = tc.arguments.get("query", "")
                elif tc.name == "repo_map":
                    target = tc.arguments.get("path", "")
                elif tc.name == "search_code":
                    target = tc.arguments.get("pattern", "")
                elif tc.name == "run_tests":
                    cmd = tc.arguments.get("command", "")
                    target = cmd if cmd else "(default)"
                else:
                    # Generic fallback check for custom/MCP tools
                    for key in ("path", "file", "filepath", "filename", "uri", "url", "command", "query"):
                        if key in tc.arguments:
                            val = str(tc.arguments[key]).strip()
                            if val:
                                target = (val[:60] + "...") if len(val) > 60 else val
                                break

                # ANSI styles locally to avoid circular dependencies
                no_color = os.environ.get("NO_COLOR")
                c_reset = "\033[0m"
                c_yellow_dim = "\033[2m\033[93m"
                c_cyan = "\033[96m"
                c_green = "\033[92m"
                c_red = "\033[91m"
                
                def color_text(txt: str, code: str) -> str:
                    return f"{code}{txt}{c_reset}" if not no_color else txt

                target_str = f" ➔ {color_text(target, c_cyan)}" if target else ""
                print(f"🛠️  [{color_text('Tool Call', c_yellow_dim)}] {tc.name}{target_str}", flush=True)

                # Log tool call attempt
                self.json_logger.log_tool_call(
                    tool_name=tc.name,
                    arguments=tc.arguments,
                )
                result = self._execute_tool(tc)
                duration = (time.time() - t_start) * 1000
                
                # Format friendly console display for tool result
                duration_str = f"{duration/1000:.2f}s" if duration >= 1000 else f"{int(duration)}ms"
                is_err = "error" in result.lower() or "[error]" in result.lower() or "denied" in result.lower()
                status_icon = "❌" if is_err else "✓"
                status_color = c_red if is_err else c_green
                status_text = "Failure" if is_err else "Success"
                
                # Extract clean/useful details (e.g., number of lines)
                detail_parts = []
                if not is_err:
                    if tc.name == "read_file":
                        lines_count = len(result.splitlines())
                        detail_parts.append(f"{lines_count} lines read")
                    elif tc.name in ("write_file", "append_file"):
                        content = tc.arguments.get("content", "")
                        lines_count = len(content.splitlines())
                        detail_parts.append(f"{lines_count} lines written")
                    elif tc.name == "apply_diff":
                        diff = tc.arguments.get("diff", "")
                        lines_count = len(diff.splitlines())
                        detail_parts.append(f"{lines_count} lines diff")
                    elif tc.name == "execute_command":
                        if result.strip() and result != "(no output)":
                            lines_count = len(result.splitlines())
                            detail_parts.append(f"{lines_count} lines output")
                
                detail_parts.append(f"took {duration_str}")
                detail_str = ", ".join(detail_parts)
                print(f"{status_icon}  [{color_text('Tool ' + status_text, status_color)}] {tc.name}{target_str} ({detail_str})", flush=True)

                # Log tool result
                self.json_logger.log_tool_result(
                    tool_name=tc.name,
                    result=result,
                    duration_ms=duration,
                )

                # Audit the tool call
                self.audit.log_tool_call(
                    tool_name=tc.name,
                    arguments=tc.arguments,
                    result=result,
                    duration_ms=duration,
                )

                use_anthropic = self.config.provider == "anthropic"
                self.context.add_tool_result(tc.id, tc.name, result, use_anthropic_format=use_anthropic)

            # Continue the loop (propagate on_token for recursive calls)
            result = self._loop(on_token=on_token)
            self.call_depth -= 1
            return result

        # No tool calls — final response
        if response.content:
            logger.debug("Final response (%d chars)", len(response.content))
            self.context.add("assistant", response.content)
            self.memory.add_entry("assistant", response.content)

        self.call_depth -= 1
        return response.content

    def _execute_tool(self, tc: ToolCall) -> str:
        """Execute a single tool call and return the result as a string."""
        from pathlib import Path
        name = tc.name
        args = tc.arguments

        try:
            # -- Web tools --
            if name == "web_search":
                query = args.get("query", "")
                max_results = min(args.get("max_results", 5), 10)
                if self.config.web_search_enabled:
                    results = search_web(query, self.config.web_search_provider, max_results)
                    return format_search_results(results)
                return "Web search is disabled in config."

            if name == "web_fetch":
                return fetch_page(args.get("url", ""))

            # -- Subagent tools (with controlled tool subset) --
            if name == "subagent_run":
                s_name = args.get("name", "unnamed")
                s_task = args.get("task", "")
                s_prompt = args.get("system_prompt", "")
                agent = self.subagents.spawn(s_name, s_prompt)
                # Pass controlled tool subset to subagent
                result = agent.execute(s_task, tools=self._subagent_tools if self._subagent_tools else None)
                if result.error:
                    return f"[Subagent:{s_name}] Error: {result.error}"
                return f"[Subagent:{s_name}] Result:\n{result.output}"

            if name == "parallel_subagents":
                tasks_data = args.get("tasks", [])
                if not tasks_data:
                    return "[parallel_subagents] No tasks provided."
                tasks = [
                    ParallelTask(
                        name=t.get("name", f"task_{i}"),
                        task=t["task"],
                        context=t.get("context", ""),
                        system_prompt=t.get("system_prompt", ""),
                    )
                    for i, t in enumerate(tasks_data)
                ]
                result = self.async_subagents.run_parallel(tasks)
                parts = [f"✅ Parallel execution: {result.completed} completed, {result.failed} failed"]
                for r in result.results:
                    status = "✓" if not r.error else "✗"
                    parts.append(f"\n[{status}] {r.task[:60]}")
                    if r.error:
                        parts.append(f"  Error: {r.error}")
                    else:
                        parts.append(f"  Output: {r.output[:500]}")
                return "\n".join(parts)

            # -- Memory tools --
            if name == "memory_save":
                note = args.get("note", "")
                tags_raw = args.get("tags", "")
                # Store as a dedicated memory entry with role "memory"
                self.memory.add_entry("memory", note)
                # Also save to a dedicated notes file for persistence across conversations
                self.memory._save_note(note, tags_raw)
                tag_info = f" (tags: {tags_raw})" if tags_raw else ""
                return f"Note saved to memory{tag_info}: {note[:100]}..."

            if name == "memory_recall":
                query = args.get("query", "")
                if not query:
                    return "Please provide a query to search for."

                # Search across all conversation entries (including saved notes)
                relevant = []

                # Search within conversation entries (using full data, not truncated)
                for c_id, conv in self.memory.conversations.items():
                    # Search title
                    if query.lower() in conv.title.lower():
                        relevant.append(f"- [{c_id[:8]}] {conv.title} ({len(conv.entries)} messages): {conv.summary[:200]}")
                    # Search summary
                    if query.lower() in conv.summary.lower():
                        relevant.append(f"- [{c_id[:8]}] {conv.title} ({len(conv.entries)} messages): {conv.summary[:200]}")
                    # Search within entries
                    for entry in conv.entries:
                        if query.lower() in entry.content.lower():
                            c_title = conv.title[:40]
                            preview = entry.content[:200]
                            relevant.append(f"- [{c_id[:8]}] {c_title}: {preview}")

                # Search saved notes
                notes = self.memory._load_notes()
                for note in notes:
                    if query.lower() in note["content"].lower() or query.lower() in note.get("tags", "").lower():
                        relevant.append(f"- [NOTE] {note['content'][:200]}")

                if not relevant:
                    return f"No relevant memories found for: {query}"
                return "Relevant memories:\n" + "\n".join(relevant[:10])

            # -- Shell command execution (with permissions + sandbox) --
            if name == "execute_command":
                import subprocess
                command = args.get("command", "").strip()
                timeout = args.get("timeout", 30)

                # Validate that the command is not empty
                if not command:
                    logger.warning("Empty command received from AI (args=%s)", args)
                    raw = args.get("_raw_buffer", "")
                    msg = (
                        "[ERROR] Empty command received. You must provide a valid shell command "
                        "in the 'command' parameter. For example: 'ls -la', 'cat file.txt', "
                        "'python3 script.py', 'which python3', etc."
                    )
                    if raw:
                        msg += f"\n(debug: raw arguments buffer was: {raw[:500]})"
                    return msg

                # 1. Check permissions (granular permission model)
                approved, reason, perm_level = self.permissions.authorize_shell(command)
                self.audit.log_permission_check("shell", command, perm_level.value, approved, reason)

                if not approved:
                    if perm_level == PermissionLevel.DENY:
                        logger.warning("Command explicitly denied: %s", command)
                        return (
                            f"[SECURITY] Command denied by security policy: '{command[:200]}'\n"
                            f"Reason: {reason}\n"
                            f"This command is not allowed under any circumstances."
                        )
                    # PROMPT level denied by user
                    logger.warning("User denied command: %s", command)
                    return (
                        f"[SECURITY] Command denied by user: '{command[:200]}'\n"
                        f"Reason: {reason}\n"
                        f"Please try a different approach that doesn't require this command."
                    )

                # 2. Also check legacy dangerous patterns (backward compat)
                if self._is_dangerous_command(command):
                    approved, reason = self._request_command_approval(command)
                    if not approved:
                        logger.warning("User denied dangerous command: %s", command)
                        return (
                            f"[SECURITY] Command denied by user: '{command[:200]}'\n"
                            f"Reason: {reason}\n"
                            f"Please try a different approach that doesn't require this command."
                        )
                    logger.info("User approved dangerous command: %s", command)

                # 3. Prepare command with sandbox (chdir to project root)
                final_command = self.sandbox.prepare_command(command)

                # 4. Execute
                logger.info("Executing command: %s", command)
                try:
                    proc = subprocess.run(
                        final_command, shell=True, capture_output=True, text=True, timeout=timeout,
                    )
                    out = proc.stdout or ""
                    err = proc.stderr or ""
                    if proc.returncode != 0:
                        logger.warning("Command exit code %d: %s", proc.returncode, command)
                        return f"Exit code: {proc.returncode}\nstdout:\n{out[:3000]}\nstderr:\n{err[:2000]}"
                    logger.debug("Command succeeded: %s", command)
                    return out[:5000] or "(no output)"
                except subprocess.TimeoutExpired:
                    logger.warning("Command timed out (%ds): %s", timeout, command)
                    return f"Command timed out after {timeout}s."
                except Exception as e:
                    logger.error("Command error: %s", e)
                    return f"Command error: {e}"

            # -- File read (with sandbox path resolution) --
            if name == "read_file":
                from pathlib import Path
                raw_path = args.get("path", "")
                try:
                    resolved = self.sandbox.safe_read_path(raw_path)
                except PathTraversalError as e:
                    self.audit.log_security_event("path_traversal_blocked", {
                        "path": raw_path,
                        "resolved": str(e.resolved),
                        "root": e.root,
                    })
                    return f"[SECURITY] Path traversal blocked: {raw_path}"
                except Exception as e:
                    return f"Error resolving path: {e}"

                if not resolved.exists():
                    return f"File not found: {raw_path}"
                try:
                    content = resolved.read_text(encoding="utf-8")[:8000]
                    self.audit.log_file_operation("read", str(resolved), len(content))
                    return content
                except Exception as e:
                    self.audit.log_file_operation("read", str(resolved), 0, success=False, error=str(e))
                    return f"Error reading file: {e}"

            # -- File write (via PatchTool with diff/approval) --
            if name == "write_file":
                raw_path = args.get("path", "")
                content = args.get("content", "")

                # Resolve path through sandbox
                try:
                    resolved = self.sandbox.resolve(raw_path, for_write=True)
                except PathTraversalError as e:
                    self.audit.log_security_event("path_traversal_blocked", {
                        "path": raw_path,
                        "resolved": str(e.resolved),
                        "root": e.root,
                    })
                    return f"[SECURITY] Path traversal blocked: {raw_path}"
                except Exception as e:
                    return f"Error resolving path: {e}"

                # Use PatchTool for diff-based approval
                success, message = self.patch_tool.propose_write(str(resolved), content)
                self.audit.log_file_operation(
                    "write", str(resolved), len(content), success=success,
                    error="" if success else message,
                )
                return message

            # -- Apply diff (with patch parsing, validation, conflict detection) --
            if name == "apply_diff":
                raw_path = args.get("path", "")
                diff_text = args.get("diff", "")
                description = args.get("description", "")

                # Resolve path through sandbox
                try:
                    resolved = self.sandbox.resolve(raw_path, for_write=True)
                except PathTraversalError as e:
                    self.audit.log_security_event("path_traversal_blocked", {
                        "path": raw_path,
                        "resolved": str(e.resolved),
                        "root": e.root,
                    })
                    return f"[SECURITY] Path traversal blocked: {raw_path}"
                except Exception as e:
                    return f"Error resolving path: {e}"

                # Audit the permission check (informational only — approval is handled
                # interactively by PatchTool via _file_approval_handler)
                perm_level = self.permissions.check_file_write(str(resolved))
                self.audit.log_permission_check(
                    "filesystem", str(resolved), perm_level.value,
                    perm_level != PermissionLevel.DENY, "",
                )
                if perm_level == PermissionLevel.DENY:
                    return (
                        f"[SECURITY] File write explicitly denied by policy: '{raw_path}'\n"
                        f"This path is blocked under all circumstances."
                    )

                # Use PatchTool.propose_patch — approval prompt is handled via
                # _file_approval_handler which calls on_file_approval if configured.
                success, message = self.patch_tool.propose_patch(str(resolved), diff_text)
                self.audit.log_file_operation(
                    "apply_diff", str(resolved), len(diff_text), success=success,
                    error="" if success else message,
                )

                # Auto-fallback: if a conflict is detected, try propose_write using the
                # proposed content already computed inside propose_patch (embedded in message).
                # This avoids an infinite loop where the AI keeps retrying the same bad patch.
                if not success and message.startswith("[CONFLICT]"):
                    from nyx.diff_tool import _apply_unified_diff_to_content, _apply_search_replace_to_content, _SEARCH_MARKER_RE
                    original_content = ""
                    if resolved.exists():
                        try:
                            original_content = resolved.read_text(encoding="utf-8")
                        except Exception:
                            pass
                    # Attempt to reconstruct proposed content from the diff
                    proposed_content = None
                    lines = diff_text.splitlines()
                    if any(_SEARCH_MARKER_RE.match(l) for l in lines):
                        proposed_content = _apply_search_replace_to_content(original_content, diff_text)
                    else:
                        proposed_content = _apply_unified_diff_to_content(original_content, diff_text)

                    if proposed_content is not None:
                        logger.info("apply_diff conflict: falling back to write_file for %s", resolved)
                        fb_success, fb_message = self.patch_tool.propose_write(str(resolved), proposed_content)
                        self.audit.log_file_operation(
                            "write_file_fallback", str(resolved), len(proposed_content),
                            success=fb_success, error="" if fb_success else fb_message,
                        )
                        fallback_note = "[auto-fallback from apply_diff] " if fb_success else "[fallback also failed] "
                        return fallback_note + fb_message
                    # Could not reconstruct content — return original conflict message
                    return message

                return message

            # -- Rollback file --
            if name == "rollback_file":
                raw_path = args.get("path", "")

                try:
                    resolved = self.sandbox.resolve(raw_path, for_write=True)
                except PathTraversalError as e:
                    return f"[SECURITY] Path traversal blocked: {raw_path}"
                except Exception as e:
                    return f"Error resolving path: {e}"

                success, message = self.patch_tool.rollback_last(str(resolved))
                self.audit.log_file_operation(
                    "rollback", str(resolved), 0, success=success,
                    error="" if success else message,
                )
                return message

            # -- Patch history --
            if name == "patch_history":
                limit = args.get("limit", 20)
                history = self.patch_tool.get_history(limit=limit)
                if not history:
                    return "No patch history available."
                lines = [f"Patch history ({len(history)} entries):"]
                for h in history:
                    lines.append(
                        f"  [{h.get('type', '?')}] {h.get('filepath', '?')} "
                        f"— {h.get('summary', '?')} "
                        f"({h.get('time', '?')})"
                    )
                return "\n".join(lines)

            # -- Append file (with sandbox + permissions) --
            if name == "append_file":
                raw_path = args.get("path", "")
                content = args.get("content", "")

                # Resolve path through sandbox
                try:
                    resolved = self.sandbox.resolve(raw_path, for_write=True)
                except PathTraversalError as e:
                    self.audit.log_security_event("path_traversal_blocked", {
                        "path": raw_path,
                        "resolved": str(e.resolved),
                        "root": e.root,
                    })
                    return f"[SECURITY] Path traversal blocked: {raw_path}"
                except Exception as e:
                    return f"Error resolving path: {e}"

                # Check permissions
                approved, reason, perm_level = self.permissions.authorize_file_write(str(resolved))
                self.audit.log_permission_check("filesystem", str(resolved), perm_level.value, approved, reason)

                if not approved:
                    return (
                        f"[SECURITY] File append denied: '{raw_path}'\n"
                        f"Reason: {reason}"
                    )

                # Use PatchTool for append
                success, message = self.patch_tool.propose_append(str(resolved), content)
                self.audit.log_file_operation(
                    "append", str(resolved), len(content), success=success,
                    error="" if success else message,
                )
                return message

            # -- List files (with sandbox) --
            if name == "list_files":
                from pathlib import Path
                raw_path = args.get("path", ".")
                recursive = args.get("recursive", False)

                try:
                    resolved = self.sandbox.resolve(raw_path)
                except PathTraversalError as e:
                    self.audit.log_security_event("path_traversal_blocked", {
                        "path": raw_path,
                        "resolved": str(e.resolved),
                        "root": e.root,
                    })
                    return f"[SECURITY] Path traversal blocked: {raw_path}"
                except Exception as e:
                    return f"Error resolving path: {e}"

                if not resolved.exists() or not resolved.is_dir():
                    return f"Directory not found: {raw_path}"
                if recursive:
                    files = [str(f.relative_to(resolved)) for f in resolved.rglob("*")]
                else:
                    files = [str(f.name) for f in resolved.iterdir()]
                return "\n".join(sorted(files)) if files else "(empty directory)"

            # -- Repo map --
            if name == "repo_map":
                path = args.get("path", "")
                short = args.get("short", False)
                root = Path(path).resolve() if path else (
                    Path(self.config.project_dir).resolve() if self.config.project_dir else Path(".").resolve()
                )
                try:
                    if short:
                        return build_repo_map_short(root)
                    return build_repo_map(root)
                except Exception as e:
                    logger.error("repo_map error: %s", e)
                    return f"Error building repo map: {e}"

            # -- Code search --
            if name == "search_code":
                pattern = args.get("pattern", "")
                if not pattern:
                    return "Please provide a search pattern."
                file_pattern = args.get("file_pattern")
                max_results = args.get("max_results", 30)
                context_lines = args.get("context_lines", 2)
                case_sensitive = args.get("case_sensitive", False)
                regex = args.get("regex", False)
                fixed_strings = args.get("fixed_strings", False)
                root = Path(self.config.project_dir).resolve() if self.config.project_dir else Path(".").resolve()
                try:
                    result = _search_code(
                        pattern=pattern,
                        root=root,
                        file_pattern=file_pattern,
                        max_results=max_results,
                        context_lines=context_lines,
                        case_sensitive=case_sensitive,
                        regex=regex,
                        fixed_strings=fixed_strings,
                    )
                    return result.formatted(max_results=max_results, context_lines=context_lines)
                except Exception as e:
                    logger.error("search_code error: %s", e)
                    return f"Error searching code: {e}"

            # -- Run tests --
            if name == "run_tests":
                command = args.get("command", "")
                timeout = args.get("timeout", 120)
                root = Path(self.config.project_dir).resolve() if self.config.project_dir else Path(".").resolve()
                try:
                    result = _run_tests(
                        command=command if command else None,
                        root=root,
                        timeout=timeout,
                    )
                    parts = [result.summary]
                    if result.failures:
                        parts.append(f"\nFailures ({len(result.failures)}):")
                        for f in result.failures[:10]:
                            parts.append(f"  • {f.summary()}")
                        if len(result.failures) > 10:
                            parts.append(f"  ... and {len(result.failures) - 10} more")
                    if result.stdout:
                        # Show last 20 lines of output
                        lines = result.stdout.splitlines()
                        tail = lines[-20:] if len(lines) > 20 else lines
                        parts.append("\nOutput (last lines):")
                        parts.extend(tail)
                    return "\n".join(parts)
                except Exception as e:
                    logger.error("run_tests error: %s", e)
                    return f"Error running tests: {e}"

            # -- Auto-correct tests --
            if name == "auto_correct_tests":
                test_command = args.get("test_command", "")
                max_iterations = args.get("max_iterations", 5)
                timeout = args.get("timeout", 120)
                root = Path(self.config.project_dir).resolve() if self.config.project_dir else Path(".").resolve()

                # Define the fix function that uses a subagent to fix failures
                def _fix_with_subagent(failures: list[TestFailure], raw_output: str) -> str:
                    """Use a subagent to fix test failures."""
                    if not failures and not raw_output:
                        return ""
                    prompt = format_failures_for_llm(failures, raw_output)
                    prompt += (
                        "\n\nAnalyze the test failures above and fix the source code. "
                        "Use read_file to understand the code, then apply_diff to fix it. "
                        "Focus on making the tests pass. Return a summary of what you fixed."
                    )
                    try:
                        agent = self.subagents.spawn("test_fixer", system_prompt=(
                            "You are a test-fixing specialist. Analyze test failures, "
                            "identify root causes in the source code, and fix them. "
                            "Be precise and minimal in your changes."
                        ))
                        result = agent.execute(prompt)
                        if result.error:
                            return f"Subagent error: {result.error}"
                        return result.output[:500]
                    except Exception as e:
                        return f"Fix error: {e}"

                try:
                    correction = auto_correct_loop(
                        fix_function=_fix_with_subagent,
                        root=root,
                        test_command=test_command if test_command else None,
                        max_iterations=max_iterations,
                        timeout=timeout,
                    )
                    parts = []
                    if correction.success:
                        parts.append(f"✅ All tests passed after {correction.iterations} iteration(s)!")
                    else:
                        parts.append(f"❌ Tests still failing after {correction.iterations} iteration(s).")
                    if correction.corrections:
                        parts.append("\nCorrections applied:")
                        for c in correction.corrections:
                            parts.append(f"  • {c}")
                    if correction.errors:
                        parts.append("\nIssues:")
                        for e in correction.errors:
                            parts.append(f"  • {e}")
                    if correction.final_result:
                        parts.append(f"\nFinal: {correction.final_result.summary}")
                    return "\n".join(parts)
                except Exception as e:
                    logger.error("auto_correct_tests error: %s", e)
                    return f"Error in auto-correction loop: {e}"

            if name == "finish":
                summary = args.get("summary", "")
                result = args.get("result", "")
                return f"[TASK COMPLETE]\nSummary: {summary}\nResult: {result}"

            # -- MCP tools --
            if name.startswith("mcp_"):
                return self.mcp.call_tool(name, args)

            # -- Skill tools --
            if name.startswith("skill_"):
                return self.skills.execute_skill(name[6:], args)

            logger.warning("Unknown tool called: %s", name)
            return f"Unknown tool: {name}"

        except Exception as e:
            logger.error("Tool '%s' error: %s", name, e)
            self.audit.log_error("tool_execution", str(e), {"tool": name, "args": args})
            return f"Tool '{name}' error: {e}"

    def reset_context(self) -> None:
        self.context.clear()
        self.call_depth = 0

    def shutdown(self) -> None:
        self.mcp.close_all()
        self.sandbox.restore_cwd()
        self.memory.save_all()
        self.audit.close()
        self.json_logger.close()