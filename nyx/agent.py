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
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from nyx.config import Config, MODE_SYSTEM_PROMPTS, AUTONOMY_CONFIGS, ARCHITECT_TOOLS
from nyx.providers.base import BaseLLMProvider, ToolCall, ToolDefinition
from nyx.providers import get_provider
from nyx.mcp_client import MCPManager
from nyx.skill_manager import SkillManager
from nyx.subagent import SubagentManager
from nyx.async_subagent import AsyncSubagentManager
from nyx.memory import MemoryManager
from nyx.permissions import PermissionManager, PermissionLevel
from nyx.sandbox import Sandbox
from nyx.audit import AuditTrail
from nyx.json_logger import JSONLogger
from nyx.diff_tool import (
    PatchTool,
)
from nyx.repo_map import build_repo_map_short

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

from nyx.tools import BUILTIN_TOOLS, ToolContext



# ---------------------------------------------------------------------------
# AgentContext
# ---------------------------------------------------------------------------


def _semantic_code_fold(content: str, max_chars: int) -> str:
    """Fold code blocks in the middle of a file to preserve signatures and fit max_chars."""
    lines = content.splitlines()
    if len(lines) < 80:
        return _middle_truncate(content, max_chars)

    # Detect coding constructs
    code_indicators = ("import ", "def ", "class ", "fn ", "struct ", "impl ", "const ", "let ", "function ")
    is_code = sum(1 for line in lines if any(line.startswith(ind) for ind in code_indicators)) > 3

    if not is_code:
        return _middle_truncate(content, max_chars)

    keep_head = 50
    keep_tail = 50

    if len(lines) <= keep_head + keep_tail:
        return content

    head_part = lines[:keep_head]
    tail_part = lines[-keep_tail:]
    middle_part = lines[keep_head:-keep_tail]

    folded_lines = []
    in_fold = False

    for line in middle_part:
        if line.startswith(("def ", "class ", "function ", "async function ", "export class ")):
            folded_lines.append(line)
            comment = "//" if not line.startswith(("def ", "class ")) else "#"
            folded_lines.append(f"    {comment} ... [CODE FOLDED SEMANTICALLY TO PRESERVE CONTEXT] ...")
            in_fold = True
        elif in_fold and (line.startswith((" ", "\t")) or not line.strip()):
            continue
        else:
            in_fold = False
            folded_lines.append(line)

    reconstructed = "\n".join(head_part + folded_lines + tail_part)
    if len(reconstructed) <= max_chars:
        return reconstructed

    return _middle_truncate(content, max_chars)


def _middle_truncate(content: str, max_chars: int) -> str:
    """Fallback standard truncation in the middle."""
    half = max_chars // 2
    header = content[:half]
    footer = content[-half:]
    return (
        f"{header}\n\n... [TRUNCATED {len(content) - max_chars} CHARACTERS TO PRESERVE CONTEXT WINDOW] ...\n\n{footer}"
    )


def _truncate_to_max_tokens(content: str, max_chars: int = 16000) -> str:
    """Truncate content if it exceeds max_chars, using semantic code folding if possible."""
    if len(content) <= max_chars:
        return content
    return _semantic_code_fold(content, max_chars)


def _compress_text(text: str, keep_chars: int, threshold: int) -> str:
    """Middle-truncate tool output to save tokens."""
    if len(text) <= threshold:
        return text
    # Avoid re-compressing already compressed text
    if "... [Tool output compressed" in text:
        return text
    half = keep_chars // 2
    header = text[:half]
    footer = text[-half:]
    return (
        f"{header}\n\n"
        f"... [Tool output compressed from {len(text)} to {keep_chars} characters to save tokens] ...\n\n"
        f"{footer}"
    )


@dataclass
class AgentContext:
    """Conversation context for the agent."""
    messages: list[dict[str, Any]] = field(default_factory=list)
    max_history: int = 50
    token_optimizer: bool = False
    token_optimizer_threshold: int = 0
    token_optimizer_keep_chars: int = 0

    def add(self, role: str, content: str) -> None:
        truncated_content = _truncate_to_max_tokens(content)
        self.messages.append({"role": role, "content": truncated_content})
        if len(self.messages) > self.max_history:
            keep = [self.messages[0]] if self.messages[0]["role"] == "system" else []
            keep.extend(self.messages[-(self.max_history - len(keep)):])
            self.messages = keep
        self.compress_history()

    def add_tool_result(self, tool_call_id: str, name: str, content: str, use_anthropic_format: bool = False) -> None:
        truncated_content = _truncate_to_max_tokens(content)
        if use_anthropic_format:
            # Anthropic uses "tool_use_id" and content blocks with role "user" for tool results
            self.messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": [{"type": "text", "text": truncated_content}],
                    }
                ],
            })
        else:
            # OpenAI-compatible format
            self.messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": name,
                "content": truncated_content,
            })
        self.compress_history()

    def compress_history(self) -> None:
        """Compress older tool output messages to reduce token consumption."""
        pass  # Completely disabled to prevent amnesia

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

    # Advisory command-risk heuristics, not a security boundary. Clever shell
    # syntax, local scripts, aliases, or platform quirks can bypass keyword
    # matching; Docker/OS sandbox isolation is the real containment layer.
    # Matches trigger an interactive user approval prompt.
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
        """Return whether command matches advisory dangerous-command heuristics."""
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
        on_event: Callable[[dict[str, Any]], None] | None = None,
        on_command_approval: Callable[[str], tuple[bool, str]] | None = None,
        on_file_approval: Callable[[str, str, str], tuple[bool, str]] | None = None,
    ) -> None:
        self.config = config
        self.provider = provider or get_provider(config)
        self.mcp = mcp_manager or MCPManager(
            request_timeout=config.mcp_request_timeout,
            connect_timeout=config.mcp_connect_timeout,
            max_response_chars=config.mcp_max_response_chars,
            restart_on_failure=config.mcp_restart_on_failure,
            sandbox_enabled=config.mcp_sandbox_enabled,
            sandbox_docker_image=config.mcp_sandbox_docker_image,
            sandbox_network=config.mcp_sandbox_network,
            sandbox_read_only=config.mcp_sandbox_read_only,
            sandbox_project_dir=config.project_dir,
        )
        self.skills = skill_manager or SkillManager(
            config.skills_dir,
            process_isolation=config.skills_process_isolation,
            default_timeout_seconds=config.skills_default_timeout_seconds,
            max_output_chars=config.skills_max_output_chars,
        )
        self.subagents = subagent_manager or SubagentManager(
            config,
            process_isolation=config.subagents_process_isolation,
            default_timeout_seconds=config.subagents_default_timeout_seconds,
        )
        self.memory = memory_manager or MemoryManager(provider=self.provider)
        self.async_subagents = async_subagent_manager or AsyncSubagentManager(
            config=config,
            process_isolation=config.subagents_process_isolation,
            default_timeout_seconds=config.subagents_default_timeout_seconds,
        )
        self.on_token = on_token
        self.on_event = on_event
        self._user_on_command_approval = on_command_approval
        self._user_on_file_approval = on_file_approval
        self._on_command_approval = on_command_approval
        self._on_file_approval = on_file_approval
        self.context = AgentContext()
        self.call_depth = 0
        self.max_depth = config.agent_max_depth
        # Controlled tool subset for subagents (set after setup)
        self._subagent_tools: list[ToolDefinition] = []
        self.tool_context: ToolContext | None = None

        # -- Security subsystems --
        # Permission manager
        self.permissions = PermissionManager(config.permissions_config if config.permissions_config else None)

        # Sandbox (project root)
        self.sandbox = Sandbox(
            project_root=config.project_dir if config.sandbox_enabled else None,
            allow_paths=config.sandbox_allow_paths,
            deny_paths=config.sandbox_deny_paths,
            auto_chdir=config.sandbox_auto_chdir,
            use_docker=config.sandbox_use_docker,
            docker_image=config.sandbox_docker_image,
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

        # Collect base tools once; active tools are recalculated for each mode.
        self._base_tools: list[ToolDefinition] = list(BUILTIN_TOOLS)
        self._active_tools: list[ToolDefinition] = list(self._base_tools)

    def _emit(self, event_type: str, **payload: Any) -> None:
        """Emit a UI/logging event without coupling the agent to a renderer."""
        if self.on_event:
            self.on_event({"type": event_type, **payload})
            return
        if event_type == "tool_start":
            target = payload.get("target", "")
            suffix = f" ➔ {target}" if target else ""
            print(f"🛠️  [Tool Call] {payload.get('name')}{suffix}", flush=True)
        elif event_type == "tool_finish":
            status = "Success" if payload.get("ok") else "Failure"
            details = ", ".join(payload.get("details", []))
            target = payload.get("target", "")
            suffix = f" ➔ {target}" if target else ""
            print(f"✓  [Tool {status}] {payload.get('name')}{suffix} ({details})", flush=True)

    # ------------------------------------------------------------------
    # Approval callback properties (keep PermissionManager in sync)
    # ------------------------------------------------------------------

    @property
    def on_command_approval(self) -> Callable[[str], tuple[bool, str]] | None:
        return self._user_on_command_approval

    @on_command_approval.setter
    def on_command_approval(self, callback: Callable[[str], tuple[bool, str]] | None) -> None:
        self._user_on_command_approval = callback
        if hasattr(self, "permissions"):
            self._refresh_autonomy_callbacks()

    @property
    def on_file_approval(self) -> Callable[[str, str, str], tuple[bool, str]] | None:
        return self._user_on_file_approval

    @on_file_approval.setter
    def on_file_approval(self, callback: Callable[[str, str, str], tuple[bool, str]] | None) -> None:
        self._user_on_file_approval = callback
        if hasattr(self, "permissions"):
            self._refresh_autonomy_callbacks()

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
            self._emit("status", message="Connecting MCP servers...")

            # Wire up progress callback if available
            if self.json_logger and self.json_logger.enabled:
                self.mcp.set_progress_callback(
                    lambda label, cur, total: self.json_logger.log_event(
                        "mcp_progress", {"server": label, "current": cur, "total": total}
                    )
                )

            self.mcp.connect_all(self.config.mcp_servers)
            self._base_tools.extend(self.mcp.get_tool_definitions())

        # Wire up subagent progress callback
        if self.json_logger and self.json_logger.enabled:
            self.subagents.set_progress_callback(
                lambda label, cur, total: self.json_logger.log_event(
                    "subagent_progress", {"name": label, "current": cur, "total": total}
                )
            )

        # Skills
        skills_dir = self.config.skills_dir
        if skills_dir and self.config.skills_enabled:
            logger.info("Loading skills from %s", skills_dir)
            self._emit("status", message="Loading skills...")
            skills_found = self.skills.discover(skills_dir)
            if skills_found:
                self._base_tools.extend(self.skills.get_tool_definitions())
        elif skills_dir and not self.config.skills_enabled:
            logger.info("Skills disabled by configuration; skipping %s", skills_dir)

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
        self.subagents.set_tool_executor(self._execute_tool)
        self.async_subagents.set_default_tools(self._subagent_tools)
        self.async_subagents.set_tool_executor(self._execute_tool)
        logger.info("Subagent tools: %d (filtered from %d)", len(self._subagent_tools), len(self._active_tools))

        logger.info("Total tools available: %d", len(self._active_tools))
        self._emit("setup_complete", tool_count=len(self._active_tools))

        # -- ToolContext initialization --
        self.tool_context = ToolContext(
            config=self.config,
            sandbox=self.sandbox,
            permissions=self.permissions,
            audit=self.audit,
            patch_tool=self.patch_tool,
            memory=self.memory,
            mcp=self.mcp,
            skills=self.skills,
            subagents=self.subagents,
            async_subagents=self.async_subagents,
            on_command_approval=lambda cmd: self._request_command_approval(cmd),
            on_file_approval=lambda path, summary, diff: self._file_approval_handler(path, summary, diff),
            subagent_tools=self._subagent_tools,
        )
        self.subagents.set_context(self.tool_context)
        self.async_subagents.set_context(self.tool_context)

    def _inject_memory_summary(self) -> None:
        """Inject a summary of past conversations into the LLM context."""
        try:
            current_id = self.memory.current.id if self.memory.current else ""
            convs = [
                conv for conv in self.memory.list_conversations()
                if conv.get("id") != current_id
            ]
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

        # 2. Recalculate active tools for the selected mode.
        if mode == "architect":
            self._active_tools = [t for t in self._base_tools if t.name in ARCHITECT_TOOLS]
            logger.info("Architect mode: tools filtered to %d read-only tools", len(self._active_tools))
        else:
            self._active_tools = list(self._base_tools)

        # 3. Apply autonomy-level behaviour
        aut = AUTONOMY_CONFIGS[autonomy]
        multiplier = aut["max_depth_multiplier"]
        self.max_depth = self.config.agent_max_depth * multiplier
        logger.info("Autonomy '%s': max_depth=%d", autonomy, self.max_depth)

        self._refresh_autonomy_callbacks()
        self._subagent_tools = self._get_subagent_tools()
        if hasattr(self, "subagents"):
            self.subagents.set_default_tools(self._subagent_tools)
        if hasattr(self, "async_subagents"):
            self.async_subagents.set_default_tools(self._subagent_tools)
        if self.tool_context:
            self.tool_context.subagent_tools = self._subagent_tools

        self._emit("mode", mode=mode, autonomy=autonomy, max_depth=self.max_depth)

    def _refresh_autonomy_callbacks(self) -> None:
        """Recalculate effective callbacks from user callbacks and autonomy level."""
        autonomy = self.config.agent_autonomy.lower()
        aut = AUTONOMY_CONFIGS.get(autonomy, AUTONOMY_CONFIGS["ask"])

        self._on_file_approval = (
            (lambda path, summary, diff: (True, ""))
            if aut["auto_approve_files"]
            else self._user_on_file_approval
        )
        self._on_command_approval = (
            (lambda cmd: (True, ""))
            if aut["auto_approve_commands"]
            else self._user_on_command_approval
        )

        def _pm_callback(cat: str, desc: str, target: str) -> tuple[bool, str]:
            if cat == "filesystem" and aut["auto_approve_files"]:
                return True, ""
            if cat == "shell" and aut["auto_approve_commands"]:
                return True, ""
            if cat == "shell" and self._user_on_command_approval:
                return self._user_on_command_approval(target)
            if cat == "filesystem" and self._user_on_file_approval:
                return self._user_on_file_approval(target, desc, "")
            return False, "No approval mechanism configured."

        self.permissions.set_approval_callback(_pm_callback)
        if hasattr(self, "patch_tool"):
            self.patch_tool.set_approval_callback(self._file_approval_handler)

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
            t for t in self._active_tools
            if t.name in self.SUBAGENT_TOOL_WHITELIST
        ]

    @property
    def tools(self) -> list[ToolDefinition]:
        return self._active_tools

    @staticmethod
    def _tool_target(name: str, arguments: dict[str, Any]) -> str:
        """Return a short human-readable target for a tool call."""
        if name in ("read_file", "write_file", "apply_diff", "append_file", "rollback_file"):
            return str(arguments.get("path", ""))
        if name == "list_files":
            return str(arguments.get("path", "."))
        if name == "execute_command":
            command = str(arguments.get("command", "")).strip()
            return (command[:60] + "...") if len(command) > 60 else command
        if name == "web_search":
            return str(arguments.get("query", ""))
        if name == "web_fetch":
            return str(arguments.get("url", ""))
        if name == "subagent_run":
            return str(arguments.get("name", ""))
        if name == "parallel_subagents":
            tasks = arguments.get("tasks", [])
            return f"{len(tasks)} tasks"
        if name == "memory_save":
            note = str(arguments.get("note", "")).strip()
            return (note[:40] + "...") if len(note) > 40 else note
        if name == "memory_recall":
            return str(arguments.get("query", ""))
        if name == "repo_map":
            return str(arguments.get("path", ""))
        if name == "search_code":
            return str(arguments.get("pattern", ""))
        if name == "run_tests":
            return str(arguments.get("command", "") or "(default)")

        for key in ("path", "file", "filepath", "filename", "uri", "url", "command", "query"):
            if key in arguments:
                value = str(arguments[key]).strip()
                if value:
                    return (value[:60] + "...") if len(value) > 60 else value
        return ""

    @staticmethod
    def _tool_result_details(name: str, arguments: dict[str, Any], result: str, duration_ms: float) -> list[str]:
        """Return compact facts about a completed tool call."""
        details: list[str] = []
        is_error = "error" in result.lower() or "[error]" in result.lower() or "denied" in result.lower()
        if not is_error:
            import re
            if name == "read_file":
                details.append(f"{len(result.splitlines())} lines read")
            elif name in ("write_file", "append_file"):
                details.append(f"{len(str(arguments.get('content', '')).splitlines())} lines written")
            elif name == "apply_diff":
                details.append(f"{len(str(arguments.get('diff', '')).splitlines())} lines diff")
            elif name == "execute_command" and result.strip() and result != "(no output)":
                details.append(f"{len(result.splitlines())} lines output")
            elif name == "subagent_run":
                m = re.search(r"completed in (\d+) step", result)
                if m:
                    details.append(f"{m.group(1)} steps")
                elif "completed" in result.lower():
                    details.append("completed")
                else:
                    details.append("failed")
            elif name == "parallel_subagents":
                m = re.search(r"Parallel execution: (\d+) completed", result)
                if m:
                    details.append(f"{m.group(1)} subagents completed")
                else:
                    details.append("execution finished")
            elif name == "web_search":
                if "No results found." in result:
                    details.append("0 results found")
                else:
                    count = len(re.findall(r"^\[\d+\]", result, re.M))
                    details.append(f"{count} results found")
            elif name == "web_fetch":
                details.append(f"{len(result)} chars fetched")
            elif name == "search_code":
                m = re.search(r"Found (\d+) matches", result)
                if m:
                    details.append(f"{m.group(1)} matches")
                else:
                    details.append("0 matches")
            elif name == "repo_map":
                details.append(f"{len(result.splitlines())} lines repo map")
            elif name == "run_tests":
                if "All" in result and "passed" in result:
                    details.append("tests passed")
                elif "failed" in result:
                    details.append("tests failed")
                else:
                    details.append("finished")
            elif name == "auto_correct_tests":
                m = re.search(r"after (\d+) iteration", result)
                iters = f" ({m.group(1)} iterations)" if m else ""
                if "passed" in result:
                    details.append(f"tests passed{iters}")
                else:
                    details.append(f"tests failed{iters}")
            elif name == "list_files":
                details.append(f"{len(result.splitlines())} files listed")
            elif name == "find_references":
                if "No references found" in result:
                    details.append("0 references")
                else:
                    m = re.search(r"\.\.\. and (\d+) more references\.", result)
                    lines_count = len(result.splitlines())
                    if m:
                        total = 100 + int(m.group(1))
                        details.append(f"{total} references")
                    else:
                        details.append(f"{lines_count} references")
            elif name == "memory_recall":
                if "No relevant memories found" in result:
                    details.append("0 memories recalled")
                else:
                    count = len(re.findall(r"\[score:", result))
                    details.append(f"{count} memories recalled")
            elif name == "memory_save":
                details.append("saved")

        duration = f"{duration_ms / 1000:.2f}s" if duration_ms >= 1000 else f"{int(duration_ms)}ms"
        details.append(f"took {duration}")
        return details

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
                target = self._tool_target(tc.name, tc.arguments)
                self._emit("tool_start", name=tc.name, target=target, arguments=tc.arguments)

                result = self._execute_tool(tc)
                duration = (time.time() - t_start) * 1000
                is_err = "error" in result.lower() or "[error]" in result.lower() or "denied" in result.lower()
                self._emit(
                    "tool_finish",
                    name=tc.name,
                    target=target,
                    ok=not is_err,
                    details=self._tool_result_details(tc.name, tc.arguments, result, duration),
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
        if not self.tool_context:
            from nyx.tools import ToolContext
            self.tool_context = ToolContext(
                config=self.config,
                sandbox=self.sandbox,
                permissions=self.permissions,
                audit=self.audit,
                patch_tool=self.patch_tool,
                memory=self.memory,
                mcp=self.mcp,
                skills=self.skills,
                subagents=self.subagents,
                async_subagents=self.async_subagents,
                on_command_approval=lambda cmd: self._request_command_approval(cmd),
                on_file_approval=lambda path, summary, diff: self._file_approval_handler(path, summary, diff),
                subagent_tools=self._subagent_tools,
            )
        # Sync current subagent tools list in context
        self.tool_context.subagent_tools = self._subagent_tools

        t_start = time.time()
        # Log tool call attempt
        self.json_logger.log_tool_call(
            tool_name=tc.name,
            arguments=tc.arguments,
        )

        from nyx.tools import execute_tool
        result = execute_tool(tc, self.tool_context)

        duration = (time.time() - t_start) * 1000

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

        return result

    def load_conversation_history(self) -> None:
        """Load conversation entries from the current memory session into the agent context."""
        # 1. Remove any previous conversation summary from system messages to avoid accumulation
        self.context.messages = [
            m for m in self.context.messages
            if not (m.get("role") == "system" and m.get("content", "").startswith("[Previous conversation summary]"))
        ]

        # 2. Clear any existing user/assistant/tool messages
        self.context.clear()

        # 3. Retrieve messages from memory
        max_tokens = getattr(self.config, "max_tokens", 32000)
        messages = self.memory.get_context_messages(max_tokens=max_tokens, include_summary=True)

        # 4. Add them to context messages
        for msg in messages:
            self.context.messages.append(msg.copy())

    def reset_context(self) -> None:
        self.context.clear()
        self.call_depth = 0

    def shutdown(self) -> None:
        self.mcp.close_all()
        self.sandbox.restore_cwd()
        self.memory.save_all()
        self.audit.close()
        self.json_logger.close()
