"""Central tool registry and execution dispatcher for Nyx."""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from nyx.providers.base import ToolCall, ToolDefinition
from nyx.web_search import search_web, format_search_results, fetch_page
from nyx.sandbox import PathTraversalError
from nyx.permissions import PermissionLevel
from nyx.repo_map import build_repo_map, build_repo_map_short
from nyx.search_code import search_code as _search_code
from nyx.test_loop import run_tests as _run_tests
from nyx.test_loop import auto_correct_loop, format_failures_for_llm, TestFailure

logger = logging.getLogger(__name__)


IGNORED_DIRS: set[str] = {
    ".git", ".venv", "venv", "node_modules", ".nyx", "__pycache__",
    ".pytest_cache", ".nyx_memory", "build", "dist", "nyx.egg-info"
}


def truncate_output(text: str, max_lines: int = 200, head_lines: int = 50, tail_lines: int = 100) -> str:
    """Truncate output text if it exceeds max_lines, keeping first head_lines and last tail_lines."""
    if not text:
        return text
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text

    head = lines[:head_lines]
    tail = lines[-tail_lines:]
    middle = f"\n\n... [TRUNCATED {len(lines) - head_lines - tail_lines} LINES OF OUTPUT TO PRESERVE CONTEXT WINDOW] ...\n\n"
    return "\n".join(head) + middle + "\n".join(tail)


# ---------------------------------------------------------------------------
# Tool definitions
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
                "mode": {
                    "type": "string",
                    "description": "Extraction mode: 'clean' (attempts to extract main article text and ignore navigation/menus) or 'raw' (all text)",
                    "enum": ["clean", "raw"],
                    "default": "clean"
                },
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
        description="Read the contents of a file from the filesystem. Can read specific line ranges to manage context size.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file (absolute or relative to project root)"},
                "start_line": {"type": "integer", "description": "Line number to start reading from (1-indexed, inclusive)", "default": 1},
                "end_line": {"type": "integer", "description": "Line number to stop reading at (1-indexed, inclusive). If not provided, reads until the end of the file."},
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
                "content": {"type": "string", "content": "Content to append"},
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
# Tool Context definition
# ---------------------------------------------------------------------------

@dataclass
class ToolContext:
    """Carries environment dependencies and security guards for execution."""
    config: Any
    sandbox: Any
    permissions: Any
    audit: Any
    patch_tool: Any
    memory: Any
    mcp: Any = None
    skills: Any = None
    subagents: Any = None
    async_subagents: Any = None
    on_command_approval: Callable[[str], tuple[bool, str]] | None = None
    on_file_approval: Callable[[str, str, str], tuple[bool, str]] | None = None
    subagent_tools: list[ToolDefinition] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Danger command checking helpers
# ---------------------------------------------------------------------------

DANGEROUS_PATTERNS: list[str] = [
    "rm", "rmdir", "mv", "cp", "chmod", "chown", "dd",
    "sudo", "su", "passwd", "kill",
    "mkfs", "fdisk", "mount", "umount", "iptables",
    "wget", "curl", "apt", "yum", "dnf", "pacman",
    "pip install", "npm install", "git push", "git reset",
    "git rebase", "git merge", "git cherry-pick",
    "docker", "systemctl", "journalctl",
]

DANGEROUS_OPERATORS: list[str] = [
    ">", ">>", "|",
]

def is_dangerous_command(command: str) -> bool:
    """Check if a command matches dangerous patterns using word-boundary matching."""
    cmd_lower = command.strip().lower()
    for op in DANGEROUS_OPERATORS:
        if re.search(rf'(?:^|\s){re.escape(op)}(?:\s|$)', cmd_lower):
            return True
    for pattern in DANGEROUS_PATTERNS:
        if re.search(rf'\b{re.escape(pattern)}\b', cmd_lower):
            return True
    return False

# ---------------------------------------------------------------------------
# Central tool execution logic
# ---------------------------------------------------------------------------

def execute_tool(tc: ToolCall, context: ToolContext) -> str:
    """Execute a single tool call and return the result as a string."""
    name = tc.name
    args = tc.arguments

    try:
        # -- Web tools --
        if name == "web_search":
            query = args.get("query", "")
            max_results = min(args.get("max_results", 5), 10)
            if context.config.web_search_enabled:
                results = search_web(query, context.config.web_search_provider, max_results)
                return format_search_results(results)
            return "Web search is disabled in config."

        if name == "web_fetch":
            return fetch_page(args.get("url", ""), mode=args.get("mode", "clean"))

        # -- Subagent tools --
        if name == "subagent_run":
            s_name = args.get("name", "unnamed")
            s_task = args.get("task", "")
            s_prompt = args.get("system_prompt", "")
            if not context.subagents:
                return "[subagent_run] Subagent manager not available."
            agent = context.subagents.spawn(s_name, s_prompt)
            # Pass controlled tool subset
            result = agent.execute(s_task, tools=context.subagent_tools if context.subagent_tools else None)
            if result.error:
                return f"[Subagent:{s_name}] Error: {result.error}"
            return f"[Subagent:{s_name}] Result:\n{result.output}"

        if name == "parallel_subagents":
            from nyx.async_subagent import ParallelTask
            tasks_data = args.get("tasks", [])
            if not tasks_data:
                return "[parallel_subagents] No tasks provided."
            if not context.async_subagents:
                return "[parallel_subagents] Async subagent manager not available."
            tasks = [
                ParallelTask(
                    name=t.get("name", f"task_{i}"),
                    task=t["task"],
                    context=t.get("context", ""),
                    system_prompt=t.get("system_prompt", ""),
                )
                for i, t in enumerate(tasks_data)
            ]
            result = context.async_subagents.run_parallel(tasks)
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
            context.memory.add_entry("memory", note)
            context.memory._save_note(note, tags_raw)
            tag_info = f" (tags: {tags_raw})" if tags_raw else ""
            return f"Note saved to memory{tag_info}: {note[:100]}..."

        if name == "memory_recall":
            query = args.get("query", "")
            if not query:
                return "Please provide a query to search for."
            relevant = []
            for c_id, conv in context.memory.conversations.items():
                if query.lower() in conv.title.lower():
                    relevant.append(f"- [{c_id[:8]}] {conv.title} ({len(conv.entries)} messages): {conv.summary[:200]}")
                if query.lower() in conv.summary.lower():
                    relevant.append(f"- [{c_id[:8]}] {conv.title} ({len(conv.entries)} messages): {conv.summary[:200]}")
                for entry in conv.entries:
                    if query.lower() in entry.content.lower():
                        c_title = conv.title[:40]
                        preview = entry.content[:200]
                        relevant.append(f"- [{c_id[:8]}] {c_title}: {preview}")
            notes = context.memory._load_notes()
            for note in notes:
                if query.lower() in note["content"].lower() or query.lower() in note.get("tags", "").lower():
                    relevant.append(f"- [NOTE] {note['content'][:200]}")
            if not relevant:
                return f"No relevant memories found for: {query}"
            return "Relevant memories:\n" + "\n".join(relevant[:10])

        # -- Shell command execution (with permissions + sandbox) --
        if name == "execute_command":
            command = args.get("command", "").strip()
            timeout = args.get("timeout", 30)

            if not command:
                logger.warning("Empty command received from AI (args=%s)", args)
                return (
                    "[ERROR] Empty command received. You must provide a valid shell command "
                    "in the 'command' parameter. For example: 'ls -la', 'cat file.txt', "
                    "'python3 script.py', 'which python3', etc."
                )

            # 1. Check permissions (granular permission model)
            approved, reason, perm_level = context.permissions.authorize_shell(command)
            context.audit.log_permission_check("shell", command, perm_level.value, approved, reason)

            if not approved:
                if perm_level == PermissionLevel.DENY:
                    logger.warning("Command explicitly denied: %s", command)
                    return (
                        f"[SECURITY] Command denied by security policy: '{command[:200]}'\n"
                        f"Reason: {reason}\n"
                        f"This command is not allowed under any circumstances."
                    )
                logger.warning("User denied command: %s", command)
                return (
                    f"[SECURITY] Command denied by user: '{command[:200]}'\n"
                    f"Reason: {reason}\n"
                    f"Please try a different approach that doesn't require this command."
                )

            # 2. Legacy dangerous patterns check (backward compatibility)
            if is_dangerous_command(command):
                if context.on_command_approval:
                    approved, reason = context.on_command_approval(command)
                else:
                    logger.warning("No approval callback configured, denying dangerous command: %s", command)
                    approved, reason = False, "No approval mechanism configured. This command requires manual approval."
                if not approved:
                    logger.warning("User denied dangerous command: %s", command)
                    return (
                        f"[SECURITY] Command denied by user: '{command[:200]}'\n"
                        f"Reason: {reason}\n"
                        f"Please try a different approach that doesn't require this command."
                    )
                logger.info("User approved dangerous command: %s", command)

            # 3. Sandbox wrapping
            final_command = context.sandbox.prepare_command(command)

            # 4. Execute
            logger.info("Executing command: %s", command)
            env = os.environ.copy()
            cwd = context.sandbox.root_str if context.sandbox.root else "."
            bin_dirs = []
            for venv_name in (".venv", "venv", "env", ".env"):
                venv_bin = Path(cwd) / venv_name / ("Scripts" if os.name == "nt" else "bin")
                if venv_bin.exists():
                    bin_dirs.append(str(venv_bin))
                    break
            node_bin = Path(cwd) / "node_modules" / ".bin"
            if node_bin.exists():
                bin_dirs.append(str(node_bin))
            if bin_dirs:
                env["PATH"] = os.pathsep.join(bin_dirs) + os.pathsep + env.get("PATH", "")

            try:
                proc = subprocess.run(
                    final_command, shell=True, capture_output=True, text=True, timeout=timeout, env=env,
                )
                out = proc.stdout or ""
                err = proc.stderr or ""
                if proc.returncode != 0:
                    logger.warning("Command exit code %d: %s", proc.returncode, command)
                    trunc_out = truncate_output(out, max_lines=200, head_lines=50, tail_lines=100)
                    trunc_err = truncate_output(err, max_lines=200, head_lines=50, tail_lines=100)
                    return f"Exit code: {proc.returncode}\nstdout:\n{trunc_out}\nstderr:\n{trunc_err}"
                logger.debug("Command succeeded: %s", command)
                trunc_out = truncate_output(out, max_lines=200, head_lines=50, tail_lines=100)
                return trunc_out or "(no output)"
            except subprocess.TimeoutExpired:
                logger.warning("Command timed out (%ds): %s", timeout, command)
                return f"Command timed out after {timeout}s."
            except Exception as e:
                logger.error("Command error: %s", e)
                return f"Command error: {e}"

        # -- File read (with sandbox path resolution) --
        if name == "read_file":
            raw_path = args.get("path", "")
            start_line = args.get("start_line", 1)
            end_line = args.get("end_line")
            try:
                resolved = context.sandbox.safe_read_path(raw_path)
            except PathTraversalError as e:
                context.audit.log_security_event("path_traversal_blocked", {
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
                content = resolved.read_text(encoding="utf-8")
                lines = content.splitlines(keepends=True)
                total_lines = len(lines)
                
                # 1-indexed conversion
                s_idx = max(0, start_line - 1)
                if end_line is not None:
                    e_idx = min(total_lines, end_line)
                else:
                    e_idx = total_lines
                    
                sliced_content = "".join(lines[s_idx:e_idx])
                context.audit.log_file_operation("read", str(resolved), len(sliced_content))
                
                # Prefix with line info header
                header = f"[File: {raw_path} | Lines {start_line} to {min(end_line or total_lines, total_lines)} of {total_lines}]\n"
                return header + sliced_content
            except Exception as e:
                context.audit.log_file_operation("read", str(resolved), 0, success=False, error=str(e))
                return f"Error reading file: {e}"

        # -- File write (via PatchTool with diff/approval) --
        if name == "write_file":
            raw_path = args.get("path", "")
            content = args.get("content", "")
            try:
                resolved = context.sandbox.resolve(raw_path, for_write=True)
            except PathTraversalError as e:
                context.audit.log_security_event("path_traversal_blocked", {
                    "path": raw_path,
                    "resolved": str(e.resolved),
                    "root": e.root,
                })
                return f"[SECURITY] Path traversal blocked: {raw_path}"
            except Exception as e:
                return f"Error resolving path: {e}"

            success, message = context.patch_tool.propose_write(str(resolved), content)
            context.audit.log_file_operation(
                "write", str(resolved), len(content), success=success,
                error="" if success else message,
            )
            return message

        # -- Apply diff (with patch parsing, validation, conflict detection) --
        if name == "apply_diff":
            raw_path = args.get("path", "")
            diff_text = args.get("diff", "")
            description = args.get("description", "")

            try:
                resolved = context.sandbox.resolve(raw_path, for_write=True)
            except PathTraversalError as e:
                context.audit.log_security_event("path_traversal_blocked", {
                    "path": raw_path,
                    "resolved": str(e.resolved),
                    "root": e.root,
                })
                return f"[SECURITY] Path traversal blocked: {raw_path}"
            except Exception as e:
                return f"Error resolving path: {e}"

            perm_level = context.permissions.check_file_write(str(resolved))
            context.audit.log_permission_check(
                "filesystem", str(resolved), perm_level.value,
                perm_level != PermissionLevel.DENY, "",
            )
            if perm_level == PermissionLevel.DENY:
                return (
                    f"[SECURITY] File write explicitly denied by policy: '{raw_path}'\n"
                    f"This path is blocked under all circumstances."
                )

            success, message = context.patch_tool.propose_patch(str(resolved), diff_text)
            context.audit.log_file_operation(
                "apply_diff", str(resolved), len(diff_text), success=success,
                error="" if success else message,
            )

            # Auto-fallback: if a conflict is detected, try write_file with reconstructed content
            if not success and message.startswith("[CONFLICT]"):
                from nyx.diff_tool import _apply_unified_diff_to_content, _apply_search_replace_to_content, _SEARCH_MARKER_RE
                original_content = ""
                if resolved.exists():
                    try:
                        original_content = resolved.read_text(encoding="utf-8")
                    except Exception:
                        pass
                proposed_content = None
                lines = diff_text.splitlines()
                if any(_SEARCH_MARKER_RE.match(l) for l in lines):
                    proposed_content = _apply_search_replace_to_content(original_content, diff_text)
                else:
                    proposed_content = _apply_unified_diff_to_content(original_content, diff_text)

                if proposed_content is not None:
                    logger.info("apply_diff conflict: falling back to write_file for %s", resolved)
                    fb_success, fb_message = context.patch_tool.propose_write(str(resolved), proposed_content)
                    context.audit.log_file_operation(
                        "write_file_fallback", str(resolved), len(proposed_content),
                        success=fb_success, error="" if fb_success else fb_message,
                    )
                    fallback_note = "[auto-fallback from apply_diff] " if fb_success else "[fallback also failed] "
                    return fallback_note + fb_message
                return message

            return message

        # -- Rollback file --
        if name == "rollback_file":
            raw_path = args.get("path", "")
            try:
                resolved = context.sandbox.resolve(raw_path, for_write=True)
            except PathTraversalError as e:
                return f"[SECURITY] Path traversal blocked: {raw_path}"
            except Exception as e:
                return f"Error resolving path: {e}"

            success, message = context.patch_tool.rollback_last(str(resolved))
            context.audit.log_file_operation(
                "rollback", str(resolved), 0, success=success,
                error="" if success else message,
            )
            return message

        # -- Patch history --
        if name == "patch_history":
            limit = args.get("limit", 20)
            history = context.patch_tool.get_history(limit=limit)
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
            try:
                resolved = context.sandbox.resolve(raw_path, for_write=True)
            except PathTraversalError as e:
                context.audit.log_security_event("path_traversal_blocked", {
                    "path": raw_path,
                    "resolved": str(e.resolved),
                    "root": e.root,
                })
                return f"[SECURITY] Path traversal blocked: {raw_path}"
            except Exception as e:
                return f"Error resolving path: {e}"

            approved, reason, perm_level = context.permissions.authorize_file_write(str(resolved))
            context.audit.log_permission_check("filesystem", str(resolved), perm_level.value, approved, reason)

            if not approved:
                return (
                    f"[SECURITY] File append denied: '{raw_path}'\n"
                    f"Reason: {reason}"
                )

            success, message = context.patch_tool.propose_append(str(resolved), content)
            context.audit.log_file_operation(
                "append", str(resolved), len(content), success=success,
                error="" if success else message,
            )
            return message

        # -- List files (with sandbox) --
        if name == "list_files":
            raw_path = args.get("path", ".")
            recursive = args.get("recursive", False)
            try:
                resolved = context.sandbox.resolve(raw_path)
            except PathTraversalError as e:
                context.audit.log_security_event("path_traversal_blocked", {
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
                files = []
                for f in resolved.rglob("*"):
                    rel_parts = f.relative_to(resolved).parts
                    if any(part in IGNORED_DIRS for part in rel_parts):
                        continue
                    files.append(str(f.relative_to(resolved)))
            else:
                files = []
                for f in resolved.iterdir():
                    if f.name in IGNORED_DIRS:
                        continue
                    files.append(f.name)
            return "\n".join(sorted(files)) if files else "(empty directory)"

        # -- Repo map --
        if name == "repo_map":
            path = args.get("path", "")
            short = args.get("short", False)
            root = Path(path).resolve() if path else (
                Path(context.config.project_dir).resolve() if context.config.project_dir else Path(".").resolve()
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
            root = Path(context.config.project_dir).resolve() if context.config.project_dir else Path(".").resolve()
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
            root = Path(context.config.project_dir).resolve() if context.config.project_dir else Path(".").resolve()
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
            root = Path(context.config.project_dir).resolve() if context.config.project_dir else Path(".").resolve()

            def _fix_with_subagent(failures: list[TestFailure], raw_output: str) -> str:
                if not failures and not raw_output:
                    return ""
                if not context.subagents:
                    return "Subagent manager not available."
                prompt = format_failures_for_llm(failures, raw_output)
                prompt += (
                    "\n\nAnalyze the test failures above and fix the source code. "
                    "Use read_file to understand the code, then apply_diff to fix it. "
                    "Focus on making the tests pass. Return a summary of what you fixed."
                )
                try:
                    agent = context.subagents.spawn("test_fixer", system_prompt=(
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
        if name.startswith("mcp_") and context.mcp:
            return context.mcp.call_tool(name, args)

        # -- Skill tools --
        if name.startswith("skill_") and context.skills:
            return context.skills.execute_skill(name[6:], args)

        logger.warning("Unknown tool called: %s", name)
        return f"Unknown tool: {name}"

    except Exception as e:
        logger.error("Tool '%s' error: %s", name, e)
        if context.audit:
            context.audit.log_error("tool_execution", str(e), {"tool": name, "args": args})
        return f"Tool '{name}' error: {e}"
