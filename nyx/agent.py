"""
Nyx — the core agentic loop.

Orchestrates: LLM calls → tool execution → context management.
Supports: skills, MCP tools, web search, subagents (sync + parallel),
          persistent memory, summarisation, code execution.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from nyx.config import Config
from nyx.providers.base import BaseLLMProvider, LLMResponse, ToolCall, ToolDefinition
from nyx.providers import get_provider
from nyx.mcp_client import MCPManager
from nyx.skill_manager import SkillManager
from nyx.subagent import SubagentManager
from nyx.async_subagent import AsyncSubagentManager, ParallelTask
from nyx.memory import MemoryManager
from nyx.web_search import search_web, format_search_results, fetch_page

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
        description="Execute a shell command on the local system. Only safe, read-only commands are allowed (ls, cat, grep, find, git status, etc.).",
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
        description="Write content to a file on the filesystem. Creates directories if needed.",
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

    def add_tool_result(self, tool_call_id: str, name: str, content: str) -> None:
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

    # ------------------------------------------------------------------
    # Command sandbox — safe commands (read-only, non-destructive)
    # ------------------------------------------------------------------
    SAFE_COMMANDS: list[str] = [
        "ls", "cat", "head", "tail", "echo", "pwd", "which", "whoami",
        "date", "uname", "find", "grep", "wc", "sort", "diff", "file",
        "python --version", "python3 --version", "pip list", "pip freeze",
        "node --version", "npm --version", "git status", "git log",
        "git diff", "git branch", "git remote",
    ]

    # Commands that are explicitly blocked (destructive)
    DANGEROUS_PATTERNS: list[str] = [
        "rm ", "rmdir ", "mv ", "cp ", "chmod ", "chown ", "dd ",
        ">", ">>", "|", "sudo ", "su ", "passwd", "kill ",
        "mkfs", "fdisk", "mount", "umount", "iptables",
        "wget ", "curl ", "apt ", "yum ", "dnf ", "pacman",
        "pip install", "npm install", "git push", "git reset",
        "git rebase", "git merge", "git cherry-pick",
    ]

    @staticmethod
    def _is_safe_command(command: str) -> bool:
        """Check if a command is in the safe list."""
        cmd_stripped = command.strip()
        for safe in Agent.SAFE_COMMANDS:
            if cmd_stripped == safe or cmd_stripped.startswith(safe + " "):
                return True
        return False

    @staticmethod
    def _is_dangerous_command(command: str) -> bool:
        """Check if a command matches dangerous patterns."""
        cmd_lower = command.strip().lower()
        for pattern in Agent.DANGEROUS_PATTERNS:
            if pattern in cmd_lower:
                return True
        return False

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
        self.context = AgentContext()
        self.call_depth = 0
        self.max_depth = 15

        # Collect all tools
        self._all_tools: list[ToolDefinition] = list(BUILTIN_TOOLS)

    def setup(self) -> None:
        """Connect MCP servers and discover skills."""
        # MCP
        if self.config.mcp_servers:
            logger.info("Connecting MCP servers...")
            print("\n🔌 Connecting MCP servers...")
            self.mcp.connect_all(self.config.mcp_servers)
            self._all_tools.extend(self.mcp.get_tool_definitions())

        # Skills
        skills_dir = self.config.skills_dir
        if skills_dir:
            logger.info("Loading skills from %s", skills_dir)
            print("🧠 Loading skills...")
            skills_found = self.skills.discover(skills_dir)
            if skills_found:
                self._all_tools.extend(self.skills.get_tool_definitions())

        # Project directory
        if self.config.project_dir:
            self.context.add("system", f"The current working directory is: {self.config.project_dir}")

        logger.info("Total tools available: %d", len(self._all_tools))
        print(f"\n📦 Total tools available: {len(self._all_tools)}")

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
            self.call_depth -= 1
            logger.warning("Max reasoning depth reached (%d)", self.max_depth)
            return "I've reached the maximum number of reasoning steps. Please ask me to continue or refine your request."

        logger.debug("LLM call (depth=%d, messages=%d)", self.call_depth, len(self.context.messages))
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
            return f"Error calling LLM: {e}"

        # Handle tool calls
        if response.tool_calls:
            tool_names = [tc.name for tc in response.tool_calls]
            logger.info("Tool calls: %s", tool_names)
            self.context.messages.append({
                "role": "assistant",
                "content": response.content or None,
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    for tc in response.tool_calls
                ],
            })

            for tc in response.tool_calls:
                result = self._execute_tool(tc)
                self.context.add_tool_result(tc.id, tc.name, result)

            # Continue the loop
            result = self._loop()
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

            # -- Subagent tools --
            if name == "subagent_run":
                s_name = args.get("name", "unnamed")
                s_task = args.get("task", "")
                s_prompt = args.get("system_prompt", "")
                agent = self.subagents.spawn(s_name, s_prompt)
                result = agent.execute(s_task)
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
                tags = args.get("tags", "")
                self.memory.add_entry("user", f"[SAVED NOTE:{tags}] {note}")
                return f"Note saved to memory: {note[:100]}..."

            if name == "memory_recall":
                query = args.get("query", "")
                convs = self.memory.list_conversations()
                relevant = []
                for c in convs:
                    if query.lower() in c.get("summary", "").lower() or query.lower() in c.get("title", "").lower():
                        relevant.append(f"- [{c['id'][:8]}] {c['title']} ({c['entry_count']} messages): {c['summary'][:200]}")
                if not relevant:
                    return f"No relevant memories found for: {query}"
                return "Relevant memories:\n" + "\n".join(relevant[:5])

            # -- File system tools --
            if name == "execute_command":
                import subprocess
                command = args.get("command", "")
                timeout = args.get("timeout", 30)

                # Sandbox checks
                if not self._is_safe_command(command):
                    if self._is_dangerous_command(command):
                        logger.warning("Blocked dangerous command: %s", command)
                        return (
                            f"[SECURITY] Command blocked for safety: '{command[:100]}'\n"
                            f"This command matches a dangerous pattern. Only safe commands "
                            f"(read-only: ls, cat, grep, find, etc.) are allowed by default."
                        )
                    logger.warning("Command not in safe list: %s", command)
                    return (
                        f"[SECURITY] Command not allowed: '{command[:100]}'\n"
                        f"Only safe commands are permitted. Allowed: ls, cat, head, tail, "
                        f"echo, pwd, which, date, uname, find, grep, wc, sort, file, "
                        f"python --version, git status, git log, git diff, etc."
                    )

                logger.info("Executing command: %s", command)
                try:
                    proc = subprocess.run(
                        command, shell=True, capture_output=True, text=True, timeout=timeout,
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

            if name == "read_file":
                from pathlib import Path
                p = Path(args.get("path", ""))
                if not p.exists():
                    return f"File not found: {args['path']}"
                try:
                    return p.read_text(encoding="utf-8")[:8000]
                except Exception as e:
                    return f"Error reading file: {e}"

            if name == "write_file":
                from pathlib import Path
                p = Path(args.get("path", ""))
                content = args.get("content", "")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
                logger.info("File written: %s (%d bytes)", args["path"], len(content))
                return f"File written: {args['path']} ({len(content)} bytes)"

            if name == "append_file":
                from pathlib import Path
                p = Path(args.get("path", ""))
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open("a", encoding="utf-8") as f:
                    f.write(args.get("content", ""))
                logger.info("Content appended to: %s", args["path"])
                return f"Content appended to: {args['path']}"

            if name == "list_files":
                from pathlib import Path
                p = Path(args.get("path", "."))
                recursive = args.get("recursive", False)
                if not p.exists() or not p.is_dir():
                    return f"Directory not found: {args['path']}"
                if recursive:
                    files = [str(f.relative_to(p)) for f in p.rglob("*")]
                else:
                    files = [str(f.name) for f in p.iterdir()]
                return "\n".join(sorted(files)) if files else "(empty directory)"

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
            return f"Tool '{name}' error: {e}"

    def reset_context(self) -> None:
        self.context.clear()
        self.call_depth = 0

    def shutdown(self) -> None:
        self.mcp.close_all()
        self.memory.save_all()