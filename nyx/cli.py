"""
 Nyx — a standard-library-first agentic coding CLI.

Auto-detects Rich for a beautiful TUI if installed, falls back to basic ANSI.

Usage:
    nyx                               # Interactive chat mode
    nyx -p "refactor this file"       # Single prompt mode
    nyx --json                        # JSON output mode (CI/CD)
    nyx --help                         # Show help
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

_approval_lock = threading.Lock()

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nyx import __version__
from nyx.config import Config, ConfigError
from nyx.config import DEFAULT_CONFIG, DEFAULT_USER_CONFIG_PATH
from nyx.providers import get_provider
from nyx.mcp_client import MCPManager
from nyx.skill_manager import SkillManager
from nyx.subagent import SubagentManager
from nyx.agent import Agent
from nyx.memory import MemoryManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import Rich TUI
# ---------------------------------------------------------------------------

try:
    from nyx.cli_rich import run_rich_interactive, run_rich_single
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Fallback ANSI helpers (used when Rich is not available)
# ---------------------------------------------------------------------------

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
DIM = "\033[2m"

HEADER = f"{BOLD}{CYAN}"
USER_COLOR = f"{BOLD}{GREEN}"
ASSISTANT_COLOR = f"{BOLD}{MAGENTA}"
TOOL_COLOR = f"{DIM}{YELLOW}"
ERROR_COLOR = f"{BOLD}{RED}"


def supports_color() -> bool:
    return not os.environ.get("NO_COLOR") and sys.stdout.isatty()


def c(text: str, color: str) -> str:
    return f"{color}{text}{RESET}" if supports_color() else text


def render_markdown(text: str, force_color: bool = False) -> str:
    """Render basic markdown using ANSI escape codes without runtime UI dependencies."""
    has_color = force_color or supports_color()

    import re
    lines = text.splitlines()
    rendered_lines = []
    in_code_block = False
    
    # Inline helper to style text if color is enabled/forced
    def style(txt: str, ansi_code: str) -> str:
        return f"{ansi_code}{txt}{RESET}" if has_color else txt

    for line in lines:
        # Code block toggle
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            border_char = "─"
            rendered_lines.append(style(f"╭{border_char * 78}╮" if in_code_block else f"╰{border_char * 78}╯", DIM + YELLOW))
            continue

        if in_code_block:
            # Code block lines
            rendered_lines.append(style("│ ", DIM + YELLOW) + style(line, YELLOW) + " " * max(0, 76 - len(line)) + style(" │", DIM + YELLOW))
            continue

        # Headings
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            heading_text = m.group(2)
            if level == 1:
                rendered_lines.append("")
                rendered_lines.append(style(heading_text.upper(), BOLD + CYAN))
                rendered_lines.append(style("═" * len(heading_text), BOLD + CYAN))
            elif level == 2:
                rendered_lines.append("")
                rendered_lines.append(style(heading_text, BOLD + YELLOW))
                rendered_lines.append(style("─" * len(heading_text), DIM + YELLOW))
            else:
                rendered_lines.append(style(heading_text, BOLD + GREEN))
            continue

        # Horizontal rules
        if re.match(r"^[-*_]{3,}\s*$", line.strip()):
            rendered_lines.append(style("─" * 80, DIM))
            continue

        # Blockquotes
        if line.strip().startswith(">"):
            content = line.strip().lstrip(">").strip()
            rendered_lines.append(style("│ ", CYAN) + style(content, DIM))
            continue

        # Lists (unordered/ordered)
        m_list = re.match(r"^(\s*)[-*+•]\s+(.*)$", line)
        if m_list:
            indent = m_list.group(1)
            content = m_list.group(2)
            rendered_lines.append(f"{indent}{style('◈', GREEN)} {content}")
            continue

        # Style Bold (**text** or __text__)
        line = re.sub(r"\*\*([^*]+)\*\*", lambda m: style(m.group(1), BOLD), line)
        line = re.sub(r"__([^_]+)__", lambda m: style(m.group(1), BOLD), line)

        # Style Italic (*text* or _text_)
        line = re.sub(r"\*([^*]+)\*", lambda m: style(m.group(1), "\033[3m"), line)
        line = re.sub(r"_([^_]+)_", lambda m: style(m.group(1), "\033[3m"), line)

        # Style Inline Code (`code`)
        line = re.sub(r"`([^`]+)`", lambda m: style(m.group(1), YELLOW), line)

        # Links [text](url) -> text (url)
        line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: style(m.group(1), BOLD + CYAN) + f" ({m.group(2)})", line)

        rendered_lines.append(line)

    return "\n".join(rendered_lines)


# ---------------------------------------------------------------------------
# ANSI fallback: token streaming callback
# ---------------------------------------------------------------------------


def _make_ansi_on_token() -> Callable[[str], None]:
    buffer: list[str] = []

    def on_token(token: str) -> None:
        buffer.append(token)
        print(token, end="", flush=True)

    return on_token


# ---------------------------------------------------------------------------
# ANSI fallback: progress bar
# ---------------------------------------------------------------------------


class ProgressBar:
    """Simple ANSI progress bar for operations like MCP loading, subagents."""

    def __init__(self, total: int = 0, label: str = "", width: int = 30):
        self.total = total
        self.label = label
        self.width = width
        self._current = 0
        self._start_time = time.time()

    def update(self, n: int = 1) -> None:
        self._current += n
        self._draw()

    def set_total(self, total: int) -> None:
        self.total = total

    def _draw(self) -> None:
        if not supports_color() or not sys.stdout.isatty():
            return
        frac = self._current / max(self.total, 1)
        filled = int(self.width * frac)
        bar = "█" * filled + "░" * (self.width - filled)
        elapsed = time.time() - self._start_time
        pct = int(frac * 100)
        print(
            f"\r  {c(self.label, DIM)} [{c(bar, CYAN)}] "
            f"{c(f'{pct}%', BOLD)} "
            f"{c(f'({self._current}/{self.total})', DIM)} "
            f"{c(f'{elapsed:.1f}s', DIM)}",
            end="",
            flush=True,
        )
        if self._current >= self.total:
            print()

    def close(self) -> None:
        if self._current < self.total:
            self._current = self.total
            self._draw()


# ---------------------------------------------------------------------------
# ANSI fallback: interactive REPL
# ---------------------------------------------------------------------------

WELCOME_BANNER = f"""
{c('╔══════════════════════════════════════════════════════╗', HEADER)}
{c('║', HEADER)}            {c('⚡ Nyx v0.2.1', BOLD)}              {c('║', HEADER)}
{c('║', HEADER)}    {c('Zero-dependency agentic coding tool', DIM)}  {c('║', HEADER)}
{c('╚══════════════════════════════════════════════════════╝', HEADER)}
"""

HELP_TEXT = f"""
{c('Commands:', BOLD)}
  {c('/help', CYAN)}         Show this help
  {c('/model', CYAN)}        Show current model
  {c('/model <name>', CYAN)} Change model
  {c('/mode <name>', CYAN)}  Switch mode: chat | code | architect | debug
  {c('/autonomy <lvl>', CYAN)} Switch autonomy: ask | auto | yolo
  {c('/config', CYAN)}        Show configuration status
  {c('/config save [--global]', CYAN)} Save current session config
  {c('/config set [--global] <k> <v>', CYAN)} Set config option
  {c('/clear', CYAN)}        Clear conversation context
  {c('/tools', CYAN)}        List all available tools
  {c('/memory', CYAN)}       Show memory status
  {c('/conversations', CYAN)} List saved conversations
  {c('/switch <id>', CYAN)}  Switch to a saved conversation
  {c('/reset', CYAN)}        Reset agent (clear context + shutdown MCP)
  {c('/exit', CYAN)}         Exit the program
"""


def print_welcome(config: Config, tool_count: int) -> None:
    print(WELCOME_BANNER)
    print(f"  {c('Provider:', DIM)} {config.provider}")
    print(f"  {c('Model:', DIM)}    {config.model}")
    print(f"  {c('Tools:', DIM)}    {tool_count}")
    if config.project_dir:
        print(f"  {c('Project:', DIM)}  {config.project_dir}")
    print(f"  {c('Ready!', GREEN)} Type your request or {c('/help', CYAN)} for commands.\n")


def print_tools(agent: Agent, page: int = 1, page_size: int = 10) -> None:
    """Print tools with pagination."""
    tools = agent.tools
    total = len(tools)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    end = min(start + page_size, total)

    print(f"\n{c('Available Tools:', BOLD)}  {c(f'Page {page}/{total_pages}', DIM)}")
    print(f"  {c('─' * 50, DIM)}")
    for t in tools[start:end]:
        desc = t.description[:80] + ("..." if len(t.description) > 80 else "")
        print(f"  {c(f'◈ {t.name}', CYAN)}")
        print(f"    {c(desc, DIM)}")
    if total_pages > 1:
        print(f"  {c(f'Page {page}/{total_pages} — use /tools {page+1} for next page', DIM)}")
    print()


def print_memory_paginated(agent: Agent, page: int = 1, page_size: int = 10) -> None:
    """Show memory status with pagination for entries."""
    conv = agent.memory.current
    if not conv:
        print(f"{c('No active conversation.', YELLOW)}")
        return

    print(f"\n{c('🧠 Memory:', BOLD)} {conv.title}")
    print(f"  {c('Messages:', DIM)} {len(conv.entries)}  {c('Tokens:', DIM)} {conv.total_tokens}")
    print(f"  {c('Summary:', DIM)} {conv.summary[:200] if conv.summary else 'None'}")

    if conv.entries:
        total = len(conv.entries)
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = max(1, min(page, total_pages))
        start = (page - 1) * page_size
        end = min(start + page_size, total)

        print(f"\n  {c(f'Entries (Page {page}/{total_pages}):', BOLD)}")
        for i in range(start, end):
            entry = conv.entries[i]
            preview = entry.content[:80].replace("\n", " ")
            print(f"    [{i+1}] {c(entry.role+':', GREEN if entry.role=='user' else MAGENTA)} {c(preview, DIM)}")
        if total_pages > 1:
            print(f"  {c(f'Use /memory {page+1} for next page', DIM)}")
    print()


def print_conversations_paginated(agent: Agent, page: int = 1, page_size: int = 10) -> None:
    """List conversations with pagination."""
    convs = agent.memory.list_conversations()
    if not convs:
        print(f"{c('No saved conversations.', YELLOW)}")
        return

    total = len(convs)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    end = min(start + page_size, total)

    print(f"\n{c('📂 Conversations:', BOLD)}  {c(f'Page {page}/{total_pages}', DIM)}")
    print(f"  {c('─' * 60, DIM)}")
    for conv in convs[start:end]:
        current_mark = " ← current" if conv["id"] == agent.memory.current.id else ""
        print(f"  [{c(conv['id'][:8], CYAN)}] {c(conv['title'][:40], BOLD)} "
              f"({conv['entry_count']} msgs){c(current_mark, GREEN)}")
        if conv["summary"]:
            print(f"       {c(conv['summary'][:60], DIM)}")
    if total_pages > 1:
        print(f"  {c(f'Page {page}/{total_pages} — use /conversations {page+1} for next page', DIM)}")
    print()


def _make_ansi_approval_handler() -> Callable[[str], tuple[bool, str]]:
    """Create an interactive approval handler for the ANSI fallback CLI."""
    def handle_approval(command: str) -> tuple[bool, str]:
        with _approval_lock:
            print(f"\n{c('⚠️  SECURITY', YELLOW)} The AI wants to execute a potentially dangerous command:")
            print(f"  {c(command, CYAN)}")
            response = input(f"  {c('Allow?', BOLD)} (y/n): ").strip().lower()
            if response == "y":
                return True, ""
            else:
                reason = input(f"  {c('Reason for denial:', DIM)} ").strip()
                return False, reason or "User denied the command."
    return handle_approval


def _make_ansi_file_approval_handler() -> Callable[[str, str, str], tuple[bool, str]]:
    """Create an interactive approval handler for file operations (diff/patch)."""
    def handle_file_approval(path: str, summary: str, diff: str) -> tuple[bool, str]:
        with _approval_lock:
            print(f"\n{c('📝 FILE OPERATION', CYAN)} {summary}")
            print(f"  {c(diff[:2000], DIM)}")
            if len(diff) > 2000:
                print(f"  {c('(... diff truncated, full diff has ' + str(len(diff)) + ' chars)', DIM)}")
            response = input(f"  {c('Apply this change?', BOLD)} (y/n): ").strip().lower()
            if response == "y":
                return True, ""
            else:
                reason = input(f"  {c('Reason for denial:', DIM)} ").strip()
                return False, reason or "User denied the file change."
    return handle_file_approval


# ---------------------------------------------------------------------------
# REPL history
# ---------------------------------------------------------------------------


class REPLHistory:
    """Simple file-backed REPL history with dedup and max length."""

    def __init__(self, history_file: str | None = None, max_size: int = 1000):
        self._max_size = max_size
        self._entries: list[str] = []
        self._path = Path(history_file) if history_file else Path.home() / ".nyx_history"
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                lines = self._path.read_text(encoding="utf-8", errors="ignore").splitlines()
                self._entries = [l for l in lines if l.strip()][-self._max_size:]
            except OSError:
                self._entries = []

    def append(self, entry: str) -> None:
        entry = entry.strip()
        if not entry:
            return
        # Dedup: remove previous occurrence
        if entry in self._entries:
            self._entries.remove(entry)
        self._entries.append(entry)
        if len(self._entries) > self._max_size:
            self._entries = self._entries[-self._max_size:]
        self._save()

    def _save(self) -> None:
        try:
            self._path.write_text("\n".join(self._entries) + "\n", encoding="utf-8")
        except OSError:
            pass

    def search(self, prefix: str) -> list[str]:
        """Return matching history entries (most recent first)."""
        if not prefix:
            return []
        return [e for e in reversed(self._entries) if e.startswith(prefix)][:10]

    def get(self, index: int) -> str | None:
        if 0 <= index < len(self._entries):
            return self._entries[index]
        return None

    @property
    def entries(self) -> list[str]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# ANSI interactive REPL with history, autocompletion, pagination
# ---------------------------------------------------------------------------


def _get_paginated_arg(user_input: str, cmd: str) -> int:
    """Extract page number from '/cmd N' input."""
    rest = user_input[len(cmd):].strip()
    try:
        return int(rest.split()[0]) if rest else 1
    except (ValueError, IndexError):
        return 1


def _autocomplete_commands(partial: str) -> list[str]:
    """Return matching built-in commands for autocompletion."""
    commands = [
        "/help", "/model", "/mode", "/autonomy", "/config", "/clear", "/tools", "/memory",
        "/conversations", "/switch", "/reset", "/exit", "/quit", "/q",
    ]
    if not partial:
        return commands
    return [cmd for cmd in commands if cmd.startswith(partial)]


def _ansi_autocomplete(partial: str) -> str:
    """Simple inline autocompletion: show matches and return completed or partial."""
    if not partial.startswith("/"):
        return partial
    matches = _autocomplete_commands(partial)
    if len(matches) == 1:
        # Auto-complete with a space
        return matches[0] + " "
    elif len(matches) > 1:
        common = os.path.commonprefix(matches)
        if len(common) > len(partial):
            return common
        print(f"\n  {c('Suggestions:', DIM)} {'  '.join(c(m, CYAN) for m in matches)}")
    return partial


def setup_readline(agent: Agent) -> None:
    """Configure readline for command history and tab autocompletion."""
    try:
        try:
            import readline
        except ImportError:
            try:
                import pyreadline3 as readline  # type: ignore[no-redef]
            except ImportError:
                return
        import glob
        import atexit
        
        commands = [
            "/help", "/model", "/mode", "/autonomy", "/config", "/clear", "/tools", "/memory",
            "/conversations", "/switch", "/reset", "/exit", "/quit", "/q"
        ]
        
        def completer(text: str, state: int) -> str | None:
            # 1. Slash commands completion
            if text.startswith("/"):
                matches = [c for c in commands if c.startswith(text)]
                return matches[state] if state < len(matches) else None
                
            # 2. File path completion
            if not text:
                expanded = "."
            else:
                expanded = os.path.expanduser(text)
            
            # Match files starting with expanded text
            matches = glob.glob(expanded + "*")
            
            formatted = []
            for m in matches:
                # Add trailing slash for directories
                display = m
                if os.path.isdir(m):
                    display += "/"
                
                # Try to keep relative paths relative if input was relative
                if text and not text.startswith("/") and not text.startswith("~"):
                    try:
                        # Relativize path
                        rel = os.path.relpath(m)
                        if os.path.isdir(m):
                            rel += "/"
                        formatted.append(rel)
                    except ValueError:
                        formatted.append(display)
                else:
                    formatted.append(display)
            
            return formatted[state] if state < len(formatted) else None

        readline.set_completer(completer)
        # Tab key completion (depends on platform)
        try:
            if sys.platform == "darwin":
                readline.parse_and_bind("bind ^I rl_complete")
            elif os.name != "nt":
                readline.parse_and_bind("tab: complete")
        except Exception:
            pass
            
        # History management
        history_path = os.path.expanduser("~/.nyx_history")
        try:
            readline.read_history_file(history_path)
        except FileNotFoundError:
            pass
            
        readline.set_history_length(1000)
        atexit.register(readline.write_history_file, history_path)
        
    except Exception:
        pass


def run_ansi_interactive(agent: Agent, config: Config) -> None:
    """Fallback interactive REPL (no Rich)."""
    setup_readline(agent)
    agent.on_command_approval = _make_ansi_approval_handler()
    agent.on_file_approval = _make_ansi_file_approval_handler()
    history = REPLHistory()
    print_welcome(config, len(agent.tools))

    while True:
        try:
            raw_input = input(f"\n{c('You', USER_COLOR)}> ")
        except (EOFError, KeyboardInterrupt):
            print(f"\n{c('Bye! 👋', GREEN)}")
            agent.memory.save_all()
            break

        # Tab-like autocompletion via double-tab (user presses Enter on empty after partial)
        user_input = raw_input.strip()

        if not user_input:
            continue

        # History: up-arrow recall not possible in basic input(), but we store history
        history.append(user_input)

        # Built-in commands
        if user_input in {"/exit", "/quit", "/q"}:
            print(f"{c('Bye! 👋', GREEN)}")
            agent.memory.save_all()
            break
        if user_input in {"/help", "/?"}:
            print(HELP_TEXT)
            continue
        if user_input == "/clear":
            agent.reset_context()
            print(f"{c('Context cleared.', YELLOW)}")
            continue
        if user_input == "/reset":
            agent.shutdown()
            agent.reset_context()
            print(f"{c('Agent reset. MCP disconnected.', YELLOW)}")
            continue
        if user_input == "/model":
            print(f"{c('Current model:', BOLD)} {config.model}")
            continue
        if user_input.startswith("/model "):
            config.model = user_input[7:].strip()
            agent.provider = get_provider(config)
            print(f"{c('Model changed:', BOLD)} {config.model}")
            continue

        if user_input.startswith("/mode"):
            rest = user_input[5:].strip()
            if not rest:
                print(f"{c('Current mode:', BOLD)} {config.agent_mode}  |  autonomy: {config.agent_autonomy}")
            else:
                msg = agent.switch_mode(rest)
                print(f"{c(msg, GREEN if 'switched' in msg else YELLOW)}")
            continue

        if user_input.startswith("/autonomy"):
            rest = user_input[9:].strip()
            if not rest:
                print(f"{c('Current autonomy:', BOLD)} {config.agent_autonomy}")
            else:
                msg = agent.switch_autonomy(rest)
                print(f"{c(msg, GREEN if 'switched' in msg else YELLOW)}")
            continue

        if user_input.startswith("/config"):
            args_list = user_input[7:].strip().split()
            if not args_list:
                # Print current configuration status
                print(f"\n{c('Nyx Configuration Status', BOLD + CYAN)}")
                print(f"  {c('Active Provider:', DIM)} {config.provider}")
                print(f"  {c('Active Model:', DIM)}    {config.model}")
                print(f"  {c('Active Mode:', DIM)}     {config.agent_mode}")
                print(f"  {c('Active Autonomy:', DIM)} {config.agent_autonomy}")
                print(f"\n{c('Configuration Files (Priority order):', BOLD)}")
                paths = [
                    ("User (Global)", DEFAULT_USER_CONFIG_PATH),
                    ("Project (Local)", _project_config_path(config.project_dir)),
                ]
                for name, path in paths:
                    status = c("exists", GREEN) if path.exists() else c("not found", DIM)
                    print(f"  • {c(name, BOLD)}: {path} ({status})")
                print(f"\n{c('Use `/config save` to persist current session settings to project config.', DIM)}")
                print(f"{c('Use `/config save --global` to persist current session settings globally.', DIM)}")
                print(f"{c('Use `/config set <key> <value>` to change a config option.', DIM)}")
                continue

            subcmd = args_list[0].lower()
            if subcmd == "save":
                # Save current settings (model, provider, agent.mode, agent.autonomy)
                use_global = "--global" in args_list or "-g" in args_list
                path = DEFAULT_USER_CONFIG_PATH if use_global else _project_config_path(config.project_dir)
                try:
                    data = _load_config_file(path)
                    data["provider"] = config.provider
                    data["model"] = config.model
                    _set_nested(data, "agent.mode", config.agent_mode)
                    _set_nested(data, "agent.autonomy", config.agent_autonomy)
                    _write_config_file(path, data)
                    print(f"{c('Successfully saved session config to:', GREEN)} {path}")
                except Exception as e:
                    print(f"{c(f'Failed to save config: {e}', RED)}")
                continue

            elif subcmd == "set":
                # Set a specific key-value pair
                use_global = False
                key_val_args = []
                for a in args_list[1:]:
                    if a in ("--global", "-g"):
                        use_global = True
                    else:
                        key_val_args.append(a)
                
                if len(key_val_args) < 2:
                    print(f"{c('Error: `/config set [--global] <key> <value>` requires a key and a value.', RED)}")
                    continue
                
                key = key_val_args[0]
                val_str = " ".join(key_val_args[1:])
                path = DEFAULT_USER_CONFIG_PATH if use_global else _project_config_path(config.project_dir)
                try:
                    data = _load_config_file(path)
                    parsed_val = _parse_config_value(val_str)
                    _set_nested(data, key, parsed_val)
                    _write_config_file(path, data)
                    print(f"{c(f'Updated config key `{key}` to `{parsed_val}` in:', GREEN)} {path}")
                    # Apply immediately
                    if key == "model":
                        config.model = parsed_val
                        agent.provider = get_provider(config)
                    elif key == "provider":
                        config.provider = parsed_val
                        agent.provider = get_provider(config)
                    elif key == "agent.mode":
                        agent.switch_mode(parsed_val)
                    elif key == "agent.autonomy":
                        agent.switch_autonomy(parsed_val)
                except Exception as e:
                    print(f"{c(f'Failed to update config: {e}', RED)}")
                continue
            else:
                print(f"{c(f'Unknown config subcommand: {subcmd}. Valid options: save, set', RED)}")
                continue

        # Paginated commands
        if user_input.startswith("/tools"):
            page = _get_paginated_arg(user_input, "/tools")
            print_tools(agent, page=page)
            continue

        if user_input.startswith("/memory"):
            page = _get_paginated_arg(user_input, "/memory")
            print_memory_paginated(agent, page=page)
            continue

        if user_input.startswith("/conversations"):
            page = _get_paginated_arg(user_input, "/conversations")
            print_conversations_paginated(agent, page=page)
            continue

        # Switch conversation
        if user_input.startswith("/switch "):
            conv_id = user_input[8:].strip()
            if agent.memory.switch_to(conv_id):
                conv = agent.memory.current
                agent.load_conversation_history()
                print(f"{c('Switched to conversation:', GREEN)} {conv.title if conv else conv_id}")
            else:
                # Try partial match
                matches = [c for c in agent.memory.conversations.values() if c.id.startswith(conv_id)]
                if len(matches) == 1:
                    agent.memory.switch_to(matches[0].id)
                    agent.load_conversation_history()
                    print(f"{c('Switched to conversation:', GREEN)} {matches[0].title}")
                elif len(matches) > 1:
                    print(f"{c('Multiple matches — use full ID:', YELLOW)}")
                    for m in matches:
                        print(f"  [{c(m.id, CYAN)}] {m.title}")
                else:
                    print(f"{c(f'Conversation not found: {conv_id}', YELLOW)}")
            continue

        # Agent execution
        on_token = _make_ansi_on_token()
        if config.stream:
            print(f"{c('Agent', ASSISTANT_COLOR)}> ", end="", flush=True)

        try:
            result = agent.run(user_input, on_token=on_token)
            if not config.stream:
                print(f"{c('Agent', ASSISTANT_COLOR)}> {render_markdown(result)}")
            else:
                print()
                # Print rendered Markdown afterwards to replace raw token display with formatted text
                print(f"\n{c('Agent (formatted)', ASSISTANT_COLOR)}>\n{render_markdown(result)}")
        except Exception as e:
            print(f"\n{c(f'Error: {e}', ERROR_COLOR)}")


def run_ansi_single(agent: Agent, prompt: str) -> None:
    """Fallback single-prompt mode (no Rich)."""
    # Wire approval callbacks (same as interactive mode)
    agent.on_command_approval = _make_ansi_approval_handler()
    agent.on_file_approval = _make_ansi_file_approval_handler()
    on_token = _make_ansi_on_token()
    if agent.config.stream:
        print(f"{c('Agent', ASSISTANT_COLOR)}> ", end="", flush=True)
    result = agent.run(prompt, on_token=on_token)
    if not agent.config.stream:
        print(f"{c('Agent', ASSISTANT_COLOR)}> {render_markdown(result)}")
    else:
        print()
        print(f"\n{c('Agent (formatted)', ASSISTANT_COLOR)}>\n{render_markdown(result)}")
    agent.shutdown()


# ---------------------------------------------------------------------------
# JSON output mode (CI/CD)
# ---------------------------------------------------------------------------


def _run_json(agent: Agent, prompt: str) -> None:
    """Run in JSON mode — output structured JSON for CI/CD pipelines."""
    start = time.time()

    try:
        result = agent.run(prompt, on_token=None)
        output = {
            "status": "success",
            "prompt": prompt,
            "result": result,
            "duration_seconds": round(time.time() - start, 2),
            "session_id": agent.json_logger.session_id if agent.json_logger else "",
            "cost": round(agent.json_logger.total_cost, 6) if agent.json_logger else 0.0,
            "llm_calls": agent.json_logger.total_llm_calls if agent.json_logger else 0,
            "tool_calls": agent.json_logger.total_tool_calls if agent.json_logger else 0,
        }
    except Exception as e:
        output = {
            "status": "error",
            "prompt": prompt,
            "error": str(e),
            "duration_seconds": round(time.time() - start, 2),
        }

    print(json.dumps(output, ensure_ascii=False, indent=2))
    agent.shutdown()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _project_config_path(project_dir: str | None = None) -> Path:
    root = Path(project_dir or os.getcwd())
    return root / ".nyx" / "config.json"


def _parse_config_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _set_nested(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = data
    parts = [p for p in dotted_key.split(".") if p]
    if not parts:
        raise ValueError("Config key cannot be empty.")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _get_nested(data: dict[str, Any], dotted_key: str) -> Any:
    current: Any = data
    for part in [p for p in dotted_key.split(".") if p]:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_key)
        current = current[part]
    return current


def _load_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_config_file(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _handle_config_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Manage Nyx configuration")
    parser.add_argument("command", choices=["init", "config"])
    parser.add_argument("action", nargs="?", choices=["list", "get", "set"], help="Config action")
    parser.add_argument("key", nargs="?", help="Dot-separated config key")
    parser.add_argument("value", nargs="?", help="Value for 'set' (JSON or string)")
    parser.add_argument("--global", dest="use_global", action="store_true", help="Use the global user config")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing config during init")
    args = parser.parse_args(argv)

    path = DEFAULT_USER_CONFIG_PATH if args.use_global else _project_config_path()

    if args.command == "init":
        if path.exists() and not args.force:
            print(f"Config already exists: {path}")
            print("Use --force to overwrite it.")
            return 1
        initial = {
            "provider": DEFAULT_CONFIG["provider"],
            "model": DEFAULT_CONFIG["model"],
            "stream": DEFAULT_CONFIG["stream"],
            "max_tokens": DEFAULT_CONFIG["max_tokens"],
            "temperature": DEFAULT_CONFIG["temperature"],
            "skills_dir": DEFAULT_CONFIG["skills_dir"],
            "web_search_enabled": DEFAULT_CONFIG["web_search_enabled"],
            "agent": DEFAULT_CONFIG["agent"],
            "sandbox": DEFAULT_CONFIG["sandbox"],
            "permissions": DEFAULT_CONFIG["permissions"],
            "audit": DEFAULT_CONFIG["audit"],
            "diff_tool": DEFAULT_CONFIG["diff_tool"],
            "mcp_servers": {},
        }
        if sys.stdin.isatty():
            try:
                from nyx.mcp_discovery import run_interactive_discovery
                run_interactive_discovery(Path.cwd(), initial)
            except Exception as e:
                print(f"MCP discovery skipped: {e}")
        _write_config_file(path, initial)
        print(f"Created config: {path}")
        print("Set your API key via OPENROUTER_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY.")
        return 0

    if args.command == "config":
        action = args.action or "list"
        data = _load_config_file(path)
        if action == "list":
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        if action == "get":
            if not args.key:
                print("Error: config get requires a key", file=sys.stderr)
                return 1
            try:
                value = _get_nested(data, args.key)
            except KeyError:
                print(f"Config key not found: {args.key}", file=sys.stderr)
                return 1
            print(json.dumps(value, ensure_ascii=False, indent=2) if isinstance(value, (dict, list)) else value)
            return 0
        if action == "set":
            if not args.key or args.value is None:
                print("Error: config set requires a key and value", file=sys.stderr)
                return 1
            _set_nested(data, args.key, _parse_config_value(args.value))
            _write_config_file(path, data)
            print(f"Updated {args.key} in {path}")
            return 0

    return 1


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _make_cli_event_handler(use_rich: bool = False) -> Callable[[dict[str, Any]], None]:
    """Render agent events without coupling Agent to terminal output."""
    if use_rich and RICH_AVAILABLE:
        from nyx.cli_rich import get_console

        console = get_console()

        def rich_handler(event: dict[str, Any]) -> None:
            event_type = event.get("type", "")
            if event_type == "status":
                console.print(f"[dim]{event.get('message', '')}[/dim]")
            elif event_type == "mode":
                console.print(
                    f"[dim]Mode:[/dim] [cyan]{event.get('mode')}[/cyan]  "
                    f"[dim]Autonomy:[/dim] [yellow]{event.get('autonomy')}[/yellow]"
                )
            elif event_type == "setup_complete":
                console.print(f"[dim]Tools loaded:[/dim] [green]{event.get('tool_count', 0)}[/green]")
            elif event_type == "tool_start":
                target = event.get("target", "")
                suffix = f" [dim]->[/dim] [cyan]{target}[/cyan]" if target else ""
                console.print(f"[dim]Tool[/dim] [yellow]{event.get('name')}[/yellow]{suffix}")
            elif event_type == "tool_finish":
                style = "green" if event.get("ok") else "red"
                status = "ok" if event.get("ok") else "failed"
                details = ", ".join(event.get("details", []))
                console.print(f"[{style}]Tool {status}[/] [yellow]{event.get('name')}[/yellow] [dim]({details})[/dim]")

        return rich_handler

    def ansi_handler(event: dict[str, Any]) -> None:
        event_type = event.get("type", "")
        if event_type == "status":
            print(c(str(event.get("message", "")), DIM))
        elif event_type == "mode":
            print(f"  {c('Mode:', DIM)} {event.get('mode')}  |  {c('Autonomy:', DIM)} {event.get('autonomy')}")
        elif event_type == "setup_complete":
            print(f"  {c('Tools loaded:', DIM)} {event.get('tool_count', 0)}")
        elif event_type == "tool_start":
            target = event.get("target", "")
            suffix = f" -> {c(str(target), CYAN)}" if target else ""
            print(f"{c('Tool', DIM)} {c(str(event.get('name')), YELLOW)}{suffix}", flush=True)
        elif event_type == "tool_finish":
            status = "ok" if event.get("ok") else "failed"
            status_color = GREEN if event.get("ok") else RED
            details = ", ".join(event.get("details", []))
            print(f"{c('Tool ' + status, status_color)} {event.get('name')} ({details})", flush=True)

    return ansi_handler


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                if stream.isatty():
                    stream.reconfigure(encoding="utf-8", errors="replace")
                else:
                    stream.reconfigure(errors="replace")
            except Exception:
                try:
                    stream.reconfigure(errors="replace")
                except Exception:
                    pass

    if len(sys.argv) > 1 and sys.argv[1] in {"init", "config"}:
        sys.exit(_handle_config_command(sys.argv[1:]))

    parser = argparse.ArgumentParser(
        description="Nyx — standard-library-first agentic coding CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  nyx                               Interactive mode\n"
            "  nyx -p 'list all files'           Single prompt\n"
            "  nyx --json -p 'list all files'    JSON output (CI/CD)\n"
            "  nyx --config ./myconf.json        Custom config\n"
            "  nyx --dir /path/to/project        Set working directory\n"
        ),
    )
    parser.add_argument("-p", "--prompt", type=str, default="", help="Run a single prompt and exit")
    parser.add_argument("-c", "--config", type=str, default="", help="Path to config.json")
    parser.add_argument("-m", "--model", type=str, default="", help="Override model (e.g. 'openai/gpt-4o')")
    parser.add_argument("--provider", type=str, default="", help="Override provider (openrouter, openai, anthropic)")
    parser.add_argument("-d", "--dir", type=str, default="", help="Working directory for the AI")
    parser.add_argument("--project", type=str, default="", help="Project directory (alias for --dir)")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color output")
    parser.add_argument("--no-rich", action="store_true", help="Force basic CLI even if Rich is installed")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose/debug logging")
    parser.add_argument("--version", action="version", version=f"nyx {__version__}")
    parser.add_argument("--json", action="store_true", help="JSON output mode for CI/CD pipelines (requires --prompt)")
    # -- Mode & autonomy --
    parser.add_argument("--mode", type=str, default="", choices=["chat", "code", "architect", "debug"],
                        help="Agent mode: chat (default) | code | architect (read-only) | debug")
    parser.add_argument("--autonomy", type=str, default="", choices=["ask", "auto", "yolo"],
                        help="Autonomy level: ask (default) | auto (skip file prompts) | yolo (skip all prompts)")
    parser.add_argument("--auto", action="store_true", help="Shortcut for --autonomy auto")
    parser.add_argument("--yolo", action="store_true", help="Shortcut for --autonomy yolo (no approval prompts)")
    parser.add_argument("--max-depth", type=int, default=0, help="Override max reasoning depth (default: 50)")

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.no_color:
        os.environ["NO_COLOR"] = "1"

    # Pipe mode: read stdin when not a TTY AND in single-prompt mode
    piped_data = ""
    if args.prompt and not sys.stdin.isatty():
        try:
            piped_data = sys.stdin.read()
        except OSError:
            pass

    if piped_data and args.prompt:
        # Prepend piped content to the given prompt
        args.prompt = f"Context from stdin:\n```\n{piped_data}\n```\n\nTask: {args.prompt}"
        if not args.json:
            print(f"{c('📥 Pipe mode: ' + str(len(piped_data)) + ' chars from stdin', DIM)}", file=sys.stderr)

    # Validate --json requires --prompt
    if args.json and not args.prompt:
        print("Error: --json mode requires --prompt/-p", file=sys.stderr)
        sys.exit(1)

    # Load config
    try:
        config = Config.load(args.config if args.config else None)
        # Run auto-discovery interactively at startup if running in interactive mode
        if not args.prompt and sys.stdin.isatty():
            try:
                from nyx.mcp_discovery import run_interactive_discovery
                config_file_path = args.config if args.config else (_project_config_path() if _project_config_path().exists() else None)
                raw_config = config.raw or {}
                updated_raw = run_interactive_discovery(config.project_dir or os.getcwd(), raw_config)
                if "mcp_servers" in updated_raw and updated_raw["mcp_servers"]:
                    config.mcp_servers = updated_raw["mcp_servers"]
                    if config_file_path:
                        _write_config_file(config_file_path, updated_raw)
            except Exception as e:
                logger.debug("Failed to run interactive MCP auto-discovery: %s", e)
    except ConfigError as e:
        print(f"{c(f'Configuration error: {e}', ERROR_COLOR)}", file=sys.stderr)
        sys.exit(1)

    # CLI overrides
    if args.model:
        config.model = args.model
    if args.provider:
        config.provider = args.provider
    if args.no_stream:
        config.stream = False
    if args.dir or args.project:
        config.project_dir = args.dir or args.project
    else:
        # Default to current working directory
        try:
            config.project_dir = os.getcwd()
        except FileNotFoundError:
            # CWD was deleted (e.g. by a previous test); fall back to script dir
            config.project_dir = os.path.dirname(os.path.abspath(__file__))

    # Mode & autonomy overrides
    if args.mode:
        config.agent_mode = args.mode
    if args.yolo:
        config.agent_autonomy = "yolo"
    elif args.auto:
        config.agent_autonomy = "auto"
    elif args.autonomy:
        config.agent_autonomy = args.autonomy
    if args.max_depth > 0:
        config.agent_max_depth = args.max_depth

    logger.info("Starting Nyx with provider=%s model=%s", config.provider, config.model)

    # Build agent with all subsystems
    provider = get_provider(config)
    mcp = MCPManager()
    skills = SkillManager(config.skills_dir)
    subagents = SubagentManager(config)
    memory = MemoryManager(provider=provider)

    agent = Agent(
        config=config,
        provider=provider,
        mcp_manager=mcp,
        skill_manager=skills,
        subagent_manager=subagents,
        memory_manager=memory,
    )

    use_rich = RICH_AVAILABLE and not args.no_rich and not args.json
    if not args.json:
        agent.on_event = _make_cli_event_handler(use_rich=use_rich)

    # Setup connections
    agent.setup()

    # Set system prompt
    if config.system_prompt:
        agent.context.add("system", config.system_prompt)

    # Load conversation history from active session
    agent.load_conversation_history()

    try:
        if args.json:
            _run_json(agent, args.prompt)
        elif args.prompt:
            if use_rich:
                from nyx.cli_rich import run_rich_single
                run_rich_single(agent, args.prompt)
            else:
                run_ansi_single(agent, args.prompt)
        else:
            if use_rich:
                from nyx.cli_rich import run_rich_interactive
                run_rich_interactive(agent, config)
            else:
                run_ansi_interactive(agent, config)
    finally:
        agent.shutdown()


if __name__ == "__main__":
    main()
