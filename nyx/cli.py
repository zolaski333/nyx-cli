"""
Nyx — a zero-dependency agentic coding CLI.

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
import time
from pathlib import Path
from typing import Any, Callable

from nyx.config import Config, ConfigError
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
        "/help", "/model", "/clear", "/tools", "/memory",
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
        import readline
        import glob
        import atexit
        
        commands = [
            "/help", "/model", "/clear", "/tools", "/memory",
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
        if sys.platform == "darwin":
            readline.parse_and_bind("bind ^I rl_complete")
        else:
            readline.parse_and_bind("tab: complete")
            
        # History management
        history_path = os.path.expanduser("~/.nyx_history")
        try:
            readline.read_history_file(history_path)
        except FileNotFoundError:
            pass
            
        readline.set_history_length(1000)
        atexit.register(readline.write_history_file, history_path)
        
    except ImportError:
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
                print(f"{c('Switched to conversation:', GREEN)} {conv.title if conv else conv_id}")
            else:
                # Try partial match
                matches = [c for c in agent.memory.conversations.values() if c.id.startswith(conv_id)]
                if len(matches) == 1:
                    agent.memory.switch_to(matches[0].id)
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
                print(f"{c('Agent', ASSISTANT_COLOR)}> {result}")
            else:
                print()
        except Exception as e:
            print(f"\n{c(f'Error: {e}', ERROR_COLOR)}")


def run_ansi_single(agent: Agent, prompt: str) -> None:
    """Fallback single-prompt mode (no Rich)."""
    on_token = _make_ansi_on_token()
    if agent.config.stream:
        print(f"{c('Agent', ASSISTANT_COLOR)}> ", end="", flush=True)
    result = agent.run(prompt, on_token=on_token)
    if not agent.config.stream:
        print(f"{c('Agent', ASSISTANT_COLOR)}> {result}")
    else:
        print()
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


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nyx — Zero-dependency agentic coding CLI",
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
    parser.add_argument("--json", action="store_true", help="JSON output mode for CI/CD pipelines (requires --prompt)")

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

    # Setup connections
    agent.setup()

    # Set system prompt
    if config.system_prompt:
        agent.context.add("system", config.system_prompt)

    try:
        if args.json:
            _run_json(agent, args.prompt)
        elif args.prompt:
            if RICH_AVAILABLE and not args.no_rich:
                from nyx.cli_rich import run_rich_single
                run_rich_single(agent, args.prompt)
            else:
                run_ansi_single(agent, args.prompt)
        else:
            if RICH_AVAILABLE and not args.no_rich:
                from nyx.cli_rich import run_rich_interactive
                run_rich_interactive(agent, config)
            else:
                run_ansi_interactive(agent, config)
    finally:
        agent.shutdown()


if __name__ == "__main__":
    main()