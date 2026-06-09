"""
Nyx — a zero-dependency agentic coding CLI.

Auto-detects Rich for a beautiful TUI if installed, falls back to basic ANSI.

Usage:
    nyx                               # Interactive chat mode
    nyx -p "refactor this file"       # Single prompt mode
    nyx --help                        # Show help
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from nyx.config import Config
from nyx.providers import get_provider
from nyx.mcp_client import MCPManager
from nyx.skill_manager import SkillManager
from nyx.subagent import SubagentManager
from nyx.agent import Agent
from nyx.memory import MemoryManager

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


def _make_ansi_on_token() -> callable:
    buffer: list[str] = []

    def on_token(token: str) -> None:
        buffer.append(token)
        print(token, end="", flush=True)

    return on_token


# ---------------------------------------------------------------------------
# ANSI fallback: interactive REPL
# ---------------------------------------------------------------------------

WELCOME_BANNER = f"""
{c('╔══════════════════════════════════════════════════════╗', HEADER)}
{c('║', HEADER)}            {c('⚡ Nyx v0.2.0', BOLD)}              {c('║', HEADER)}
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


def print_tools(agent: Agent) -> None:
    print(f"\n{c('Available Tools:', BOLD)}")
    print(f"  {c('─' * 50, DIM)}")
    for t in agent.tools:
        desc = t.description[:80] + ("..." if len(t.description) > 80 else "")
        print(f"  {c(f'◈ {t.name}', CYAN)}")
        print(f"    {c(desc, DIM)}")
    print()


def run_ansi_interactive(agent: Agent, config: Config) -> None:
    """Fallback interactive REPL (no Rich)."""
    print_welcome(config, len(agent.tools))

    while True:
        try:
            user_input = input(f"\n{c('You', USER_COLOR)}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{c('Bye! 👋', GREEN)}")
            break

        if not user_input:
            continue

        # Built-in commands
        if user_input in {"/exit", "/quit", "/q"}:
            print(f"{c('Bye! 👋', GREEN)}")
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
        if user_input == "/tools":
            print_tools(agent)
            continue
        if user_input == "/memory":
            conv = agent.memory.current
            if conv:
                print(f"{c('Memory:', BOLD)}")
                print(f"  Title: {conv.title}")
                print(f"  Messages: {len(conv.entries)}")
                print(f"  Total tokens: {conv.total_tokens}")
                print(f"  Summary: {conv.summary[:200] if conv.summary else 'None'}")
            else:
                print(f"{c('No active conversation.', YELLOW)}")
            continue
        if user_input == "/conversations":
            convs = agent.memory.list_conversations()
            if not convs:
                print(f"{c('No saved conversations.', YELLOW)}")
                continue
            for c in convs[:10]:
                print(f"  [{c['id'][:8]}] {c['title'][:40]} ({c['entry_count']} msgs)")
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
    result = agent.run(prompt)
    if not agent.config.stream:
        print(f"{c('Agent', ASSISTANT_COLOR)}> {result}")
    else:
        print()
    agent.shutdown()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nyx — Zero-dependency agentic coding CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  nyx                               Interactive mode\n"
            "  nyx -p 'list all files'           Single prompt\n"
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

    args = parser.parse_args()

    if args.no_color:
        os.environ["NO_COLOR"] = "1"

    # Load config
    config = Config.load(args.config if args.config else None)

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
        config.project_dir = os.getcwd()

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
        if args.prompt:
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