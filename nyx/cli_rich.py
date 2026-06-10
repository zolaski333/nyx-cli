"""
Nyx — Rich TUI.

A beautiful terminal UI using the Rich library.
Falls back gracefully to the basic CLI if Rich is not installed.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

# Try to import Rich; fall back to basic CLI if unavailable
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.syntax import Syntax
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        TaskProgressColumn, TimeElapsedColumn,
    )
    from rich import box
    from rich.prompt import Prompt
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from nyx.config import Config
from nyx.agent import Agent


# ---------------------------------------------------------------------------
# Console wrapper
# ---------------------------------------------------------------------------

_console: Any = None


def get_console() -> Any:
    global _console
    if _console is None and RICH_AVAILABLE:
        _console = Console()
    return _console


# ---------------------------------------------------------------------------
# Styled components
# ---------------------------------------------------------------------------


def welcome_panel(config: Config, tool_count: int) -> Any:
    """Create a welcome panel for the Rich TUI."""
    if not RICH_AVAILABLE:
        return ""

    console = get_console()
    grid = Table.grid(padding=(0, 2))
    grid.add_column()
    grid.add_column()

    grid.add_row("[bold cyan]⚡ Nyx[/bold cyan]", "")
    grid.add_row("", "")
    grid.add_row("[dim]Provider:[/dim]", f"[green]{config.provider}[/green]")
    grid.add_row("[dim]Model:[/dim]", f"[yellow]{config.model}[/yellow]")
    grid.add_row("[dim]Tools:[/dim]", f"[blue]{tool_count}[/blue]")
    if config.project_dir:
        grid.add_row("[dim]Project:[/dim]", f"[white]{config.project_dir}[/white]")
    grid.add_row("", "")
    grid.add_row("[dim]Type /help for commands[/dim]", "")

    return Panel(grid, box=box.HEAVY, border_style="cyan", title="[bold]🚀 Ready[/bold]")


def help_panel() -> Any:
    """Create a help panel."""
    if not RICH_AVAILABLE:
        return ""

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    table.add_column("Command", style="yellow")
    table.add_column("Description", style="white")

    table.add_row("/help", "Show this help")
    table.add_row("/model", "Show current model")
    table.add_row("/model <name>", "Change model")
    table.add_row("/mode <name>", "Switch mode: chat | code | architect | debug")
    table.add_row("/autonomy <lvl>", "Switch autonomy: ask | auto | yolo")
    table.add_row("/clear", "Clear conversation context")
    table.add_row("/tools [N]", "List tools (paginated, optional page N)")
    table.add_row("/memory [N]", "Show memory status (paginated entries)")
    table.add_row("/conversations [N]", "List saved conversations (paginated)")
    table.add_row("/switch <id>", "Switch to a saved conversation")
    table.add_row("/reset", "Reset agent (clear context + shutdown MCP)")
    table.add_row("/exit", "Exit the program")

    return Panel(table, box=box.ROUNDED, border_style="cyan", title="[bold]Commands[/bold]")


def tools_table(tools: list, page: int = 1, page_size: int = 10) -> Any:
    """Create a tools table with pagination."""
    if not RICH_AVAILABLE:
        return ""

    total = len(tools)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    end = min(start + page_size, total)

    table = Table(
        box=box.SIMPLE, show_header=True, header_style="bold cyan",
        title=f"[bold]🔧 Tools ({total}) — Page {page}/{total_pages}[/bold]",
    )
    table.add_column("Tool", style="green")
    table.add_column("Description", style="white", no_wrap=False)

    for t in tools[start:end]:
        desc = t.description[:100] + ("..." if len(t.description) > 100 else "")
        table.add_row(t.name, desc)

    return Panel(table, box=box.ROUNDED, border_style="green")


def memory_panel(agent: Agent, page: int = 1, page_size: int = 10) -> Any:
    """Show memory status with paginated entries."""
    if not RICH_AVAILABLE:
        return ""

    conv = agent.memory.current
    if not conv:
        return Panel("[yellow]No active conversation.[/yellow]", title="[bold]🧠 Memory[/bold]", border_style="cyan")

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    table.add_column("Property", style="yellow")
    table.add_column("Value", style="white")
    table.add_row("Title", conv.title)
    table.add_row("Messages", str(len(conv.entries)))
    table.add_row("Total tokens", str(conv.total_tokens))
    table.add_row("Has summary", "✓" if conv.summary else "✗")
    table.add_row("Summary", conv.summary[:200] if conv.summary else "None")

    # Paginated entries
    if conv.entries:
        total = len(conv.entries)
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = max(1, min(page, total_pages))
        start = max(0, total - page * page_size)
        end = min(start + page_size, total)

        entries_table = Table(
            box=box.SIMPLE, show_header=True, header_style="bold cyan",
            title=f"Entries (Page {page}/{total_pages})",
        )
        entries_table.add_column("#", style="dim", width=4)
        entries_table.add_column("Role", style="green", width=10)
        entries_table.add_column("Content", style="white", no_wrap=False)

        for i in range(start, end):
            entry = conv.entries[i]
            preview = entry.content[:100].replace("\n", " ")
            entries_table.add_row(str(i + 1), entry.role, preview)

        return Panel(
            Table.grid(padding=(0, 1))
            .add_row(table)
            .add_row(entries_table),
            title="[bold]🧠 Memory[/bold]",
            border_style="cyan",
        )

    return Panel(table, title="[bold]🧠 Memory[/bold]", border_style="cyan")


def conversations_panel(agent: Agent, page: int = 1, page_size: int = 10) -> Any:
    """List conversations with pagination."""
    if not RICH_AVAILABLE:
        return ""

    convs = agent.memory.list_conversations()
    if not convs:
        return Panel("[yellow]No saved conversations.[/yellow]", title="[bold]📂 Conversations[/bold]", border_style="cyan")

    total = len(convs)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    end = min(start + page_size, total)

    table = Table(
        box=box.SIMPLE, show_header=True, header_style="bold cyan",
        title=f"[bold]📂 Conversations ({total}) — Page {page}/{total_pages}[/bold]",
    )
    table.add_column("ID", style="dim")
    table.add_column("Title", style="white")
    table.add_column("Messages", style="blue")
    table.add_column("Summary", style="dim")

    current_id = agent.memory.current.id if agent.memory.current else ""

    for c in convs[start:end]:
        marker = " ← current" if c["id"] == current_id else ""
        table.add_row(
            c["id"][:8],
            c["title"][:40] + marker,
            str(c["entry_count"]),
            c["summary"][:60] if c["summary"] else "",
        )

    return Panel(table, box=box.ROUNDED, border_style="cyan")


def format_content(content: str) -> Any:
    """Format agent response as Rich renderable."""
    if not RICH_AVAILABLE:
        return content

    # Try to detect code blocks and render appropriately
    if content.strip().startswith("```"):
        return Syntax(content, "python", theme="monokai", line_numbers=True)
    return Markdown(content)


# ---------------------------------------------------------------------------
# Rich progress bar helpers
# ---------------------------------------------------------------------------


def make_mcp_progress() -> Progress:
    """Create a progress bar for MCP server loading."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=get_console(),
        transient=True,
    )


def make_subagent_progress() -> Progress:
    """Create a progress bar for subagent execution."""
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]{task.completed}/{task.total}[/dim]"),
        TimeElapsedColumn(),
        console=get_console(),
        transient=True,
    )


# ---------------------------------------------------------------------------
# Token streaming with Rich
# ---------------------------------------------------------------------------


def _make_rich_on_token() -> Callable[[str], None]:
    """Create a streaming callback that uses Rich for display."""
    buffer: list[str] = []
    console = get_console()

    def on_token(token: str) -> None:
        buffer.append(token)
        console.print(token, end="", style="magenta")

    return on_token


# ---------------------------------------------------------------------------
# REPL history (Rich version)
# ---------------------------------------------------------------------------


class RichREPLHistory:
    """File-backed REPL history with Rich display."""

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
        if not prefix:
            return []
        return [e for e in reversed(self._entries) if e.startswith(prefix)][:10]

    @property
    def entries(self) -> list[str]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Rich interactive REPL
# ---------------------------------------------------------------------------


def _make_rich_approval_handler(console) -> Callable[[str], tuple[bool, str]]:
    """Create an interactive approval handler using Rich prompts."""
    def handle_approval(command: str) -> tuple[bool, str]:
        console.print("\n[bold yellow]⚠️  SECURITY[/bold yellow] The AI wants to execute a potentially dangerous command:")
        console.print(f"  [cyan]{command}[/cyan]")
        response = console.input(f"  [bold]Allow?[/bold] (y/n): ").strip().lower()
        if response == "y":
            return True, ""
        else:
            reason = console.input(f"  [dim]Reason for denial:[/dim] ").strip()
            return False, reason or "User denied the command."
    return handle_approval


def _make_rich_file_approval_handler(console) -> Callable[[str, str, str], tuple[bool, str]]:
    """Create an interactive approval handler for file operations using Rich."""
    def handle_file_approval(path: str, summary: str, diff: str) -> tuple[bool, str]:
        from rich.syntax import Syntax
        from rich.panel import Panel
        from rich.text import Text

        console.print(f"\n[bold cyan]📝 FILE OPERATION[/bold cyan]")
        console.print(f"  [bold]{summary}[/bold]")

        if diff:
            # Show diff with syntax highlighting
            try:
                syntax = Syntax(diff[:3000], "diff", theme="monokai", line_numbers=True)
                console.print(Panel(syntax, title="[bold]Changes[/bold]", border_style="cyan"))
            except Exception:
                console.print(f"  [dim]{diff[:3000]}[/dim]")

            if len(diff) > 3000:
                console.print(f"  [dim](... diff truncated, {len(diff)} total chars)[/dim]")

        response = console.input(f"  [bold]Apply this change?[/bold] (y/n): ").strip().lower()
        if response == "y":
            return True, ""
        else:
            reason = console.input(f"  [dim]Reason for denial:[/dim] ").strip()
            return False, reason or "User denied the file change."
    return handle_file_approval


def _get_paginated_arg(user_input: str, cmd: str) -> int:
    """Extract page number from '/cmd N' input."""
    rest = user_input[len(cmd):].strip()
    try:
        return int(rest.split()[0]) if rest else 1
    except (ValueError, IndexError):
        return 1


def _rich_autocomplete(console, partial: str) -> str:
    """Rich autocompletion: show matching commands and return completed prefix."""
    commands = [
        "/help", "/model", "/clear", "/tools", "/memory",
        "/conversations", "/switch", "/reset", "/exit", "/quit", "/q",
    ]
    if not partial.startswith("/"):
        return partial

    matches = [cmd for cmd in commands if cmd.startswith(partial)]
    if len(matches) == 1:
        return matches[0] + " "
    elif len(matches) > 1:
        common = os.path.commonprefix(matches)
        if len(common) > len(partial):
            return common
        console.print(f"\n[dim]Suggestions:[/dim] [cyan]{'  [/cyan][cyan]'.join(matches)}[/cyan]")
    return partial


def run_rich_interactive(agent: Agent, config: Config) -> None:
    """Run the interactive REPL with Rich formatting."""
    from nyx.cli import setup_readline
    setup_readline(agent)
    console = get_console()
    agent.on_command_approval = _make_rich_approval_handler(console)
    agent.on_file_approval = _make_rich_file_approval_handler(console)
    history = RichREPLHistory()
    console.clear()
    console.print(welcome_panel(config, len(agent.tools)))

    while True:
        try:
            user_input = console.input(f"\n[bold green]You[/bold green]> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold green]Bye! 👋[/bold green]")
            agent.memory.save_all()
            break

        if not user_input:
            continue

        # Store in history
        history.append(user_input)

        # -- Built-in commands --
        if user_input in {"/exit", "/quit", "/q"}:
            console.print("[bold green]Bye! 👋[/bold green]")
            break

        if user_input in {"/help", "/?"}:
            console.print(help_panel())
            continue

        if user_input == "/clear":
            agent.reset_context()
            console.print("[yellow]Context cleared.[/yellow]")
            continue

        if user_input == "/reset":
            agent.shutdown()
            agent.reset_context()
            console.print("[yellow]Agent reset. MCP disconnected.[/yellow]")
            continue

        if user_input == "/model":
            console.print(f"[bold]Current model:[/bold] [yellow]{config.model}[/yellow]")
            continue

        if user_input.startswith("/model "):
            from nyx.providers import get_provider
            config.model = user_input[7:].strip()
            agent.provider = get_provider(config)
            console.print(f"[bold]Model changed:[/bold] [yellow]{config.model}[/yellow]")
            continue

        if user_input.startswith("/mode"):
            rest = user_input[5:].strip()
            if not rest:
                console.print(f"[bold]Current mode:[/bold] [yellow]{config.agent_mode}[/yellow]  |  autonomy: [cyan]{config.agent_autonomy}[/cyan]")
            else:
                msg = agent.switch_mode(rest)
                style = "green" if "switched" in msg else "yellow"
                console.print(f"[{style}]{msg}[/{style}]")
            continue

        if user_input.startswith("/autonomy"):
            rest = user_input[9:].strip()
            if not rest:
                console.print(f"[bold]Current autonomy:[/bold] [cyan]{config.agent_autonomy}[/cyan]")
            else:
                msg = agent.switch_autonomy(rest)
                style = "green" if "switched" in msg else "yellow"
                console.print(f"[{style}]{msg}[/{style}]")
            continue

        # Paginated commands
        if user_input.startswith("/tools"):
            page = _get_paginated_arg(user_input, "/tools")
            console.print(tools_table(agent.tools, page=page))
            continue

        if user_input.startswith("/memory"):
            page = _get_paginated_arg(user_input, "/memory")
            console.print(memory_panel(agent, page=page))
            continue

        if user_input.startswith("/conversations"):
            page = _get_paginated_arg(user_input, "/conversations")
            console.print(conversations_panel(agent, page=page))
            continue

        # Switch conversation
        if user_input.startswith("/switch "):
            conv_id = user_input[8:].strip()
            if agent.memory.switch_to(conv_id):
                conv = agent.memory.current
                console.print(f"[green]Switched to conversation:[/green] {conv.title if conv else conv_id}")
            else:
                # Try partial match
                matches = [c for c in agent.memory.conversations.values() if c.id.startswith(conv_id)]
                if len(matches) == 1:
                    agent.memory.switch_to(matches[0].id)
                    console.print(f"[green]Switched to conversation:[/green] {matches[0].title}")
                elif len(matches) > 1:
                    console.print("[yellow]Multiple matches — use full ID:[/yellow]")
                    for m in matches:
                        console.print(f"  [cyan]{m.id}[/cyan] {m.title}")
                else:
                    console.print(f"[yellow]Conversation not found: {conv_id}[/yellow]")
            continue

        # -- Agent execution --
        on_token = _make_rich_on_token()

        if config.stream:
            console.print("[bold magenta]Agent[/bold magenta]> ", end="")

        try:
            result = agent.run(user_input, on_token=on_token)
            if not config.stream:
                console.print(format_content(result))
            else:
                console.print()
        except Exception as e:
            console.print(f"\n[bold red]Error: {e}[/bold red]")

    # Save memory on exit
    agent.memory.save_all()


# ---------------------------------------------------------------------------
# Rich single-prompt mode
# ---------------------------------------------------------------------------


def run_rich_single(agent: Agent, prompt: str) -> None:
    """Run a single prompt with Rich formatting."""
    console = get_console()
    # Wire approval callbacks (same as interactive mode)
    agent.on_command_approval = _make_rich_approval_handler(console)
    agent.on_file_approval = _make_rich_file_approval_handler(console)
    on_token = _make_rich_on_token()

    if agent.config.stream:
        console.print("[bold magenta]Agent[/bold magenta]> ", end="")

    result = agent.run(prompt, on_token=on_token)

    if not agent.config.stream:
        console.print(format_content(result))
    else:
        console.print()

    agent.shutdown()
    agent.memory.save_all()