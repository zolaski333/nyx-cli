"""
Nyx — Rich TUI.

A beautiful terminal UI using the Rich library.
Falls back gracefully to the basic CLI if Rich is not installed.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from nyx.approval import run_exclusive_approval

# Try to import Rich; fall back to basic CLI if unavailable
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.syntax import Syntax
    from rich.live import Live
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        TaskProgressColumn, TimeElapsedColumn,
    )
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from nyx.config import Config
from nyx.agent import Agent
from nyx.repl_controller import run_interactive_repl


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

    get_console()
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
    table.add_row("/config", "Show configuration status")
    table.add_row("/config save [--global]", "Save current session config")
    table.add_row("/config set [--global] <key> <val>", "Set config option")
    table.add_row("/clear", "Clear conversation context")
    table.add_row("/tools [N]", "List tools (paginated, optional page N)")
    table.add_row("/memory [N]", "Show memory status (paginated entries)")
    table.add_row("/conversations [N]", "List saved conversations (paginated)")
    table.add_row("/switch <id>", "Switch to a saved conversation")
    table.add_row("/reset", "Reset agent (clear context + shutdown MCP)")
    table.add_row("/exit", "Exit the program")

    return Panel(table, box=box.ROUNDED, border_style="cyan", title="[bold]Commands[/bold]")


def tools_table(tools: list[Any], page: int = 1, page_size: int = 10) -> Any:
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

        grid = Table.grid(padding=(0, 1))
        grid.add_row(table)
        grid.add_row(entries_table)
        return Panel(
            grid,
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


class RichMarkdownStreamer:
    """Render streamed tokens as a live-updating Markdown block."""

    def __init__(self, console: Any) -> None:
        self.console = console
        self.buffer: list[str] = []
        self._live: Any = None

    def start(self) -> None:
        self._live = Live(
            format_content(""),
            console=self.console,
            refresh_per_second=8,
            transient=False,
        )
        self._live.start()

    def on_token(self, token: str) -> None:
        self.buffer.append(token)
        if self._live:
            self._live.update(format_content("".join(self.buffer)))

    def finish(self, final_text: str | None = None) -> None:
        if final_text is not None:
            self.buffer = [final_text]
        if self._live:
            self._live.update(format_content("".join(self.buffer)))
            self._live.stop()
            self._live = None


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


def _make_rich_approval_handler(console: Any) -> Callable[[str], tuple[bool, str]]:
    """Create an interactive approval handler using Rich prompts."""
    def handle_approval(command: str) -> tuple[bool, str]:
        def prompt() -> tuple[bool, str]:
            console.print("\n[bold yellow]⚠️  SECURITY[/bold yellow] The AI wants to execute a potentially dangerous command:")
            console.print(f"  [cyan]{command}[/cyan]")
            response = console.input("  [bold]Allow?[/bold] (y/n): ").strip().lower()
            if response == "y":
                return True, ""
            reason = console.input("  [dim]Reason for denial:[/dim] ").strip()
            return False, reason or "User denied the command."
        return run_exclusive_approval(prompt)
    return handle_approval


def _make_rich_file_approval_handler(console: Any) -> Callable[[str, str, str], tuple[bool, str]]:
    """Create an interactive approval handler for file operations using Rich."""
    def handle_file_approval(path: str, summary: str, diff: str) -> tuple[bool, str]:
        from rich.syntax import Syntax
        from rich.panel import Panel

        def prompt() -> tuple[bool, str]:
            console.print("\n[bold cyan]📝 FILE OPERATION[/bold cyan]")
            console.print(f"  [bold]{summary}[/bold]")

            if diff:
                # Determine change type and styling
                border_style = "cyan"
                title = "[bold]Changes[/bold]"
                if "CREATE" in summary:
                    border_style = "green"
                    title = f"[bold green]CREATE: {path}[/bold green]"
                elif "DELETE" in summary:
                    border_style = "red"
                    title = f"[bold red]DELETE: {path}[/bold red]"
                elif "MODIFY" in summary or "APPEND" in summary or "Patch" in summary:
                    border_style = "yellow"
                    title = f"[bold yellow]MODIFY: {path}[/bold yellow]"

                # Show diff with syntax highlighting
                try:
                    syntax = Syntax(diff[:5000], "diff", theme="monokai", line_numbers=True)
                    console.print(Panel(syntax, title=title, border_style=border_style))
                except Exception:
                    console.print(f"  [dim]{diff[:5000]}[/dim]")

                if len(diff) > 5000:
                    console.print(f"  [dim](... diff truncated, {len(diff)} total chars)[/dim]")

            response = console.input("  [bold]Apply this change?[/bold] (y/n): ").strip().lower()
            if response == "y":
                return True, ""
            reason = console.input("  [dim]Reason for denial:[/dim] ").strip()
            return False, reason or "User denied the file change."
        return run_exclusive_approval(prompt)
    return handle_file_approval



def _get_paginated_arg(user_input: str, cmd: str) -> int:
    """Extract page number from '/cmd N' input."""
    rest = user_input[len(cmd):].strip()
    try:
        return int(rest.split()[0]) if rest else 1
    except (ValueError, IndexError):
        return 1


def _rich_autocomplete(console: Any, partial: str) -> str:
    """Rich autocompletion: show matching commands and return completed prefix."""
    commands = [
        "/help", "/model", "/mode", "/autonomy", "/config", "/clear", "/tools", "/memory",
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


class RichReplUI:
    """Rich rendering adapter for the shared REPL controller."""

    def __init__(self, console: Any) -> None:
        self.console = console
        self.history = RichREPLHistory()
        self._streamer: RichMarkdownStreamer | None = None

    def setup(self, agent: Agent, config: Config) -> None:
        from nyx.cli import setup_readline
        setup_readline(agent)
        agent.on_command_approval = _make_rich_approval_handler(self.console)
        agent.on_file_approval = _make_rich_file_approval_handler(self.console)
        self.console.clear()
        self.console.print(welcome_panel(config, len(agent.tools)))

    def read_input(self) -> str:
        return str(self.console.input("\n[bold green]You[/bold green]> "))

    def append_history(self, text: str) -> None:
        self.history.append(text)

    def show_bye(self) -> None:
        self.console.print("[bold green]Bye![/bold green]")

    def show_help(self) -> None:
        self.console.print(help_panel())

    def show_context_cleared(self) -> None:
        self.console.print("[yellow]Context cleared.[/yellow]")

    def show_agent_reset(self) -> None:
        self.console.print("[yellow]Agent reset. MCP disconnected.[/yellow]")

    def show_model(self, model: str) -> None:
        self.console.print(f"[bold]Current model:[/bold] [yellow]{model}[/yellow]")

    def show_model_changed(self, model: str) -> None:
        self.console.print(f"[bold]Model changed:[/bold] [yellow]{model}[/yellow]")

    def show_status(self, message: str, *, success: bool = False) -> None:
        style = "green" if success else "yellow"
        self.console.print(f"[{style}]{message}[/{style}]")

    def show_mode_status(self, mode: str, autonomy: str) -> None:
        self.console.print(
            f"[bold]Current mode:[/bold] [yellow]{mode}[/yellow]  |  autonomy: [cyan]{autonomy}[/cyan]"
        )

    def show_autonomy_status(self, autonomy: str) -> None:
        self.console.print(f"[bold]Current autonomy:[/bold] [cyan]{autonomy}[/cyan]")

    def show_config_status(self, config: Config, paths: list[tuple[str, Path]]) -> None:
        table = Table(box=box.SIMPLE, show_header=False)
        table.add_row("[bold]Active Provider[/bold]", config.provider)
        table.add_row("[bold]Active Model[/bold]", config.model)
        table.add_row("[bold]Active Mode[/bold]", config.agent_mode)
        table.add_row("[bold]Active Autonomy[/bold]", config.agent_autonomy)

        paths_table = Table(box=box.SIMPLE, show_header=True)
        paths_table.add_column("Level", style="bold")
        paths_table.add_column("Path", style="dim")
        paths_table.add_column("Status")
        for name, path in paths:
            status = "[green]exists[/green]" if path.exists() else "[dim]not found[/dim]"
            paths_table.add_row(name, str(path), status)

        self.console.print(Panel(table, title="[bold cyan]Active Session Configuration[/bold cyan]", border_style="cyan"))
        self.console.print(Panel(paths_table, title="[bold]Configuration Files[/bold]", border_style="dim"))
        self.console.print("[dim]Use `/config save` to persist current session settings to project config.[/dim]")
        self.console.print("[dim]Use `/config save --global` to persist current session settings globally.[/dim]")
        self.console.print("[dim]Use `/config set <key> <value>` to change a config option.[/dim]")

    def show_config_saved(self, path: Path) -> None:
        self.console.print(f"[green]Successfully saved session config to:[/green] {path}")

    def show_config_set(self, key: str, value: Any, path: Path) -> None:
        self.console.print(f"[green]Updated config key `{key}` to `{value}` in:[/green] {path}")

    def show_config_error(self, message: str) -> None:
        self.console.print(f"[red]{message}[/red]")

    def show_tools(self, agent: Agent, page: int) -> None:
        self.console.print(tools_table(agent.tools, page=page))

    def show_memory(self, agent: Agent, page: int) -> None:
        self.console.print(memory_panel(agent, page=page))

    def show_conversations(self, agent: Agent, page: int) -> None:
        self.console.print(conversations_panel(agent, page=page))

    def show_switched_conversation(self, title: str) -> None:
        self.console.print(f"[green]Switched to conversation:[/green] {title}")

    def show_multiple_conversation_matches(self, matches: list[Any]) -> None:
        self.console.print("[yellow]Multiple matches - use full ID:[/yellow]")
        for match in matches:
            self.console.print(f"  [cyan]{match.id}[/cyan] {match.title}")

    def show_conversation_not_found(self, conv_id: str) -> None:
        self.console.print(f"[yellow]Conversation not found: {conv_id}[/yellow]")

    def make_on_token(self) -> Callable[[str], None]:
        if self._streamer is not None:
            return self._streamer.on_token
        return _make_rich_on_token()

    def before_agent_response(self, *, stream: bool) -> None:
        if stream:
            self.console.print("[bold magenta]Agent[/bold magenta]>")
            self._streamer = RichMarkdownStreamer(self.console)
            self._streamer.start()

    def show_agent_result(self, result: str, *, stream: bool) -> None:
        if not stream:
            self.console.print(format_content(result))
        elif self._streamer is not None:
            self._streamer.finish(result)
            self._streamer = None

    def show_error(self, error: Exception) -> None:
        if self._streamer is not None:
            self._streamer.finish()
            self._streamer = None
        self.console.print(f"\n[bold red]Error: {error}[/bold red]")


def run_rich_interactive(agent: Agent, config: Config) -> None:
    """Run the interactive REPL with Rich formatting."""
    console = get_console()
    run_interactive_repl(agent, config, RichReplUI(console))


# ---------------------------------------------------------------------------
# Rich single-prompt mode
# ---------------------------------------------------------------------------


def run_rich_single(agent: Agent, prompt: str) -> None:
    """Run a single prompt with Rich formatting."""
    console = get_console()
    # Wire approval callbacks (same as interactive mode)
    agent.on_command_approval = _make_rich_approval_handler(console)
    agent.on_file_approval = _make_rich_file_approval_handler(console)
    streamer: RichMarkdownStreamer | None = None

    if agent.config.stream:
        console.print("[bold magenta]Agent[/bold magenta]>")
        streamer = RichMarkdownStreamer(console)
        streamer.start()

    try:
        result = agent.run(prompt, on_token=streamer.on_token if streamer else _make_rich_on_token())

        if not agent.config.stream:
            console.print(format_content(result))
        elif streamer:
            streamer.finish(result)
    except Exception:
        if streamer:
            streamer.finish()
        raise
    finally:
        agent.shutdown()
        agent.memory.save_all()
