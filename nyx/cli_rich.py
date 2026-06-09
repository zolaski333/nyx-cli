"""
Nyx — Rich TUI.

A beautiful terminal UI using the Rich library.
Falls back gracefully to the basic CLI if Rich is not installed.
"""
from __future__ import annotations

import sys
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
    from rich import box
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
    table.add_row("/clear", "Clear conversation context")
    table.add_row("/tools", "List all available tools")
    table.add_row("/memory", "Show memory status")
    table.add_row("/conversations", "List saved conversations")
    table.add_row("/reset", "Reset agent (clear context + shutdown MCP)")
    table.add_row("/exit", "Exit the program")

    return Panel(table, box=box.ROUNDED, border_style="cyan", title="[bold]Commands[/bold]")


def tools_table(tools: list) -> Any:
    """Create a tools table."""
    if not RICH_AVAILABLE:
        return ""

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    table.add_column("Tool", style="green")
    table.add_column("Description", style="white", no_wrap=False)

    for t in tools:
        desc = t.description[:100] + ("..." if len(t.description) > 100 else "")
        table.add_row(t.name, desc)

    return Panel(table, box=box.ROUNDED, border_style="green", title=f"[bold]🔧 Tools ({len(tools)})[/bold]")


def format_content(content: str) -> Any:
    """Format agent response as Rich renderable."""
    if not RICH_AVAILABLE:
        return content

    # Try to detect code blocks and render appropriately
    if content.strip().startswith("```"):
        return Syntax(content, "python", theme="monokai", line_numbers=True)
    return Markdown(content)


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


def run_rich_interactive(agent: Agent, config: Config) -> None:
    """Run the interactive REPL with Rich formatting."""
    console = get_console()
    agent.on_command_approval = _make_rich_approval_handler(console)
    agent.on_file_approval = _make_rich_file_approval_handler(console)
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

        if user_input == "/tools":
            console.print(tools_table(agent.tools))
            continue

        if user_input == "/memory":
            conv = agent.memory.current
            if conv:
                table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
                table.add_column("Property", style="yellow")
                table.add_column("Value", style="white")
                table.add_row("Title", conv.title)
                table.add_row("Messages", str(len(conv.entries)))
                table.add_row("Total tokens", str(conv.total_tokens))
                table.add_row("Has summary", "✓" if conv.summary else "✗")
                table.add_row("Summary", conv.summary[:200] if conv.summary else "None")
                console.print(Panel(table, title="[bold]🧠 Memory[/bold]", border_style="cyan"))
            else:
                console.print("[yellow]No active conversation.[/yellow]")
            continue

        if user_input == "/conversations":
            convs = agent.memory.list_conversations()
            if not convs:
                console.print("[yellow]No saved conversations.[/yellow]")
                continue
            table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
            table.add_column("ID", style="dim")
            table.add_column("Title", style="white")
            table.add_column("Messages", style="blue")
            table.add_column("Summary", style="dim")
            for c in convs[:10]:
                table.add_row(
                    c["id"][:8],
                    c["title"][:40],
                    str(c["entry_count"]),
                    c["summary"][:60] if c["summary"] else "",
                )
            console.print(Panel(table, title="[bold]📂 Conversations[/bold]", border_style="cyan"))
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