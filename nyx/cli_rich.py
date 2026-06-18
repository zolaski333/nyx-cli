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
# Theme system
# ---------------------------------------------------------------------------

THEMES = {
    "cyberpunk": {
        "primary": "magenta",
        "secondary": "cyan",
        "accent": "yellow",
        "success": "green",
        "error": "red",
        "info": "blue",
        "dim": "dim cyan",
        "border": "magenta",
        "title": "bold cyan",
        "syntax_theme": "monokai"
    },
    "dracula": {
        "primary": "#bd93f9",   # Purple
        "secondary": "#ff79c6", # Pink
        "accent": "#f1fa8c",    # Yellow
        "success": "#50fa7b",   # Green
        "error": "#ff5555",     # Red
        "info": "#8be9fd",      # Cyan
        "dim": "dim #6272a4",    # Comment
        "border": "#bd93f9",
        "title": "bold #ff79c6",
        "syntax_theme": "dracula"
    },
    "nord": {
        "primary": "#81a1c1",   # Frost Blue
        "secondary": "#88c0d0", # Frost Ice
        "accent": "#ebcb8b",    # Yellow/Gold
        "success": "#a3be8c",   # Green
        "error": "#bf616a",     # Red
        "info": "#8fbcbb",      # Teal
        "dim": "dim #4c566a",    # Slate Gray
        "border": "#81a1c1",
        "title": "bold #88c0d0",
        "syntax_theme": "nord"
    },
    "sunset": {
        "primary": "#ff5f00",   # Bright Orange
        "secondary": "#ff005f", # Deep Rose
        "accent": "#ffd700",    # Gold
        "success": "#87af00",   # Olive Green
        "error": "#d70000",     # Dark Red
        "info": "#00afaf",      # Cyan
        "dim": "dim #8a8a8a",    # Dark Gray
        "border": "#ff5f00",
        "title": "bold #ff005f",
        "syntax_theme": "autumn"
    },
    "emerald": {
        "primary": "#00af5f",   # Emerald Green
        "secondary": "#5fafff", # Sky Blue
        "accent": "#d7af00",    # Amber
        "success": "#00d700",   # Lime Green
        "error": "#d75f5f",     # Red-pink
        "info": "#00afaf",      # Cyan
        "dim": "dim #808080",    # Mid Gray
        "border": "#00af5f",
        "title": "bold #5fafff",
        "syntax_theme": "emacs"
    }
}


def get_theme_palette(config: Config) -> dict[str, str]:
    """Get color palette for active theme."""
    theme_name = getattr(config, "theme", "cyberpunk").lower()
    return THEMES.get(theme_name, THEMES["cyberpunk"])


# ---------------------------------------------------------------------------
# Styled components
# ---------------------------------------------------------------------------


def welcome_panel(config: Config, tool_count: int) -> Any:
    """Create a welcome panel for the Rich TUI."""
    if not RICH_AVAILABLE:
        return ""

    get_console()
    theme = get_theme_palette(config)
    p = theme["primary"]
    s = theme["secondary"]
    a = theme["accent"]

    # Beautiful logo
    logo = f"""[bold {p}]███╗   ██╗[/bold {p}][bold {s}]██╗   ██╗[/bold {s}][bold {a}]██╗  ██╗[/bold {a}]
[bold {p}]████╗  ██║[/bold {p}][bold {s}]╚██╗ ██╔╝[/bold {s}][bold {a}]╚██╗██╔╝[/bold {a}]
[bold {p}]██╔██╗ ██║[/bold {p}][bold {s}] ╚████╔╝ [/bold {s}][bold {a}] ╚███╔╝ [/bold {a}]
[bold {p}]██║╚██╗██║[/bold {p}][bold {s}]  ╚██╔╝  [/bold {s}][bold {a}] ██╔██╗ [/bold {a}]
[bold {p}]██║ ╚████║[/bold {p}][bold {s}]   ██║   [/bold {s}][bold {a}]██╔╝ ██╗[/bold {a}]
[bold {p}]╚═╝  ╚═══╝[/bold {p}][bold {s}]   ╚═╝   [/bold {s}][bold {a}]╚═╝  ╚═╝[/bold {a}]"""

    # Grid layout for metadata
    meta_table = Table.grid(padding=(0, 4))
    meta_table.add_column()
    meta_table.add_column()

    # Left column content
    left_grid = Table.grid(padding=(0, 1))
    left_grid.add_column(style=f"bold {p}")
    left_grid.add_column()
    left_grid.add_row("Provider  ", f"[white]{config.provider}[/white]")
    left_grid.add_row("Model     ", f"[white]{config.model}[/white]")
    if config.project_dir:
        # Truncate long project paths nicely
        proj_path = str(config.project_dir)
        if len(proj_path) > 40:
            proj_path = "..." + proj_path[-37:]
        left_grid.add_row("Project   ", f"[white]{proj_path}[/white]")

    # Right column content
    right_grid = Table.grid(padding=(0, 1))
    right_grid.add_column(style=f"bold {s}")
    right_grid.add_column()
    right_grid.add_row("Mode      ", f"[white]{config.agent_mode}[/white]")
    right_grid.add_row("Autonomy  ", f"[white]{config.agent_autonomy}[/white]")
    right_grid.add_row("Tools     ", f"[white]{tool_count} active[/white]")

    meta_table.add_row(left_grid, right_grid)

    # Combined layout
    main_layout = Table.grid(padding=(1, 0))
    main_layout.add_column()
    main_layout.add_row(logo)
    main_layout.add_row(meta_table)
    main_layout.add_row("")
    main_layout.add_row(f"[dim]Type [/dim][bold {a}]/help[/bold {a}][dim] to view available commands • Theme: [/dim][bold {p}]{getattr(config, 'theme', 'cyberpunk')}[/bold {p}]")

    return Panel(
        main_layout,
        box=box.ROUNDED,
        border_style=theme["border"],
        title=f"[bold {p}]🚀 System Ready[/bold {p}]",
        title_align="left"
    )


def help_panel(config: Config) -> Any:
    """Create a help panel."""
    if not RICH_AVAILABLE:
        return ""

    theme = get_theme_palette(config)
    p = theme["primary"]
    s = theme["secondary"]

    table = Table(box=box.SIMPLE, show_header=True, header_style=f"bold {p}")
    table.add_column("Command", style=f"bold {s}")
    table.add_column("Description", style="white")

    table.add_row("/help", "Show this help")
    table.add_row("/model", "Show current model")
    table.add_row("/model <name>", "Change model")
    table.add_row("/mode <name>", "Switch mode: chat | code | architect | debug")
    table.add_row("/autonomy <lvl>", "Switch autonomy: ask | auto | yolo")
    table.add_row("/theme <theme>", "Switch UI theme: cyberpunk | dracula | nord | sunset | emerald")
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

    return Panel(table, box=box.ROUNDED, border_style=theme["border"], title=f"[bold {p}]Commands[/bold {p}]")


def tools_table(config: Config, tools: list[Any], page: int = 1, page_size: int = 10) -> Any:
    """Create a tools table with pagination."""
    if not RICH_AVAILABLE:
        return ""

    theme = get_theme_palette(config)
    p = theme["primary"]
    s = theme["secondary"]

    total = len(tools)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    end = min(start + page_size, total)

    table = Table(
        box=box.SIMPLE, show_header=True, header_style=f"bold {p}",
        title=f"[bold]🔧 Tools ({total}) — Page {page}/{total_pages}[/bold]",
    )
    table.add_column("Tool", style=f"bold {s}")
    table.add_column("Description", style="white", no_wrap=False)

    for t in tools[start:end]:
        desc = t.description[:100] + ("..." if len(t.description) > 100 else "")
        table.add_row(t.name, desc)

    return Panel(table, box=box.ROUNDED, border_style=theme["border"])


def memory_panel(agent: Agent, page: int = 1, page_size: int = 10) -> Any:
    """Show memory status with paginated entries."""
    if not RICH_AVAILABLE:
        return ""

    theme = get_theme_palette(agent.config)
    p = theme["primary"]
    s = theme["secondary"]
    a = theme["accent"]

    conv = agent.memory.current
    if not conv:
        return Panel(f"[{a}]No active conversation.[/{a}]", title=f"[bold {p}]🧠 Memory[/bold {p}]", border_style=theme["border"])

    table = Table(box=box.SIMPLE, show_header=True, header_style=f"bold {p}")
    table.add_column("Property", style=f"bold {s}")
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
            box=box.SIMPLE, show_header=True, header_style=f"bold {p}",
            title=f"Entries (Page {page}/{total_pages})",
        )
        entries_table.add_column("#", style="dim", width=4)
        entries_table.add_column("Role", style=f"bold {s}", width=10)
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
            title=f"[bold {p}]🧠 Memory[/bold {p}]",
            border_style=theme["border"],
        )

    return Panel(table, title=f"[bold {p}]🧠 Memory[/bold {p}]", border_style=theme["border"])


def conversations_panel(agent: Agent, page: int = 1, page_size: int = 10) -> Any:
    """List conversations with pagination."""
    if not RICH_AVAILABLE:
        return ""

    theme = get_theme_palette(agent.config)
    p = theme["primary"]
    s = theme["secondary"]
    a = theme["accent"]

    convs = agent.memory.list_conversations()
    if not convs:
        return Panel(f"[{a}]No saved conversations.[/{a}]", title=f"[bold {p}]📂 Conversations[/bold {p}]", border_style=theme["border"])

    total = len(convs)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    end = min(start + page_size, total)

    table = Table(
        box=box.SIMPLE, show_header=True, header_style=f"bold {p}",
        title=f"[bold]📂 Conversations ({total}) — Page {page}/{total_pages}[/bold]",
    )
    table.add_column("ID", style="dim")
    table.add_column("Title", style="white")
    table.add_column("Messages", style=f"bold {s}")
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

    return Panel(table, box=box.ROUNDED, border_style=theme["border"])


def format_content(content: str, theme_name: str = "cyberpunk") -> Any:
    """Format agent response as Rich renderable."""
    if not RICH_AVAILABLE:
        return content

    theme = THEMES.get(theme_name, THEMES["cyberpunk"])
    # Try to detect code blocks and render appropriately
    if content.strip().startswith("```"):
        return Syntax(content, "python", theme=theme["syntax_theme"], line_numbers=True)
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


def _make_rich_on_token(theme_name: str = "cyberpunk") -> Callable[[str], None]:
    """Create a streaming callback that uses Rich for display."""
    buffer: list[str] = []
    console = get_console()
    theme = THEMES.get(theme_name, THEMES["cyberpunk"])
    color = theme["primary"]

    def on_token(token: str) -> None:
        buffer.append(token)
        console.print(token, end="", style=color)

    return on_token


class RichMarkdownStreamer:
    """Render streamed tokens as a live-updating Markdown block."""

    def __init__(self, console: Any, theme_name: str = "cyberpunk") -> None:
        self.console = console
        self.theme_name = theme_name
        self.buffer: list[str] = []
        self._live: Any = None

    def start(self) -> None:
        self._live = Live(
            format_content("", self.theme_name),
            console=self.console,
            refresh_per_second=8,
            transient=False,
        )
        self._live.start()

    def on_token(self, token: str) -> None:
        self.buffer.append(token)
        if self._live:
            self._live.update(format_content("".join(self.buffer), self.theme_name))

    def finish(self, final_text: str | None = None) -> None:
        if final_text is not None:
            self.buffer = [final_text]
        if self._live:
            self._live.update(format_content("".join(self.buffer), self.theme_name))
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
        self.agent = agent
        self.config = config
        from nyx.cli import setup_readline
        setup_readline(agent)
        agent.on_command_approval = _make_rich_approval_handler(self.console)
        agent.on_file_approval = _make_rich_file_approval_handler(self.console)
        self.console.clear()
        self.console.print(welcome_panel(config, len(agent.tools)))

    def read_input(self) -> str:
        theme = get_theme_palette(self.config)
        p = theme["primary"]
        s = theme["secondary"]
        mode = self.config.agent_mode
        autonomy = self.config.agent_autonomy
        prompt_str = f"\n[bold {p}]nyx[/bold {p}] [dim]({mode} • {autonomy})[/dim] [bold {s}]❯[/bold {s} "
        
        try:
            import readline
            import re
            with self.console.capture() as capture:
                self.console.print(prompt_str, end="")
            ansi_prompt = capture.get()
            safe_prompt = re.sub(r"(\033\[[0-9;]*[a-zA-Z])", r"\001\1\002", ansi_prompt)
            return input(safe_prompt)
        except Exception:
            return str(self.console.input(prompt_str))

    def append_history(self, text: str) -> None:
        self.history.append(text)

    def show_bye(self) -> None:
        theme = get_theme_palette(self.config)
        p = theme["primary"]
        self.console.print(f"[bold {p}]Bye![/bold {p}]")

    def show_help(self) -> None:
        self.console.print(help_panel(self.config))

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
        self.console.print(tools_table(self.config, agent.tools, page=page))

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

    def show_theme_status(self, theme_name: str, available_themes: list[str]) -> None:
        theme = get_theme_palette(self.config)
        p = theme["primary"]
        s = theme["secondary"]

        grid = Table.grid(padding=(0, 2))
        grid.add_column()
        grid.add_column()

        grid.add_row(f"[bold {p}]Active Theme:[/bold {p}]", f"[white]{theme_name}[/white]")
        grid.add_row("", "")
        grid.add_row(f"[bold {s}]Available Themes:[/bold {s}]", "")
        for t in available_themes:
            marker = "  ← active" if t == theme_name else ""
            t_palette = THEMES[t]
            t_primary = t_palette["primary"]
            grid.add_row("", f"[bold {t_primary}]{t}[/bold {t_primary}]{marker}")

        self.console.print(Panel(
            grid,
            box=box.ROUNDED,
            border_style=theme["border"],
            title=f"[bold {p}]🎨 Themes[/bold {p}]"
        ))

    def show_theme_changed(self, theme_name: str) -> None:
        theme = get_theme_palette(self.config)
        p = theme["primary"]
        self.console.print(f"[bold {p}]🎨 Theme switched to:[/bold {p}] [bold white]{theme_name}[/bold white]")

    def show_conversation_not_found(self, conv_id: str) -> None:
        self.console.print(f"[yellow]Conversation not found: {conv_id}[/yellow]")

    def make_on_token(self) -> Callable[[str], None]:
        if self._streamer is not None:
            return self._streamer.on_token
        return _make_rich_on_token(self.config.theme)

    def before_agent_response(self, *, stream: bool) -> None:
        if stream:
            theme = get_theme_palette(self.config)
            p = theme["primary"]
            self.console.print(f"[bold {p}]Agent[/bold {p}]❯")
            self._streamer = RichMarkdownStreamer(self.console, self.config.theme)
            self._streamer.start()

    def show_agent_result(self, result: str, *, stream: bool) -> None:
        if not stream:
            theme = get_theme_palette(self.config)
            p = theme["primary"]
            self.console.print(f"[bold {p}]Agent[/bold {p}]❯")
            self.console.print(format_content(result, self.config.theme))
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
    theme_name = getattr(agent.config, "theme", "cyberpunk")

    if agent.config.stream:
        theme = get_theme_palette(agent.config)
        p = theme["primary"]
        console.print(f"[bold {p}]Agent[/bold {p}]❯")
        streamer = RichMarkdownStreamer(console, theme_name)
        streamer.start()

    try:
        result = agent.run(prompt, on_token=streamer.on_token if streamer else _make_rich_on_token(theme_name))

        if not agent.config.stream:
            theme = get_theme_palette(agent.config)
            p = theme["primary"]
            console.print(f"[bold {p}]Agent[/bold {p}]❯")
            console.print(format_content(result, theme_name))
        elif streamer:
            streamer.finish(result)
    except Exception:
        if streamer:
            streamer.finish()
        raise
    finally:
        agent.shutdown()
        agent.memory.save_all()
