"""Terminal helpers shared by Nyx frontends."""
from __future__ import annotations

import os
import sys


def clear_terminal() -> None:
    """Clear the visible terminal and scrollback when stdout is interactive."""
    try:
        if not sys.stdout.isatty():
            return
    except Exception:
        return

    try:
        # ANSI clear screen + scrollback + cursor home. Supported by modern
        # Windows Terminal, PowerShell, cmd, and Unix-like terminals.
        sys.stdout.write("\033[2J\033[3J\033[H")
        sys.stdout.flush()
    except Exception:
        command = "cls" if os.name == "nt" else "clear"
        try:
            os.system(command)
        except Exception:
            pass
