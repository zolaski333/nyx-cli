"""
Nyx — Code search tool.

Provides ripgrep-based (rg) code search with fallback to grep/ag.
Returns structured, context-rich results suitable for LLM consumption.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class SearchMatch:
    """A single search match with context."""
    file: str
    line: int
    column: int = 0
    line_content: str = ""
    context_before: list[str] = field(default_factory=list)
    context_after: list[str] = field(default_factory=list)
    match_length: int = 0

    def formatted(self, context_lines: int = 2) -> str:
        """Format this match with surrounding context."""
        parts: list[str] = []
        parts.append(f"{self.file}:{self.line}:{self.column}")

        # Context before
        for i, line in enumerate(self.context_before[-context_lines:]):
            line_num = self.line - len(self.context_before[-context_lines:]) + i
            parts.append(f"  {line_num:>6} │ {line}")

        # The matched line
        parts.append(f"  {self.line:>6} › {self.line_content}")

        # Context after
        for i, line in enumerate(self.context_after[:context_lines]):
            line_num = self.line + i + 1
            parts.append(f"  {line_num:>6} │ {line}")

        return "\n".join(parts)


@dataclass
class SearchResult:
    """Result of a code search."""
    query: str
    matches: list[SearchMatch] = field(default_factory=list)
    total_matches: int = 0
    files_matched: int = 0
    duration_ms: float = 0.0
    error: str = ""
    engine: str = ""

    @property
    def summary(self) -> str:
        if self.error:
            return f"Search error: {self.error}"
        if self.total_matches == 0:
            return f"No results for: {self.query}"
        return (
            f"Found {self.total_matches} matches in {self.files_matched} files "
            f"(engine: {self.engine})"
        )

    def formatted(self, max_results: int = 30, context_lines: int = 2) -> str:
        """Format all matches into a readable string."""
        if self.error:
            return f"Search error: {self.error}"
        if not self.matches:
            return f"No results found for: {self.query}"

        parts: list[str] = []
        parts.append(f"🔍 Search results for: {self.query}")
        parts.append(f"   {self.summary}\n")

        for i, match in enumerate(self.matches[:max_results]):
            parts.append(f"[{i + 1}] {match.formatted(context_lines=context_lines)}\n")

        if len(self.matches) > max_results:
            parts.append(f"... and {len(self.matches) - max_results} more matches")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Engine detection
# ---------------------------------------------------------------------------


def _find_engine() -> str:
    """Find the best available search engine. Returns 'rg', 'grep', or ''."""
    if shutil.which("rg"):
        return "rg"
    if shutil.which("ag"):
        return "ag"
    if shutil.which("grep"):
        return "grep"
    return ""


# ---------------------------------------------------------------------------
# Ripgrep search
# ---------------------------------------------------------------------------


def _search_rg(
    pattern: str,
    root: str | Path,
    *,
    file_pattern: str | None = None,
    max_results: int = 100,
    context_lines: int = 2,
    case_sensitive: bool = False,
    regex: bool = False,
    fixed_strings: bool = False,
) -> SearchResult:
    """Search using ripgrep (rg)."""
    cmd = [
        "rg",
        "--json",
        "--no-heading",
        "-n",
        "--max-count", "50",  # max matches per file
    ]

    if context_lines > 0:
        cmd.extend(["-C", str(context_lines)])
    if not case_sensitive:
        cmd.append("-i")
    if fixed_strings:
        cmd.append("-F")
    if regex:
        cmd.append("-P")  # PCRE2 if available, fallback to default
    if file_pattern:
        cmd.extend(["-g", file_pattern])
    if max_results:
        cmd.extend(["-m", str(max_results)])

    # Path limit
    cmd.extend(["--max-filesize", "1M"])

    # Exclude common non-source dirs
    cmd.extend([
        "--glob", "!.git",
        "--glob", "!node_modules",
        "--glob", "!__pycache__",
        "--glob", "!.venv",
        "--glob", "!venv",
        "--glob", "!.tox",
        "--glob", "!.nyx_memory",
        "--glob", "!*.min.*",
        "--glob", "!*.map",
    ])

    cmd.append(pattern)
    cmd.append(str(root))

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return SearchResult(query=pattern, error="Search timed out", engine="rg")
    except FileNotFoundError:
        return SearchResult(query=pattern, error="rg not found", engine="rg")
    except Exception as e:
        return SearchResult(query=pattern, error=str(e), engine="rg")

    if proc.returncode not in (0, 1):
        return SearchResult(
            query=pattern,
            error=f"rg exited with code {proc.returncode}: {proc.stderr[:500]}",
            engine="rg",
        )

    if not proc.stdout:
        return SearchResult(query=pattern, engine="rg")

    # Parse JSON lines output
    matches: list[SearchMatch] = []
    files_matched: set[str] = set()
    parsed = 0

    # Buffer context lines
    pending_context: dict[int, dict[str, Any]] = {}

    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        data_type = data.get("type")
        obj = data.get("data", {})

        if data_type == "match":
            text = obj.get("lines", {}).get("text", "")
            path = obj.get("path", {}).get("text", "")
            line_num = obj.get("line_number", 0)
            col = obj.get("column", 0)
            submatches = obj.get("submatches", [])

            # Get context lines from the buffer
            before: list[str] = []
            after: list[str] = []
            if line_num in pending_context:
                ctx = pending_context.pop(line_num)
                before = ctx.get("before", [])
                after = ctx.get("after", [])

            # Build match length from submatch
            match_len = 0
            for sm in submatches:
                ml = sm.get("end", 0) - sm.get("start", 0)
                if ml > match_len:
                    match_len = ml

            match_obj = SearchMatch(
                file=path,
                line=line_num,
                column=col,
                line_content=text.rstrip("\n\r"),
                context_before=before,
                context_after=after,
                match_length=match_len,
            )
            matches.append(match_obj)
            files_matched.add(path)
            parsed += 1

        elif data_type == "context":
            text = obj.get("lines", {}).get("text", "").rstrip("\n\r")
            path = obj.get("path", {}).get("text", "")
            line_num = obj.get("line_number", 0)
            kind = obj.get("kind", "")

            # We store context lines for the next match
            if kind == "before":
                ctx = pending_context.setdefault(line_num + 1, {"before": [], "after": []})
                ctx["before"] = (ctx["before"] + [text])[-context_lines:]
            elif kind == "after":
                ctx = pending_context.setdefault(line_num - 1, {"before": [], "after": []})
                ctx["after"] = (ctx["after"] + [text])[:context_lines]

        if parsed >= max_results:
            break

    return SearchResult(
        query=pattern,
        matches=matches,
        total_matches=len(matches),
        files_matched=len(files_matched),
        engine="rg",
    )


# ---------------------------------------------------------------------------
# Grep fallback
# ---------------------------------------------------------------------------


def _search_grep(
    pattern: str,
    root: str | Path,
    *,
    file_pattern: str | None = None,
    max_results: int = 100,
    context_lines: int = 2,
    case_sensitive: bool = False,
    fixed_strings: bool = False,
) -> SearchResult:
    """Search using grep as fallback."""
    cmd = ["grep", "-n", "-r"]

    if context_lines > 0:
        cmd.extend(["-C", str(context_lines)])
    if not case_sensitive:
        cmd.append("-i")
    if fixed_strings:
        cmd.append("-F")
    else:
        cmd.append("-E")  # Extended regex

    # Exclude dirs
    cmd.extend([
        "--exclude-dir=.git",
        "--exclude-dir=node_modules",
        "--exclude-dir=__pycache__",
        "--exclude-dir=.venv",
        "--exclude-dir=venv",
        "--exclude-dir=.tox",
        "--exclude-dir=.nyx_memory",
    ])

    if file_pattern:
        cmd.extend(["--include", file_pattern])

    # Binary files
    cmd.append("-I")  # Ignore binary

    cmd.append(pattern)
    cmd.append(str(root))

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return SearchResult(query=pattern, error="Search timed out", engine="grep")
    except Exception as e:
        return SearchResult(query=pattern, error=str(e), engine="grep")

    if proc.returncode not in (0, 1):
        return SearchResult(query=pattern, error=f"grep error: {proc.stderr[:500]}", engine="grep")

    if not proc.stdout:
        return SearchResult(query=pattern, engine="grep")

    # Parse grep output
    matches: list[SearchMatch] = []
    files_matched: set[str] = set()
    current_match: SearchMatch | None = None
    current_context_before: list[str] = []
    current_context_after: list[str] = []
    current_file = ""
    current_line = 0

    # Grep -C output format:
    #   file:line:content
    #   file-line-context_before
    #   file-line-context_after
    #   --

    for line in proc.stdout.splitlines():
        stripped = line.rstrip("\n\r")

        # Separator between matches
        if stripped == "--":
            if current_match:
                matches.append(current_match)
                files_matched.add(current_match.file)
                current_match = None
            current_context_before = []
            current_context_after = []
            continue

        # Parse context or match line
        # Match lines have "file:line:content" format
        # Context lines have "file-line:content" format
        if ": " in stripped or ":-" in stripped[:3]:
            # Try to parse as file:line:content
            m = re.match(r"^([^:]+):(\d+):(.+)$", stripped)
            if m:
                if current_match:
                    matches.append(current_match)
                    files_matched.add(current_match.file)
                path = m.group(1)
                line_num = int(m.group(2))
                content = m.group(3)
                current_match = SearchMatch(
                    file=path,
                    line=line_num,
                    line_content=content,
                    context_before=list(current_context_before),
                    context_after=list(current_context_after),
                )
                current_context_before = []
                current_context_after = []
                current_file = path
                current_line = line_num
                continue

        # Context line: file-N-content
        m = re.match(r"^([^:]+)-(\d+)-(.+)$", stripped)
        if m:
            ctx_file = m.group(1)
            ctx_line = int(m.group(2))
            ctx_content = m.group(3)

            if current_match and ctx_file == current_match.file:
                if ctx_line < current_match.line:
                    current_context_before.append(ctx_content)
                else:
                    current_context_after.append(ctx_content)
            else:
                # Orphan context line — stash it
                if not current_match:
                    current_context_before.append(ctx_content)
            continue

    # Don't forget last match
    if current_match:
        matches.append(current_match)
        files_matched.add(current_match.file)

    # Trim to max
    matches = matches[:max_results]

    return SearchResult(
        query=pattern,
        matches=matches,
        total_matches=len(matches),
        files_matched=len(files_matched),
        engine="grep",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_code(
    pattern: str,
    root: str | Path | None = None,
    *,
    file_pattern: str | None = None,
    max_results: int = 100,
    context_lines: int = 2,
    case_sensitive: bool = False,
    regex: bool = False,
    fixed_strings: bool = False,
) -> SearchResult:
    """
    Search codebase using the best available engine (rg > ag > grep).

    Args:
        pattern: Search pattern (plain text or regex).
        root: Project root directory. Defaults to current working directory.
        file_pattern: Optional glob filter (e.g., "*.py", "*.rs").
        max_results: Maximum number of matches to return.
        context_lines: Number of context lines before/after each match.
        case_sensitive: If True, perform case-sensitive search.
        regex: If True, treat pattern as regex.
        fixed_strings: If True, treat pattern as literal string (overrides regex).

    Returns:
        SearchResult with matched lines and context.
    """
    root = Path(root).resolve() if root else Path.cwd().resolve()
    engine = _find_engine()

    if not engine:
        return SearchResult(
            query=pattern,
            error="No search engine found. Install ripgrep (rg), ag, or grep.",
        )

    kwargs = {
        "file_pattern": file_pattern,
        "max_results": max_results,
        "context_lines": context_lines,
        "case_sensitive": case_sensitive,
        "regex": regex or (not fixed_strings and bool(re.search(r'[.*+?^${}()|\[\]\\]', pattern))),
        "fixed_strings": fixed_strings,
    }

    if engine == "rg":
        result = _search_rg(pattern, root, **kwargs)
    elif engine == "ag":
        # Fall through to grep for now (ag is similar to rg)
        result = _search_grep(pattern, root, **kwargs)
        result.engine = "ag"
    else:
        result = _search_grep(pattern, root, **kwargs)

    return result


def search_symbol(
    symbol: str,
    root: str | Path | None = None,
    *,
    file_pattern: str | None = None,
    max_results: int = 50,
) -> SearchResult:
    """
    Search for a symbol definition or reference (class, function, variable).

    Shortcut for: search_code(symbol, regex=True, ...)
    """
    return search_code(
        pattern=symbol,
        root=root,
        file_pattern=file_pattern,
        max_results=max_results,
        context_lines=3,
        regex=True,
    )


def search_text(
    text: str,
    root: str | Path | None = None,
    *,
    file_pattern: str | None = None,
    case_sensitive: bool = False,
) -> SearchResult:
    """
    Search for literal text in the codebase.

    Shortcut for: search_code(text, fixed_strings=True, ...)
    """
    return search_code(
        pattern=text,
        root=root,
        file_pattern=file_pattern,
        fixed_strings=True,
        case_sensitive=case_sensitive,
    )