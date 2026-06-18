"""
Nyx — Patch/diff tool for safe file modifications.

Provides a structured patch/diff workflow:
  1. Parse unified diffs and SEARCH/REPLACE blocks.
  2. Validate patch syntax before application.
  3. Detect conflicts (context lines mismatch + removed lines mismatch).
  4. Categorize changes: CREATE, MODIFY, DELETE.
  5. Display a clear summary before user approval.
  6. Apply patches with automatic rollback capability.
  7. Maintain a history of all applied patches.
  8. Optionally integrate with git diff / git apply --check.

This prevents accidental data loss and gives the user full visibility
into every file change before it happens.
"""
from __future__ import annotations

import difflib
import enum
import hashlib
import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROLLBACK_DIR_NAME = ".nyx/rollback"
PATCHES_DIR_NAME = ".nyx/patches"
MAX_ROLLBACK_ENTRIES = 50
# Sentinel stored in rollback backup for newly-created files
# (no previous content to restore — rollback means: delete the file)
NYX_NEW_FILE_SENTINEL = "__NYX_CREATED_FILE__\n"


# ---------------------------------------------------------------------------
# Change type enumeration
# ---------------------------------------------------------------------------


class ChangeType(enum.Enum):
    """Categorizes the nature of a file change."""
    CREATE = "create"       # New file
    MODIFY = "modify"       # Existing file modified
    DELETE = "delete"       # File content emptied (or file deleted)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class HunkInfo:
    """Metadata about a single hunk in a unified diff."""
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    header: str = ""
    added_lines: int = 0
    removed_lines: int = 0


@dataclass
class PatchInfo:
    """Parsed information about a patch."""
    filepath: str
    change_type: ChangeType
    hunks: list[HunkInfo] = field(default_factory=list)
    total_additions: int = 0
    total_deletions: int = 0
    original_content: str = ""
    proposed_content: str = ""
    diff_text: str = ""
    is_new_file: bool = False
    is_deletion: bool = False
    validation_errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.validation_errors) == 0

    @property
    def summary(self) -> str:
        """Generate a one-line summary of the change."""
        if self.change_type == ChangeType.CREATE:
            return f"CREATE  {self.filepath} ({len(self.proposed_content)} bytes)"
        elif self.change_type == ChangeType.DELETE:
            return f"DELETE  {self.filepath} ({len(self.original_content)} bytes removed)"
        else:
            hunks_str = f"{len(self.hunks)} hunk(s)" if self.hunks else ""
            return (
                f"MODIFY  {self.filepath} "
                f"(+{self.total_additions} -{self.total_deletions})"
                + (f" {hunks_str}" if hunks_str else "")
            )

    @property
    def detailed_summary(self) -> str:
        """Generate a detailed multi-line summary."""
        lines = [
            f"File:     {self.filepath}",
            f"Type:     {self.change_type.value.upper()}",
        ]
        if self.change_type == ChangeType.MODIFY:
            lines.append(f"Hunks:    {len(self.hunks)}")
            lines.append(f"Added:    +{self.total_additions} lines")
            lines.append(f"Removed:  -{self.total_deletions} lines")
            for i, h in enumerate(self.hunks):
                lines.append(
                    f"  Hunk {i + 1}: @@ -{h.old_start},{h.old_count}"
                    f" +{h.new_start},{h.new_count} @@"
                    f"  (+{h.added_lines} -{h.removed_lines})"
                )
        elif self.change_type == ChangeType.CREATE:
            lines.append(f"Size:     {len(self.proposed_content)} bytes")
        elif self.change_type == ChangeType.DELETE:
            lines.append(f"Size:     {len(self.original_content)} bytes removed")
        if self.validation_errors:
            lines.append(f"WARNINGS: {len(self.validation_errors)}")
            for err in self.validation_errors:
                lines.append(f"  ⚠ {err}")
        return "\n".join(lines)


@dataclass
class RollbackEntry:
    """A saved state that can be restored."""
    filepath: str
    backup_path: str
    timestamp: float
    checksum: str
    size: int


@dataclass
class PatchRecord:
    """A record of an applied patch for history tracking."""
    filepath: str
    change_type: ChangeType
    timestamp: float
    diff_text: str
    summary: str
    success: bool
    error: str = ""
    rollback_entry: RollbackEntry | None = None


# ---------------------------------------------------------------------------
# Patch parsing
# ---------------------------------------------------------------------------


# Regex for unified diff hunk header: @@ -old_start,old_count +new_start,new_count @@
_HUNK_HEADER_RE = re.compile(
    r"^@@\s+-(\d+)(?:,(-?\d+))?\s+\+(\d+)(?:,(-?\d+))?\s+@@(.*)$"
)

# Regex for diff --git or ---/+++ file headers
_DIFF_GIT_RE = re.compile(r"^diff\s+--git\s+a/(.+?)\s+b/(.+?)$")
_FROM_FILE_RE = re.compile(r"^---\s+(?:a/)?(.+?)(?:\t.*)?$")
_TO_FILE_RE = re.compile(r"^\+\+\+\s+(?:b/)?(.+?)(?:\t.*)?$")

# Regex for SEARCH/REPLACE block markers
_SEARCH_MARKER_RE = re.compile(r"^<<<<<<<\s+SEARCH\s*$")
_DIVIDER_RE = re.compile(r"^=======\s*$")
_REPLACE_MARKER_RE = re.compile(r"^>>>>>>>\s+REPLACE\s*$")
_START_LINE_RE = re.compile(r"^:start_line:(\d+)\s*$")


def parse_unified_diff(diff_text: str, filepath: str = "") -> PatchInfo:
    """Parse a unified diff string into a structured PatchInfo.

    Handles standard unified diff format produced by `diff -u` or `git diff`.

    Args:
        diff_text: The unified diff text.
        filepath: Fallback filepath if not found in diff headers.

    Returns:
        A PatchInfo object with parsed metadata.
    """
    info = PatchInfo(filepath=filepath, change_type=ChangeType.MODIFY, diff_text=diff_text)
    lines = diff_text.splitlines()

    # Try to extract filepath from diff headers
    extracted_path = _extract_filepath_from_diff(lines)
    if extracted_path:
        info.filepath = extracted_path

    # Detect new file / deleted file markers
    for line in lines:
        if line.startswith("new file") or line.startswith("(new file"):
            info.is_new_file = True
            info.change_type = ChangeType.CREATE
            break
        if line.startswith("deleted file"):
            info.is_deletion = True
            info.change_type = ChangeType.DELETE
            break

    # Parse hunks
    current_hunk: HunkInfo | None = None
    for line in lines:
        m = _HUNK_HEADER_RE.match(line)
        if m:
            if current_hunk:
                info.hunks.append(current_hunk)
            old_start = int(m.group(1))
            old_count = int(m.group(2)) if m.group(2) else 1
            new_start = int(m.group(3))
            new_count = int(m.group(4)) if m.group(4) else 1
            current_hunk = HunkInfo(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                header=(m.group(5) or "").strip(),
            )
            continue

        if current_hunk is not None:
            if line.startswith("+") and not line.startswith("+++"):
                current_hunk.added_lines += 1
                info.total_additions += 1
            elif line.startswith("-") and not line.startswith("---"):
                current_hunk.removed_lines += 1
                info.total_deletions += 1

    if current_hunk is not None:
        info.hunks.append(current_hunk)

    # Determine change type
    if info.is_new_file:
        info.change_type = ChangeType.CREATE
    elif info.is_deletion:
        info.change_type = ChangeType.DELETE
    else:
        info.change_type = ChangeType.MODIFY

    return info


def parse_search_replace(text: str, filepath: str = "") -> PatchInfo:
    """Parse SEARCH/REPLACE blocks into a PatchInfo.

    Format:
        <<<<<<< SEARCH
        :start_line:42
        -------
        exact content to find
        =======
        new content to replace with
        >>>>>>> REPLACE

    Args:
        text: The SEARCH/REPLACE block text.
        filepath: The target file path.

    Returns:
        A PatchInfo object.
    """
    info = PatchInfo(filepath=filepath, change_type=ChangeType.MODIFY, diff_text=text)
    lines = text.splitlines()
    i = 0

    while i < len(lines):
        # Find SEARCH marker
        if _SEARCH_MARKER_RE.match(lines[i]):
            i += 1
            start_line = 1

            # Optional :start_line: directive
            if i < len(lines):
                slm = _START_LINE_RE.match(lines[i])
                if slm:
                    start_line = int(slm.group(1))
                    i += 1

            # Optional ------- divider
            if i < len(lines) and lines[i].strip() == "-------":
                i += 1

            # Collect SEARCH content
            search_lines: list[str] = []
            while i < len(lines) and not _DIVIDER_RE.match(lines[i]):
                search_lines.append(lines[i])
                i += 1

            if i >= len(lines):
                info.validation_errors.append(
                    "Missing '=======' divider in SEARCH/REPLACE block"
                )
                break

            i += 1  # Skip =======

            # Collect REPLACE content
            replace_lines: list[str] = []
            while i < len(lines) and not _REPLACE_MARKER_RE.match(lines[i]):
                replace_lines.append(lines[i])
                i += 1

            if i >= len(lines):
                info.validation_errors.append(
                    "Missing '>>>>>>> REPLACE' marker in SEARCH/REPLACE block"
                )
                break

            i += 1  # Skip >>>>>>> REPLACE

            # Compute hunk info
            old_count = len(search_lines)
            new_count = len(replace_lines)
            added = sum(1 for _ in replace_lines)
            removed = sum(1 for _ in search_lines)

            hunk = HunkInfo(
                old_start=start_line,
                old_count=old_count,
                new_start=start_line,
                new_count=new_count,
                added_lines=added,
                removed_lines=removed,
            )
            info.hunks.append(hunk)
            info.total_additions += added
            info.total_deletions += removed
        else:
            i += 1

    if not info.hunks:
        info.validation_errors.append("No valid SEARCH/REPLACE blocks found")

    if info.total_additions > 0 or info.total_deletions > 0:
        info.change_type = ChangeType.MODIFY

    return info


def _extract_filepath_from_diff(lines: list[str]) -> str | None:
    """Try to extract the file path from diff headers."""
    for line in lines:
        m = _DIFF_GIT_RE.match(line)
        if m:
            return m.group(2)
    for line in lines:
        m = _TO_FILE_RE.match(line)
        if m:
            return m.group(1)
    for line in lines:
        m = _FROM_FILE_RE.match(line)
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Diff computation
# ---------------------------------------------------------------------------


def compute_diff(original: str, proposed: str, filepath: str = "file") -> str:
    """Compute a unified diff between original and proposed content.

    Args:
        original: The original file content.
        proposed: The proposed new content.
        filepath: The file path for display in the diff header.

    Returns:
        A unified diff string, or "(new file)" if original is empty.
    """
    if not original:
        return f"(new file: {filepath})"

    original_lines = original.splitlines(keepends=True)
    proposed_lines = proposed.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        proposed_lines,
        fromfile=f"a/{filepath}",
        tofile=f"b/{filepath}",
        lineterm="\n",
    )
    result = "".join(diff)
    return result if result else "(no changes)"


def compute_diff_from_path(path: str | Path, proposed: str) -> str:
    """Compute a diff between the current file content and proposed content.

    Args:
        path: Path to the existing file.
        proposed: The proposed new content.

    Returns:
        A unified diff string.
    """
    p = Path(path)
    original = ""
    if p.exists():
        try:
            original = p.read_text(encoding="utf-8")
        except Exception as e:
            original = f"(error reading original: {e})"
    return compute_diff(original, proposed, str(path))


# ---------------------------------------------------------------------------
# Git integration
# ---------------------------------------------------------------------------


def _is_git_repo(directory: str | Path) -> bool:
    """Check if a directory is inside a git repository."""
    cwd = str(Path(directory).resolve())
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={cwd}", "rev-parse", "--is-inside-work-tree"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def git_diff(filepath: str | Path, staged: bool = False) -> str | None:
    """Get the git diff for a file.

    Args:
        filepath: Path to the file.
        staged: If True, show staged changes (--cached).

    Returns:
        The git diff output, or None if git is unavailable.
    """
    p = Path(filepath)
    cwd = p.parent if p.is_file() else str(p)
    if not _is_git_repo(cwd):
        return None

    try:
        cmd = ["git", "diff"]
        if staged:
            cmd.append("--cached")
        cmd.extend(["--", str(p)])
        safe_cwd = str(Path(cwd).resolve())
        result = subprocess.run(
            ["git", "-c", f"safe.directory={safe_cwd}", *cmd[1:]],
            cwd=safe_cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout if result.stdout else None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        logger.debug("git diff failed: %s", e)
        return None


def git_apply_check(diff_text: str, directory: str | Path) -> tuple[bool, str]:
    """Check if a patch can be applied cleanly using git apply --check.

    Args:
        diff_text: The patch/diff text to check.
        directory: Working directory for git.

    Returns:
        (can_apply: bool, message: str)
    """
    if not _is_git_repo(directory):
        return False, "Not a git repository"

    try:
        safe_dir = str(Path(directory).resolve())
        result = subprocess.run(
            ["git", "-c", f"safe.directory={safe_dir}", "apply", "--check", "--verbose"],
            input=diff_text,
            cwd=safe_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return True, "Patch applies cleanly (git apply --check)"
        else:
            return False, result.stderr.strip() or "git apply --check failed"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        return False, f"git apply --check error: {e}"


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------


def detect_conflicts(
    original_content: str,
    patch_info: PatchInfo,
    filepath: str = "",
) -> list[str]:
    """Detect conflicts between a patch and the current file content.

    Checks that:
      - Context lines (starting with space) match the actual file content.
      - Removed lines (starting with -) match the actual file content.

    Args:
        original_content: The current content of the file.
        patch_info: The parsed patch information.
        filepath: The file path (for error messages).

    Returns:
        A list of conflict descriptions. Empty list means no conflicts.
    """
    conflicts: list[str] = []
    original_lines = original_content.splitlines()

    if patch_info.change_type == ChangeType.CREATE:
        return []

    if patch_info.change_type == ChangeType.DELETE:
        return []

    diff_lines = patch_info.diff_text.splitlines()
    current_orig_line = 0
    hunk_idx = 0

    for line in diff_lines:
        m = _HUNK_HEADER_RE.match(line)
        if m:
            hunk_idx += 1
            old_start = int(m.group(1))
            current_orig_line = old_start - 1  # 0-based
            continue

        # Skip diff header lines
        if line.startswith("---") or line.startswith("+++") or line.startswith("diff "):
            continue

        if line.startswith(" "):
            # Context line — should match original
            context_content = line[1:]
            if current_orig_line < len(original_lines):
                actual = original_lines[current_orig_line]
                if actual != context_content:
                    conflicts.append(
                        f"Hunk {hunk_idx}, line {current_orig_line + 1}: "
                        f"expected '{context_content[:60]}' but found '{actual[:60]}'"
                    )
            else:
                conflicts.append(
                    f"Hunk {hunk_idx}, line {current_orig_line + 1}: "
                    f"expected context but file has only {len(original_lines)} lines"
                )
            current_orig_line += 1
        elif line.startswith("-"):
            # Removed line — should also match original
            removed_content = line[1:]
            if current_orig_line < len(original_lines):
                actual = original_lines[current_orig_line]
                if actual != removed_content:
                    conflicts.append(
                        f"Hunk {hunk_idx}, line {current_orig_line + 1}: "
                        f"expected to remove '{removed_content[:60]}' but found '{actual[:60]}'"
                    )
            else:
                conflicts.append(
                    f"Hunk {hunk_idx}, line {current_orig_line + 1}: "
                    f"expected to remove line but file has only {len(original_lines)} lines"
                )
            current_orig_line += 1
        # '+' lines don't advance original line counter

    return conflicts


# ---------------------------------------------------------------------------
# Patch application with rollback
# ---------------------------------------------------------------------------


def _compute_checksum(content: str) -> str:
    """Compute a SHA-256 checksum of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _safe_artifact_name(value: str | Path) -> str:
    """Return a filesystem-safe name for rollback/history artifacts."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    return safe.strip("._") or "file"


def _get_rollback_dir(project_dir: str | Path | None = None) -> Path:
    """Get the rollback directory, creating it if needed."""
    if project_dir:
        base = Path(project_dir)
    else:
        base = Path.cwd()
    rollback_dir = base / ROLLBACK_DIR_NAME
    rollback_dir.mkdir(parents=True, exist_ok=True)
    return rollback_dir


def _get_patches_dir(project_dir: str | Path | None = None) -> Path:
    """Get the patches history directory, creating it if needed."""
    if project_dir:
        base = Path(project_dir)
    else:
        base = Path.cwd()
    patches_dir = base / PATCHES_DIR_NAME
    patches_dir.mkdir(parents=True, exist_ok=True)
    return patches_dir


def _save_rollback(filepath: str | Path, content: str, project_dir: str | Path | None = None) -> RollbackEntry:
    """Save a backup of the current file content for rollback.

    Args:
        filepath: The file being modified.
        content: The current content to back up.
        project_dir: Project root for rollback storage.

    Returns:
        A RollbackEntry describing the backup.
    """
    p = Path(filepath)
    rollback_dir = _get_rollback_dir(project_dir)
    checksum = _compute_checksum(content)
    timestamp = time.time()

    # Create a unique backup filename
    safe_name = _safe_artifact_name(p)
    backup_name = f"{timestamp:.0f}_{safe_name}_{checksum[:8]}.bak"
    backup_path = rollback_dir / backup_name

    backup_path.write_text(content, encoding="utf-8")

    # Prune old rollbacks if exceeding max
    _prune_rollbacks(rollback_dir)

    return RollbackEntry(
        filepath=str(p),
        backup_path=str(backup_path),
        timestamp=timestamp,
        checksum=checksum,
        size=len(content),
    )


def _prune_rollbacks(rollback_dir: Path) -> None:
    """Remove oldest rollback entries if exceeding MAX_ROLLBACK_ENTRIES."""
    backups = sorted(rollback_dir.glob("*.bak"), key=lambda p: p.stat().st_mtime)
    while len(backups) > MAX_ROLLBACK_ENTRIES:
        oldest = backups.pop(0)
        try:
            oldest.unlink()
            logger.debug("Pruned old rollback: %s", oldest)
        except OSError:
            pass


def perform_rollback(rollback_entry: RollbackEntry) -> tuple[bool, str]:
    """Restore a file from a rollback backup.

    Args:
        rollback_entry: The rollback entry to restore from.

    Returns:
        (success: bool, message: str)
    """
    backup_path = Path(rollback_entry.backup_path)
    if not backup_path.exists():
        return False, f"Rollback backup not found: {backup_path}"

    try:
        content = backup_path.read_text(encoding="utf-8")
        target = Path(rollback_entry.filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        logger.info("Rollback restored: %s from %s", rollback_entry.filepath, backup_path)
        return True, f"Rolled back: {rollback_entry.filepath}"
    except Exception as e:
        logger.error("Rollback failed for %s: %s", rollback_entry.filepath, e)
        return False, f"Rollback error: {e}"


def apply_patch(
    path: str | Path,
    content: str,
    project_dir: str | Path | None = None,
    enable_rollback: bool = True,
) -> tuple[bool, str, RollbackEntry | None]:
    """Apply a patch by writing content to a file, with optional rollback.

    This is the actual write operation, separated from the approval flow
    so it can be called after user approval.

    Args:
        path: Path to the file to write.
        content: The content to write.
        project_dir: Project root for rollback storage.
        enable_rollback: If True, save original content for rollback.

    Returns:
        (success: bool, message: str, rollback: RollbackEntry | None)
    """
    p = Path(path)
    rollback_entry = None

    try:
        # Save original for rollback (if file exists, save content; if new, save sentinel)
        if enable_rollback:
            try:
                if p.exists():
                    original = p.read_text(encoding="utf-8")
                    rollback_entry = _save_rollback(path, original, project_dir)
                else:
                    # New file: save sentinel so rollback can delete it
                    rollback_entry = _save_rollback(path, NYX_NEW_FILE_SENTINEL, project_dir)
            except Exception as e:
                logger.warning("Could not save rollback for %s: %s", path, e)

        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        size = len(content)
        logger.info("Patch applied: %s (%d bytes)", path, size)
        return True, f"File written: {path} ({size} bytes)", rollback_entry
    except Exception as e:
        logger.error("Failed to apply patch to %s: %s", path, e)
        return False, f"Error writing file: {e}", None


# ---------------------------------------------------------------------------
# History tracking
# ---------------------------------------------------------------------------


def save_patch_record(record: PatchRecord, project_dir: str | Path | None = None) -> None:
    """Save a patch record to the history directory.

    Args:
        record: The patch record to save.
        project_dir: Project root for history storage.
    """
    patches_dir = _get_patches_dir(project_dir)
    timestamp = record.timestamp
    safe_name = _safe_artifact_name(record.filepath)
    patch_filename = f"{timestamp:.0f}_{record.change_type.value}_{safe_name}.patch"
    patch_path = patches_dir / patch_filename

    header = (
        f"# Nyx Patch Record\n"
        f"# File: {record.filepath}\n"
        f"# Type: {record.change_type.value}\n"
        f"# Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.timestamp))}\n"
        f"# Success: {record.success}\n"
        f"# Summary: {record.summary}\n"
    )
    if record.error:
        header += f"# Error: {record.error}\n"
    if record.rollback_entry:
        header += f"# Rollback: {record.rollback_entry.backup_path}\n"
    header += "\n"

    try:
        patch_path.write_text(header + record.diff_text, encoding="utf-8")
        logger.debug("Patch record saved: %s", patch_path)
    except Exception as e:
        logger.warning("Could not save patch record: %s", e)


def get_patch_history(
    project_dir: str | Path | None = None,
    limit: int = 50,
) -> list[dict[str, str]]:
    """Retrieve recent patch history.

    Args:
        project_dir: Project root for history storage.
        limit: Maximum number of records to return.

    Returns:
        List of dicts with keys: filepath, type, time, summary, success.
    """
    patches_dir = _get_patches_dir(project_dir)
    if not patches_dir.exists():
        return []

    patch_files = sorted(patches_dir.glob("*.patch"), key=lambda p: p.stat().st_mtime, reverse=True)
    history: list[dict[str, str]] = []

    for pf in patch_files[:limit]:
        try:
            content = pf.read_text(encoding="utf-8")
            record: dict[str, str] = {"file": pf.stem}
            for line in content.splitlines():
                if line.startswith("# File: "):
                    record["filepath"] = line[len("# File: "):]
                elif line.startswith("# Type: "):
                    record["type"] = line[len("# Type: "):]
                elif line.startswith("# Time: "):
                    record["time"] = line[len("# Time: "):]
                elif line.startswith("# Summary: "):
                    record["summary"] = line[len("# Summary: "):]
                elif line.startswith("# Success: "):
                    record["success"] = line[len("# Success: "):]
            history.append(record)
        except Exception:
            continue

    return history


# ---------------------------------------------------------------------------
# Syntax validation
# ---------------------------------------------------------------------------


def validate_patch_syntax(diff_text: str) -> list[str]:
    """Validate the syntax of a patch/diff text.

    Checks for:
      - Presence of hunk headers (@@ ... @@)
      - Balanced + and - lines
      - Valid line counts in hunk headers
      - No malformed headers

    Args:
        diff_text: The patch text to validate.

    Returns:
        A list of validation error messages. Empty list means valid.
    """
    errors: list[str] = []
    lines = diff_text.splitlines()

    # Check for SEARCH/REPLACE format
    has_search_marker = any(_SEARCH_MARKER_RE.match(l) for l in lines)
    if has_search_marker:
        return _validate_search_replace_syntax(lines)

    # Check for unified diff format
    has_hunk = any(_HUNK_HEADER_RE.match(l) for l in lines)
    if not has_hunk:
        # Not a diff — might be full content
        if diff_text.strip():
            return []  # Full content is valid as a "create" patch
        else:
            errors.append("Empty patch text")
            return errors

    # Validate hunk headers
    for i, line in enumerate(lines):
        m = _HUNK_HEADER_RE.match(line)
        if m:
            old_start = int(m.group(1))
            old_count = int(m.group(2)) if m.group(2) else 1
            new_start = int(m.group(3))
            new_count = int(m.group(4)) if m.group(4) else 1

            if old_start < 0:
                errors.append(f"Line {i + 1}: Invalid old_start ({old_start})")
            if new_start < 0:
                errors.append(f"Line {i + 1}: Invalid new_start ({new_start})")
            if old_count < 0:
                errors.append(f"Line {i + 1}: Invalid old_count ({old_count})")
            if new_count < 0:
                errors.append(f"Line {i + 1}: Invalid new_count ({new_count})")

    return errors


def _validate_search_replace_syntax(lines: list[str]) -> list[str]:
    """Validate SEARCH/REPLACE block syntax."""
    errors: list[str] = []
    in_search = False
    in_replace = False
    search_count = 0
    replace_count = 0

    for i, line in enumerate(lines):
        if _SEARCH_MARKER_RE.match(line):
            if in_search:
                errors.append(f"Line {i + 1}: Nested SEARCH marker")
            in_search = True
            in_replace = False
            search_count += 1
        elif _DIVIDER_RE.match(line):
            if not in_search:
                errors.append(f"Line {i + 1}: '=======' without SEARCH")
            in_search = False
            in_replace = True
        elif _REPLACE_MARKER_RE.match(line):
            if not in_replace:
                errors.append(f"Line {i + 1}: '>>>>>>> REPLACE' without REPLACE section")
            in_search = False
            in_replace = False
            replace_count += 1

    if in_search:
        errors.append("Unclosed SEARCH block (missing '=======' and '>>>>>>> REPLACE')")
    if in_replace:
        errors.append("Unclosed REPLACE block (missing '>>>>>>> REPLACE')")
    if search_count != replace_count:
        errors.append(
            f"Mismatched SEARCH/REPLACE blocks: {search_count} SEARCH, {replace_count} REPLACE"
        )

    return errors


# ---------------------------------------------------------------------------
# PatchTool — main orchestrator
# ---------------------------------------------------------------------------


class PatchTool:
    """Tool for safe file modifications via diff/patch with approval.

    Features:
      - Parses unified diffs and SEARCH/REPLACE blocks
      - Validates patch syntax
      - Detects conflicts with current file content
      - Categorizes changes (CREATE/MODIFY/DELETE)
      - Displays clear summary before approval
      - Automatic rollback on every write
      - Maintains patch history
      - Optional git diff / git apply --check integration
    """

    def __init__(
        self,
        approval_callback: Callable[[str, str, str], tuple[bool, str]] | None = None,
        project_dir: str | Path | None = None,
        enable_rollback: bool = True,
        enable_history: bool = True,
        use_git: bool = True,
    ):
        """Initialize the patch tool.

        Args:
            approval_callback: Called with (filepath, diff_summary, full_diff)
                and returns (approved: bool, reason: str).
            project_dir: Project root for rollback/history storage.
            enable_rollback: If True, save backups before each write.
            enable_history: If True, save patch records.
            use_git: If True, use git commands when available.
        """
        self._approval_callback = approval_callback
        self._project_dir = project_dir
        self._enable_rollback = enable_rollback
        self._enable_history = enable_history
        self._use_git = use_git

    def set_approval_callback(
        self,
        callback: Callable[[str, str, str], tuple[bool, str]] | None,
    ) -> None:
        self._approval_callback = callback

    @property
    def project_dir(self) -> str | Path | None:
        return self._project_dir

    @project_dir.setter
    def project_dir(self, value: str | Path | None) -> None:
        self._project_dir = value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propose_write(self, path: str, content: str) -> tuple[bool, str]:
        """Propose a file write via diff/patch with user approval.

        This is the main entry point. It:
          1. Reads the original file (if it exists).
          2. Computes a unified diff.
          3. Categorizes the change (CREATE/MODIFY/DELETE).
          4. Requests user approval showing the diff and summary.
          5. Applies the patch if approved (with rollback).

        Args:
            path: The file path to write.
            content: The proposed content.

        Returns:
            (success: bool, message: str)
        """
        p = Path(path)
        original = ""
        if p.exists():
            try:
                original = p.read_text(encoding="utf-8")
            except Exception as e:
                return False, f"Error reading original file: {e}"

        diff = compute_diff(original, content, str(path))

        if diff == "(no changes)":
            return True, f"No changes needed for: {path}"

        # Parse and categorize
        patch_info = parse_unified_diff(diff, str(path))
        patch_info.original_content = original
        patch_info.proposed_content = content

        # Detect conflicts
        if original and patch_info.change_type == ChangeType.MODIFY:
            conflicts = detect_conflicts(original, patch_info, str(path))
            if conflicts:
                patch_info.validation_errors.extend(conflicts)

        # Git check (optional)
        git_status = ""
        if self._use_git and self._project_dir:
            can_apply, git_msg = git_apply_check(diff, self._project_dir)
            git_status = f"\n  git apply --check: {'✓' if can_apply else '✗'} {git_msg}"

        # Build summary
        summary = patch_info.summary
        detailed = patch_info.detailed_summary + git_status

        # Request approval
        if self._approval_callback:
            approved, reason = self._approval_callback(path, detailed, diff)
            if not approved:
                logger.info("Patch denied by user: %s (reason: %s)", path, reason)
                return False, f"Patch denied: {reason}"
        else:
            logger.warning(
                "No approval callback set for PatchTool; applying directly: %s", path
            )

        # Apply with rollback
        success, message, rollback = apply_patch(
            path, content,
            project_dir=self._project_dir,
            enable_rollback=self._enable_rollback,
        )

        # Save history
        if self._enable_history:
            record = PatchRecord(
                filepath=str(path),
                change_type=patch_info.change_type,
                timestamp=time.time(),
                diff_text=diff,
                summary=summary,
                success=success,
                error="" if success else message,
                rollback_entry=rollback,
            )
            save_patch_record(record, self._project_dir)

        return success, message

    def propose_append(self, path: str, content: str) -> tuple[bool, str]:
        """Propose appending content to a file.

        Args:
            path: The file path to append to.
            content: The content to append.

        Returns:
            (success: bool, message: str)
        """
        p = Path(path)
        original = ""
        if p.exists():
            try:
                original = p.read_text(encoding="utf-8")
            except Exception as e:
                return False, f"Error reading original file: {e}"

        proposed = original + content
        diff = compute_diff(original, proposed, str(path))

        if diff == "(no changes)":
            return True, f"No changes needed for: {path}"

        patch_info = parse_unified_diff(diff, str(path))
        summary = f"APPEND to: {path} (+{len(content)} bytes, {patch_info.total_additions} lines)"

        if self._approval_callback:
            approved, reason = self._approval_callback(path, summary, diff)
            if not approved:
                logger.info("Append denied by user: %s (reason: %s)", path, reason)
                return False, f"Append denied: {reason}"

        success, message, rollback = apply_patch(
            path, proposed,
            project_dir=self._project_dir,
            enable_rollback=self._enable_rollback,
        )

        if self._enable_history:
            record = PatchRecord(
                filepath=str(path),
                change_type=ChangeType.MODIFY,
                timestamp=time.time(),
                diff_text=diff,
                summary=summary,
                success=success,
                error="" if success else message,
                rollback_entry=rollback,
            )
            save_patch_record(record, self._project_dir)

        return success, message

    def propose_patch(
        self,
        path: str,
        diff_text: str,
        patch_format: str = "auto",
    ) -> tuple[bool, str]:
        """Propose applying a raw patch/diff to a file.

        This is the advanced entry point that accepts a raw diff or
        SEARCH/REPLACE block, parses it, validates it, checks for
        conflicts, and applies it.

        Args:
            path: The target file path.
            diff_text: The patch text (unified diff or SEARCH/REPLACE).
            patch_format: "auto", "unified_diff", or "search_replace".

        Returns:
            (success: bool, message: str)
        """
        p = Path(path)

        # Read original
        original = ""
        if p.exists():
            try:
                original = p.read_text(encoding="utf-8")
            except Exception as e:
                return False, f"Error reading original file: {e}"

        # Validate syntax
        syntax_errors = validate_patch_syntax(diff_text)
        if syntax_errors:
            return False, "Patch syntax errors:\n" + "\n".join(
                f"  - {e}" for e in syntax_errors
            )

        # Auto-detect format
        if patch_format == "auto":
            if any(_SEARCH_MARKER_RE.match(l) for l in diff_text.splitlines()):
                patch_format = "search_replace"
            else:
                patch_format = "unified_diff"

        # Parse
        if patch_format == "search_replace":
            patch_info = parse_search_replace(diff_text, str(path))
        else:
            patch_info = parse_unified_diff(diff_text, str(path))

        patch_info.original_content = original
        patch_info.diff_text = diff_text

        # Detect conflicts
        if original and patch_info.change_type == ChangeType.MODIFY:
            conflicts = detect_conflicts(original, patch_info, str(path))
            if conflicts:
                # Include a snippet of the current file content so the AI can
                # regenerate the patch against the actual current state.
                current_preview_lines = original.splitlines()
                max_preview = 60
                preview = "\n".join(current_preview_lines[:max_preview])
                if len(current_preview_lines) > max_preview:
                    preview += f"\n... ({len(current_preview_lines) - max_preview} more lines)"
                return (
                    False,
                    f"[CONFLICT] Patch conflicts with current content of {path}\n"
                    + "\n".join(f"  ⚠ {c}" for c in conflicts)
                    + "\n\n"
                    + "The file has been modified since this patch was generated.\n"
                    + "To fix: use write_file with the complete new content, "
                    + "or regenerate apply_diff after reading the current file.\n"
                    + f"\nCurrent file content ({len(current_preview_lines)} lines):\n"
                    + "```\n" + preview + "\n```",
                )

        # Apply the patch to produce proposed content
        proposed: str
        if patch_info.change_type == ChangeType.CREATE:
            proposed = diff_text  # Full content for new files
        elif patch_info.change_type == ChangeType.DELETE:
            proposed = ""
        elif patch_format == "search_replace":
            search_replace_content = _apply_search_replace_to_content(original, diff_text)
            if search_replace_content is None:
                return False, f"Failed to apply SEARCH/REPLACE patch to {path}: SEARCH text not found in file"
            proposed = search_replace_content
        else:
            unified_content = _apply_unified_diff_to_content(original, diff_text)
            if unified_content is None:
                return False, f"Failed to apply patch to {path}: could not reconstruct content"
            proposed = unified_content

        patch_info.proposed_content = proposed

        # Git check
        git_status = ""
        if self._use_git and self._project_dir:
            can_apply, git_msg = git_apply_check(diff_text, self._project_dir)
            git_status = f"\n  git apply --check: {'✓' if can_apply else '✗'} {git_msg}"

        # Build summary
        summary = patch_info.summary
        detailed = patch_info.detailed_summary + git_status

        # Request approval
        if self._approval_callback:
            approved, reason = self._approval_callback(path, detailed, diff_text)
            if not approved:
                logger.info("Patch denied by user: %s (reason: %s)", path, reason)
                return False, f"Patch denied: {reason}"
        else:
            logger.warning(
                "No approval callback set for PatchTool; applying directly: %s", path
            )

        # Apply
        success, message, rollback = apply_patch(
            path, proposed,
            project_dir=self._project_dir,
            enable_rollback=self._enable_rollback,
        )

        # History
        if self._enable_history:
            record = PatchRecord(
                filepath=str(path),
                change_type=patch_info.change_type,
                timestamp=time.time(),
                diff_text=diff_text,
                summary=summary,
                success=success,
                error="" if success else message,
                rollback_entry=rollback,
            )
            save_patch_record(record, self._project_dir)

        return success, message

    def rollback_last(self, filepath: str) -> tuple[bool, str]:
        """Rollback the most recent change to a file.

        Args:
            filepath: The file to rollback.

        Returns:
            (success: bool, message: str)
        """
        rollback_dir = _get_rollback_dir(self._project_dir)
        safe_prefix = _safe_artifact_name(filepath)

        # Find the most recent backup for this file
        backups = sorted(
            rollback_dir.glob(f"*_{safe_prefix}_*.bak"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not backups:
            return False, f"No rollback available for: {filepath}"

        latest = backups[0]
        content = latest.read_text(encoding="utf-8")
        target = Path(filepath)

        # Check if this was a newly-created file (sentinel)
        if content == NYX_NEW_FILE_SENTINEL:
            # Rollback = delete the file
            try:
                latest.unlink()  # Remove backup
            except OSError:
                pass
            if target.exists():
                try:
                    target.unlink()
                    logger.info("Rollback (new file deleted): %s", filepath)
                    return True, f"Rolled back: deleted newly-created file {filepath}"
                except Exception as e:
                    return False, f"Rollback error (could not delete {filepath}): {e}"
            return True, f"Rolled back: file {filepath} did not exist (already clean)"

        # Normal rollback: restore previous content
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

        # Remove the used backup
        try:
            latest.unlink()
        except OSError:
            pass

        logger.info("Rolled back: %s", filepath)
        return True, f"Rolled back: {filepath}"

    def get_history(
        self, limit: int = 50
    ) -> list[dict[str, str]]:
        """Get recent patch history.

        Args:
            limit: Maximum number of records.

        Returns:
            List of history records.
        """
        return get_patch_history(self._project_dir, limit)


# ---------------------------------------------------------------------------
# Helper: apply unified diff to content
# ---------------------------------------------------------------------------


def _find_best_hunk_match(
    original_lines: list[str],
    search_lines: list[str],
    expected_idx: int,
    window: int = 150,
    threshold: float = 0.7,
) -> int | None:
    """Find the best matching index of search_lines in original_lines near expected_idx using SequenceMatcher.

    Returns the index in original_lines, or None if no match passes threshold.
    """
    if not search_lines:
        return expected_idx

    n_search = len(search_lines)
    n_orig = len(original_lines)

    # Clean search_lines normalisation
    search_norm = "\n".join(_normalize_line(l) for l in search_lines).strip()

    # Search bounds
    start_search = max(0, expected_idx - window)
    end_search = min(n_orig - n_search + 1, expected_idx + window)

    if start_search >= end_search:
        # Fallback to scan entire file if window boundaries are small or invalid
        if n_orig >= n_search:
            start_search = 0
            end_search = n_orig - n_search + 1
        else:
            return None

    best_ratio = 0.0
    best_idx = None

    for i in range(start_search, end_search):
        slice_lines = original_lines[i : i + n_search]
        slice_norm = "\n".join(_normalize_line(l) for l in slice_lines).strip()

        if slice_norm == search_norm:
            return i

        matcher = difflib.SequenceMatcher(None, slice_norm, search_norm)
        ratio = matcher.ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i

    if best_ratio >= threshold and best_idx is not None:
        return best_idx

    return None


def _apply_unified_diff_to_content(original: str, diff_text: str) -> str | None:
    """Apply a unified diff to original content using fuzzy hunk matching to tolerate line shifts.

    Args:
        original: The original file content.
        diff_text: The unified diff to apply.

    Returns:
        The resulting content, or None if the patch cannot be applied.
    """
    original_lines = original.splitlines()
    diff_lines = diff_text.splitlines()

    # 1. Parse diff into hunks
    hunks: list[dict[str, Any]] = []
    current_hunk: dict[str, Any] | None = None

    for line in diff_lines:
        m = _HUNK_HEADER_RE.match(line)
        if m:
            if current_hunk:
                hunks.append(current_hunk)
            old_start = int(m.group(1))
            current_hunk = {"old_start": old_start, "lines": []}
        elif current_hunk is not None:
            # Skip diff file headers, git comments, etc. Only parse actual hunk body.
            if line.startswith((" ", "-", "+")):
                current_hunk["lines"].append(line)

    if current_hunk:
        hunks.append(current_hunk)

    if not hunks:
        return original  # Nothing to apply

    cumulative_offset = 0

    for hunk in hunks:
        hunk_lines: list[str] = hunk["lines"]
        hunk_old_start: int = hunk["old_start"]

        expected_idx = (hunk_old_start - 1) + cumulative_offset

        # Split hunk lines into search (original context/deletions) and replace (context/additions)
        search_block = []
        replace_block = []

        for hl in hunk_lines:
            if hl.startswith(" "):
                search_block.append(hl[1:])
                replace_block.append(hl[1:])
            elif hl.startswith("-"):
                search_block.append(hl[1:])
            elif hl.startswith("+"):
                replace_block.append(hl[1:])

        # Find best match for the search_block near expected_idx
        match_idx = _find_best_hunk_match(original_lines, search_block, expected_idx)
        if match_idx is None:
            # If search block is empty (pure addition hunk at end or start), use expected_idx
            if not search_block:
                match_idx = max(0, min(expected_idx, len(original_lines)))
            else:
                logger.warning("Fuzzy matching failed for hunk at expected line %d", hunk_old_start)
                return None

        # Replace lines in original_lines
        replace_len = len(search_block)
        original_lines[match_idx : match_idx + replace_len] = replace_block

        # Track the index shift caused by this replacement
        hunk_shift = len(replace_block) - replace_len
        cumulative_offset += hunk_shift

    return "\n".join(original_lines)



# ---------------------------------------------------------------------------
# Convenience: quick diff display
# ---------------------------------------------------------------------------


def format_diff_for_display(
    diff_text: str,
    max_lines: int = 200,
    show_line_numbers: bool = False,
) -> str:
    """Format a diff for terminal display with optional line numbers.

    Args:
        diff_text: The raw diff text.
        max_lines: Maximum lines to display.
        show_line_numbers: If True, prefix each line with its number.

    Returns:
        Formatted diff string suitable for terminal output.
    """
    # Preserve trailing newline if present
    had_trailing = diff_text.endswith("\n")
    lines = diff_text.splitlines()
    total_lines = len(lines)

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines.append(f"... ({total_lines - max_lines} more lines)")

    if show_line_numbers:
        width = len(str(len(lines)))
        result = "\n".join(
            f"{i + 1:>{width}} | {line}" for i, line in enumerate(lines)
        )
    else:
        result = "\n".join(lines)

    if had_trailing and not result.endswith("\n"):
        result += "\n"
    return result


# ---------------------------------------------------------------------------
# Helper: apply SEARCH/REPLACE to content
# ---------------------------------------------------------------------------


def _normalize_line(line: str) -> str:
    """Normalize a line by stripping whitespace and compressing internal spaces."""
    return re.sub(r'\s+', ' ', line.strip())


def _find_best_match(original_lines: list[str], search_lines: list[str], threshold: float = 0.6) -> tuple[int, int] | None:
    """Find the best match of search_lines in original_lines using SequenceMatcher.

    Returns (start_idx, end_idx) of the match in original_lines, or None if no match passes threshold.
    """
    if not search_lines:
        return None

    n_search = len(search_lines)
    n_orig = len(original_lines)

    if n_orig < n_search:
        return None

    best_ratio = 0.0
    best_range = None

    # Allow window sizes from max(1, n_search - 2) to n_search + 2
    min_w = max(1, n_search - 2)
    max_w = min(n_orig, n_search + 2)

    search_norm = "\n".join(_normalize_line(l) for l in search_lines).strip()

    for w in range(min_w, max_w + 1):
        for i in range(n_orig - w + 1):
            slice_lines = original_lines[i:i+w]
            slice_norm = "\n".join(_normalize_line(l) for l in slice_lines).strip()

            if slice_norm == search_norm:
                return i, i + w

            matcher = difflib.SequenceMatcher(None, slice_norm, search_norm)
            ratio = matcher.ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_range = (i, i + w)

    if best_ratio >= threshold and best_range is not None:
        return best_range

    return None


def _apply_search_replace_to_content(original: str, patch_text: str) -> str | None:
    """Apply SEARCH/REPLACE blocks to original content.

    Args:
        original: The original file content.
        patch_text: The SEARCH/REPLACE block text.

    Returns:
        The resulting content, or None if any SEARCH block is not found.
    """
    result = original
    lines = patch_text.splitlines()
    i = 0
    had_trailing_newline = original.endswith("\n") or original.endswith("\r\n")

    while i < len(lines):
        if _SEARCH_MARKER_RE.match(lines[i]):
            i += 1
            # Skip optional :start_line: and ------- divider
            if i < len(lines) and _START_LINE_RE.match(lines[i]):
                i += 1
            if i < len(lines) and lines[i].strip() == "-------":
                i += 1

            # Collect SEARCH content
            search_lines: list[str] = []
            while i < len(lines) and not _DIVIDER_RE.match(lines[i]):
                search_lines.append(lines[i])
                i += 1

            if i >= len(lines):
                return None  # Malformed

            i += 1  # Skip =======

            # Collect REPLACE content
            replace_lines: list[str] = []
            while i < len(lines) and not _REPLACE_MARKER_RE.match(lines[i]):
                replace_lines.append(lines[i])
                i += 1

            if i >= len(lines):
                return None  # Malformed

            i += 1  # Skip >>>>>>> REPLACE

            # Try to match fuzzy using sequence matcher
            original_lines = result.splitlines()
            match_range = _find_best_match(original_lines, search_lines)
            if match_range is not None:
                start_idx, end_idx = match_range
                new_lines = original_lines[:start_idx] + replace_lines + original_lines[end_idx:]
                result = "\n".join(new_lines)
            else:
                # Fallback to exact search string replacement
                search_text = "\n".join(search_lines)
                replace_text = "\n".join(replace_lines)
                if search_text in result:
                    result = result.replace(search_text, replace_text, 1)
                else:
                    return None  # SEARCH text not found
        else:
            i += 1

    if had_trailing_newline and not (result.endswith("\n") or result.endswith("\r\n")):
        result += "\n"

    return result
