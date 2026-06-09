"""
Nyx — Patch/diff tool for safe file modifications.

Replaces the raw write_file approach with a structured patch/diff workflow:
  1. Read the original file (or confirm it's new).
  2. Compute a unified diff between original and proposed content.
  3. Present the diff to the user for approval.
  4. Apply the patch only after approval.

This prevents accidental data loss and gives the user full visibility
into every file change before it happens.
"""
from __future__ import annotations

import difflib
import logging
import os
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


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
# Patch application
# ---------------------------------------------------------------------------


def apply_patch(path: str | Path, content: str) -> tuple[bool, str]:
    """Apply a patch by writing content to a file.

    This is the actual write operation, separated from the approval flow
    so it can be called after user approval.

    Args:
        path: Path to the file to write.
        content: The content to write.

    Returns:
        (success: bool, message: str)
    """
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        size = len(content)
        logger.info("Patch applied: %s (%d bytes)", path, size)
        return True, f"File written: {path} ({size} bytes)"
    except Exception as e:
        logger.error("Failed to apply patch to %s: %s", path, e)
        return False, f"Error writing file: {e}"


# ---------------------------------------------------------------------------
# Patch tool
# ---------------------------------------------------------------------------


class PatchTool:
    """Tool for safe file modifications via diff/patch with approval."""

    def __init__(
        self,
        approval_callback: Callable[[str, str, str], tuple[bool, str]] | None = None,
    ):
        """Initialize the patch tool.

        Args:
            approval_callback: Called with (filepath, diff_summary, full_diff)
                and returns (approved: bool, reason: str).
        """
        self._approval_callback = approval_callback

    def set_approval_callback(
        self,
        callback: Callable[[str, str, str], tuple[bool, str]] | None,
    ) -> None:
        self._approval_callback = callback

    def propose_write(self, path: str, content: str) -> tuple[bool, str]:
        """Propose a file write via diff/patch with user approval.

        This is the main entry point. It:
          1. Reads the original file (if it exists).
          2. Computes a unified diff.
          3. Requests user approval showing the diff.
          4. Applies the patch if approved.

        Args:
            path: The file path to write.
            content: The proposed content.

        Returns:
            (success: bool, message: str)
        """
        # Compute diff
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

        # Generate a summary
        if not original:
            summary = f"NEW FILE: {path} ({len(content)} bytes)"
        else:
            added = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
            removed = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))
            summary = f"MODIFY: {path} ({added} insertions, {removed} deletions)"

        # Request approval
        if self._approval_callback:
            approved, reason = self._approval_callback(path, summary, diff)
            if not approved:
                logger.info("Patch denied by user: %s (reason: %s)", path, reason)
                return False, f"Patch denied: {reason}"
        else:
            # No approval callback — apply directly (fallback for non-interactive mode)
            logger.warning("No approval callback set for PatchTool; applying directly: %s", path)

        # Apply the patch
        return apply_patch(path, content)

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

        summary = f"APPEND to: {path} ({len(content)} bytes added)"

        if self._approval_callback:
            approved, reason = self._approval_callback(path, summary, diff)
            if not approved:
                logger.info("Append denied by user: %s (reason: %s)", path, reason)
                return False, f"Append denied: {reason}"

        return apply_patch(path, proposed)