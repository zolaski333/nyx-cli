"""Tests for Nyx diff_tool — patch parsing, validation, conflict detection, rollback, history."""
from __future__ import annotations

import tempfile
from pathlib import Path


from nyx.diff_tool import (
    # Core
    PatchTool,
    compute_diff,
    compute_diff_from_path,
    apply_patch,
    # Parsing
    parse_unified_diff,
    parse_search_replace,
    # Validation
    validate_patch_syntax,
    # Conflict detection
    detect_conflicts,
    # Change type
    ChangeType,
    PatchInfo,
    HunkInfo,
    # Rollback
    RollbackEntry,
    perform_rollback,
    # History
    PatchRecord,
    save_patch_record,
    get_patch_history,
    # Display
    format_diff_for_display,
    # Git integration
    git_diff,
    git_apply_check,
    _is_git_repo,
)


# ============================================================================
# Unified diff parsing
# ============================================================================


class TestParseUnifiedDiff:
    """Test parsing of unified diff format."""

    def test_parse_simple_modification(self):
        diff = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,4 @@
 line1
-line2
+line2_modified
 line3
+line4
"""
        info = parse_unified_diff(diff, "file.py")
        assert info.filepath == "file.py"
        assert info.change_type == ChangeType.MODIFY
        assert len(info.hunks) == 1
        assert info.total_additions == 2
        assert info.total_deletions == 1
        assert info.hunks[0].added_lines == 2
        assert info.hunks[0].removed_lines == 1

    def test_parse_new_file(self):
        diff = """--- /dev/null
+++ b/new_file.py
@@ -0,0 +1,3 @@
+line1
+line2
+line3
"""
        info = parse_unified_diff(diff, "new_file.py")
        assert info.change_type == ChangeType.MODIFY  # No "new file" marker
        assert info.total_additions == 3
        assert info.total_deletions == 0

    def test_parse_new_file_with_marker(self):
        diff = """new file mode 100644
--- /dev/null
+++ b/new_file.py
@@ -0,0 +1,3 @@
+line1
+line2
+line3
"""
        info = parse_unified_diff(diff, "new_file.py")
        assert info.is_new_file
        assert info.change_type == ChangeType.CREATE

    def test_parse_deleted_file(self):
        diff = """deleted file mode 100644
--- a/old_file.py
+++ /dev/null
@@ -1,3 +0,0 @@
-line1
-line2
-line3
"""
        info = parse_unified_diff(diff, "old_file.py")
        assert info.is_deletion
        assert info.change_type == ChangeType.DELETE

    def test_parse_multiple_hunks(self):
        diff = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 line1
-line2
+line2_new
 line3
@@ -10,3 +10,4 @@
 line10
 line11
 line12
+line13
"""
        info = parse_unified_diff(diff, "file.py")
        assert len(info.hunks) == 2
        assert info.total_additions == 2
        assert info.total_deletions == 1

    def test_parse_diff_git_header(self):
        diff = """diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -1,1 +1,1 @@
-old
+new
"""
        info = parse_unified_diff(diff)
        assert info.filepath == "src/main.py"

    def test_parse_no_changes(self):
        diff = "(no changes)"
        info = parse_unified_diff(diff, "file.py")
        assert info.change_type == ChangeType.MODIFY
        assert info.total_additions == 0
        assert info.total_deletions == 0

    def test_summary_create(self):
        info = PatchInfo(
            filepath="new.py",
            change_type=ChangeType.CREATE,
            proposed_content="hello world",
        )
        assert "CREATE" in info.summary
        assert "new.py" in info.summary

    def test_summary_modify(self):
        info = PatchInfo(
            filepath="mod.py",
            change_type=ChangeType.MODIFY,
            total_additions=5,
            total_deletions=3,
            hunks=[HunkInfo(1, 3, 1, 5, added_lines=5, removed_lines=3)],
        )
        assert "MODIFY" in info.summary
        assert "+5" in info.summary
        assert "-3" in info.summary

    def test_summary_delete(self):
        info = PatchInfo(
            filepath="del.py",
            change_type=ChangeType.DELETE,
            original_content="x" * 100,
        )
        assert "DELETE" in info.summary

    def test_detailed_summary(self):
        info = PatchInfo(
            filepath="test.py",
            change_type=ChangeType.MODIFY,
            total_additions=10,
            total_deletions=5,
            hunks=[
                HunkInfo(1, 3, 1, 5, added_lines=5, removed_lines=3),
                HunkInfo(10, 2, 12, 5, added_lines=5, removed_lines=2),
            ],
        )
        detailed = info.detailed_summary
        assert "MODIFY" in detailed
        assert "Hunks:    2" in detailed
        assert "+10" in detailed
        assert "-5" in detailed


# ============================================================================
# SEARCH/REPLACE parsing
# ============================================================================


class TestParseSearchReplace:
    """Test parsing of SEARCH/REPLACE blocks."""

    def test_parse_simple_block(self):
        text = """<<<<<<< SEARCH
old line
=======
new line
>>>>>>> REPLACE"""
        info = parse_search_replace(text, "file.py")
        assert info.filepath == "file.py"
        assert info.change_type == ChangeType.MODIFY
        assert len(info.hunks) == 1
        assert info.total_additions == 1
        assert info.total_deletions == 1

    def test_parse_with_start_line(self):
        text = """<<<<<<< SEARCH
:start_line:42
-------
old content
=======
new content
>>>>>>> REPLACE"""
        info = parse_search_replace(text, "file.py")
        assert len(info.hunks) == 1
        assert info.hunks[0].old_start == 42

    def test_parse_multiple_blocks(self):
        text = """<<<<<<< SEARCH
line1
=======
new1
>>>>>>> REPLACE
<<<<<<< SEARCH
line2
=======
new2
>>>>>>> REPLACE"""
        info = parse_search_replace(text, "file.py")
        assert len(info.hunks) == 2
        assert info.total_additions == 2
        assert info.total_deletions == 2

    def test_parse_missing_divider(self):
        text = """<<<<<<< SEARCH
old line
>>>>>>> REPLACE"""
        info = parse_search_replace(text, "file.py")
        assert not info.is_valid
        assert any("Missing" in e for e in info.validation_errors)

    def test_parse_missing_end_marker(self):
        text = """<<<<<<< SEARCH
old line
=======
new line"""
        info = parse_search_replace(text, "file.py")
        assert not info.is_valid
        assert any("Missing" in e for e in info.validation_errors)

    def test_parse_empty(self):
        info = parse_search_replace("", "file.py")
        assert not info.is_valid
        assert any("No valid" in e for e in info.validation_errors)


# ============================================================================
# Syntax validation
# ============================================================================


class TestValidatePatchSyntax:
    """Test patch syntax validation."""

    def test_valid_unified_diff(self):
        diff = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,4 @@
 context
-old
+new
+extra
 context
"""
        errors = validate_patch_syntax(diff)
        assert len(errors) == 0

    def test_valid_search_replace(self):
        text = """<<<<<<< SEARCH
old
=======
new
>>>>>>> REPLACE"""
        errors = validate_patch_syntax(text)
        assert len(errors) == 0

    def test_empty_patch(self):
        errors = validate_patch_syntax("")
        assert len(errors) == 1
        assert "Empty" in errors[0]

    def test_full_content_valid(self):
        """Full file content (not a diff) should be valid as a 'create' patch."""
        errors = validate_patch_syntax("line1\nline2\nline3")
        assert len(errors) == 0

    def test_invalid_hunk_header(self):
        diff = """--- a/file.py
+++ b/file.py
@@ -1,-3 +1,4 @@
 context
"""
        errors = validate_patch_syntax(diff)
        assert len(errors) > 0

    def test_unclosed_search_block(self):
        text = """<<<<<<< SEARCH
old
=======
new"""
        errors = validate_patch_syntax(text)
        assert len(errors) > 0
        assert any("Unclosed" in e for e in errors)

    def test_nested_search_marker(self):
        text = """<<<<<<< SEARCH
<<<<<<< SEARCH
old
=======
new
>>>>>>> REPLACE"""
        errors = validate_patch_syntax(text)
        assert len(errors) > 0
        assert any("Nested" in e for e in errors)

    def test_mismatched_blocks(self):
        text = """<<<<<<< SEARCH
old
=======
new
>>>>>>> REPLACE
<<<<<<< SEARCH
unclosed"""
        errors = validate_patch_syntax(text)
        assert len(errors) > 0
        assert any("Mismatched" in e for e in errors)


# ============================================================================
# Conflict detection
# ============================================================================


class TestDetectConflicts:
    """Test conflict detection between patches and file content."""

    def test_no_conflict_simple(self):
        original = "line1\nline2\nline3\n"
        diff = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 line1
-line2
+line2_new
 line3
"""
        info = parse_unified_diff(diff, "file.py")
        conflicts = detect_conflicts(original, info)
        assert len(conflicts) == 0

    def test_conflict_context_mismatch(self):
        original = "line1\nline2_CHANGED\nline3\n"
        diff = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 line1
-line2
+line2_new
 line3
"""
        info = parse_unified_diff(diff, "file.py")
        conflicts = detect_conflicts(original, info)
        assert len(conflicts) > 0
        assert any("line2_CHANGED" in c for c in conflicts)

    def test_no_conflict_new_file(self):
        info = PatchInfo(
            filepath="new.py",
            change_type=ChangeType.CREATE,
            is_new_file=True,
        )
        conflicts = detect_conflicts("", info)
        assert len(conflicts) == 0

    def test_no_conflict_delete(self):
        info = PatchInfo(
            filepath="old.py",
            change_type=ChangeType.DELETE,
            is_deletion=True,
        )
        conflicts = detect_conflicts("some content", info)
        assert len(conflicts) == 0

    def test_conflict_file_too_short(self):
        original = "line1\n"
        diff = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 line1
 line2
 line3
"""
        info = parse_unified_diff(diff, "file.py")
        conflicts = detect_conflicts(original, info)
        assert len(conflicts) > 0


# ============================================================================
# Diff computation
# ============================================================================


class TestComputeDiff:
    """Test diff computation."""

    def test_compute_diff_modification(self):
        original = "line1\nline2\nline3\n"
        proposed = "line1\nline2_modified\nline3\n"
        diff = compute_diff(original, proposed, "test.py")
        assert "@@" in diff
        assert "-line2" in diff
        assert "+line2_modified" in diff

    def test_compute_diff_new_file(self):
        diff = compute_diff("", "new content", "new.py")
        assert "new file" in diff

    def test_compute_diff_no_changes(self):
        diff = compute_diff("same", "same", "file.py")
        assert diff == "(no changes)"

    def test_compute_diff_from_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.txt"
            filepath.write_text("original\n")
            diff = compute_diff_from_path(str(filepath), "modified\n")
            assert "@@" in diff
            assert "-original" in diff
            assert "+modified" in diff


# ============================================================================
# Patch application with rollback
# ============================================================================


class TestApplyPatch:
    """Test patch application and rollback."""

    def test_apply_patch_new_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "new.txt"
            success, msg, rollback = apply_patch(
                str(filepath), "hello world", project_dir=tmpdir
            )
            assert success
            assert filepath.exists()
            assert filepath.read_text() == "hello world"
            # New files now get a sentinel rollback entry so they can be rolled back (deleted)
            from nyx.diff_tool import NYX_NEW_FILE_SENTINEL
            assert rollback is not None, "New files should have a sentinel rollback entry"
            sentinel_content = Path(rollback.backup_path).read_text(encoding="utf-8")
            assert sentinel_content == NYX_NEW_FILE_SENTINEL, "Backup should contain sentinel"


    def test_apply_patch_with_rollback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "existing.txt"
            filepath.write_text("original content")
            success, msg, rollback = apply_patch(
                str(filepath), "modified content", project_dir=tmpdir
            )
            assert success
            assert filepath.read_text() == "modified content"
            assert rollback is not None
            assert rollback.filepath == str(filepath)

            # Perform rollback
            ok, rb_msg = perform_rollback(rollback)
            assert ok
            assert filepath.read_text() == "original content"

    def test_apply_patch_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "deep" / "nested" / "file.txt"
            success, msg, _ = apply_patch(str(filepath), "content")
            assert success
            assert filepath.exists()

    def test_rollback_last(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "rollback_test.txt"
            filepath.write_text("v1")

            tool = PatchTool(project_dir=tmpdir)
            # Write v2
            tool.propose_write(str(filepath), "v2")
            assert filepath.read_text() == "v2"

            # Rollback
            ok, msg = tool.rollback_last(str(filepath))
            assert ok
            assert filepath.read_text() == "v1"

    def test_rollback_no_backup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = PatchTool(project_dir=tmpdir)
            ok, msg = tool.rollback_last("/nonexistent/file.txt")
            assert not ok
            assert "No rollback" in msg


# ============================================================================
# History tracking
# ============================================================================


class TestPatchHistory:
    """Test patch history recording and retrieval."""

    def test_save_and_retrieve_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record = PatchRecord(
                filepath="test.py",
                change_type=ChangeType.MODIFY,
                timestamp=1234567890.0,
                diff_text="--- a/test.py\n+++ b/test.py\n@@ -1 +1 @@\n-old\n+new\n",
                summary="MODIFY test.py (+1 -1)",
                success=True,
            )
            save_patch_record(record, tmpdir)

            history = get_patch_history(tmpdir)
            assert len(history) >= 1
            assert any("test.py" in str(h) for h in history)

    def test_history_with_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record = PatchRecord(
                filepath="fail.py",
                change_type=ChangeType.MODIFY,
                timestamp=1234567890.0,
                diff_text="bad diff",
                summary="MODIFY fail.py",
                success=False,
                error="Permission denied",
            )
            save_patch_record(record, tmpdir)

            history = get_patch_history(tmpdir)
            assert len(history) >= 1

    def test_empty_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history = get_patch_history(tmpdir)
            assert history == []

    def test_patch_tool_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "hist_test.txt"
            filepath.write_text("v1")

            tool = PatchTool(project_dir=tmpdir, enable_history=True)
            tool.propose_write(str(filepath), "v2")

            history = tool.get_history()
            assert len(history) >= 1


# ============================================================================
# PatchTool integration
# ============================================================================


class TestPatchTool:
    """Test the PatchTool orchestrator."""

    def test_propose_write_new_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "new.txt"
            tool = PatchTool(project_dir=tmpdir)
            success, msg = tool.propose_write(str(filepath), "hello")
            assert success
            assert filepath.exists()
            assert filepath.read_text() == "hello"

    def test_propose_write_modify(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "mod.txt"
            filepath.write_text("original")
            tool = PatchTool(project_dir=tmpdir)
            success, msg = tool.propose_write(str(filepath), "modified")
            assert success
            assert filepath.read_text() == "modified"

    def test_propose_write_no_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "same.txt"
            filepath.write_text("same")
            tool = PatchTool(project_dir=tmpdir)
            success, msg = tool.propose_write(str(filepath), "same")
            assert success
            assert "No changes" in msg

    def test_propose_write_denied(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "denied.txt"
            filepath.write_text("original")

            def deny(path, summary, diff):
                return False, "Not allowed"

            tool = PatchTool(approval_callback=deny, project_dir=tmpdir)
            success, msg = tool.propose_write(str(filepath), "modified")
            assert not success
            assert "denied" in msg.lower()
            assert filepath.read_text() == "original"  # Unchanged

    def test_propose_append(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "append.txt"
            filepath.write_text("line1\n")
            tool = PatchTool(project_dir=tmpdir)
            success, msg = tool.propose_append(str(filepath), "line2\n")
            assert success
            content = filepath.read_text()
            assert "line1" in content
            assert "line2" in content

    def test_propose_patch_unified_diff(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "patch_test.txt"
            filepath.write_text("line1\nline2\nline3\n")

            diff = """--- a/patch_test.txt
+++ b/patch_test.txt
@@ -1,3 +1,3 @@
 line1
-line2
+line2_modified
 line3
"""
            tool = PatchTool(project_dir=tmpdir)
            success, msg = tool.propose_patch(str(filepath), diff)
            assert success
            content = filepath.read_text()
            assert "line2_modified" in content
            assert "line2\n" not in content  # Original line2 removed

    def test_propose_patch_search_replace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "sr_test.txt"
            filepath.write_text("hello world\nfoo bar\n")

            patch = """<<<<<<< SEARCH
hello world
=======
bonjour le monde
>>>>>>> REPLACE"""
            tool = PatchTool(project_dir=tmpdir)
            success, msg = tool.propose_patch(str(filepath), patch)
            assert success
            content = filepath.read_text()
            assert "bonjour le monde" in content
            assert "hello world" not in content

    def test_propose_patch_syntax_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "bad.txt"
            filepath.write_text("content")

            tool = PatchTool(project_dir=tmpdir)
            success, msg = tool.propose_patch(str(filepath), "")
            assert not success
            assert "syntax" in msg.lower()

    def test_propose_patch_conflict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "conflict.txt"
            filepath.write_text("line1\nline2_CHANGED\nline3\n")

            diff = """--- a/conflict.txt
+++ b/conflict.txt
@@ -1,3 +1,3 @@
 line1
-line2
+line2_new
 line3
"""
            tool = PatchTool(project_dir=tmpdir)
            success, msg = tool.propose_patch(str(filepath), diff)
            assert not success
            assert "Conflict" in msg or "conflict" in msg.lower()

    def test_propose_patch_denied(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "denied_patch.txt"
            filepath.write_text("line1\nline2\nline3\n")

            diff = """--- a/denied_patch.txt
+++ b/denied_patch.txt
@@ -1,3 +1,3 @@
 line1
-line2
+line2_new
 line3
"""

            def deny(path, summary, diff_text):
                return False, "User rejected"

            tool = PatchTool(approval_callback=deny, project_dir=tmpdir)
            success, msg = tool.propose_patch(str(filepath), diff)
            assert not success
            assert "denied" in msg.lower()
            assert filepath.read_text() == "line1\nline2\nline3\n"


# ============================================================================
# Formatting
# ============================================================================


class TestFormatDiff:
    """Test diff formatting for display."""

    def test_format_basic(self):
        diff = "line1\nline2\nline3\n"
        formatted = format_diff_for_display(diff)
        assert formatted == diff

    def test_format_with_line_numbers(self):
        diff = "a\nb\nc\n"
        formatted = format_diff_for_display(diff, show_line_numbers=True)
        assert "1 | a" in formatted
        assert "2 | b" in formatted
        assert "3 | c" in formatted

    def test_format_truncation(self):
        diff = "\n".join(str(i) for i in range(300))
        formatted = format_diff_for_display(diff, max_lines=10)
        lines = formatted.splitlines()
        assert len(lines) <= 11  # 10 + truncation message
        assert "more lines" in lines[-1]


# ============================================================================
# Git integration
# ============================================================================


class TestGitIntegration:
    """Test git integration functions."""

    def test_is_git_repo_in_temp(self):
        """A temp directory is not a git repo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert not _is_git_repo(tmpdir)

    def test_is_git_repo_in_project(self):
        """The project directory should be a git repo."""
        # The nyx-cli project itself is likely a git repo
        project_root = Path(__file__).parent.parent
        if (project_root / ".git").exists():
            assert _is_git_repo(project_root)

    def test_git_diff_not_repo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = git_diff(Path(tmpdir) / "nonexistent.txt")
            assert result is None

    def test_git_apply_check_not_repo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            can_apply, msg = git_apply_check("fake diff", tmpdir)
            assert not can_apply
            assert "Not a git repository" in msg


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_original_content(self):
        diff = compute_diff("", "new content", "file.py")
        assert "new file" in diff

    def test_unicode_content(self):
        original = "héllo wörld\n"
        proposed = "héllo wörld modified\n"
        diff = compute_diff(original, proposed, "file.py")
        assert "@@" in diff

    def test_large_content(self):
        original = "x" * 10000
        proposed = "y" * 10000
        diff = compute_diff(original, proposed, "file.py")
        assert "@@" in diff

    def test_patch_info_validation_errors(self):
        info = PatchInfo(filepath="test.py", change_type=ChangeType.MODIFY)
        assert info.is_valid
        info.validation_errors.append("test error")
        assert not info.is_valid

    def test_rollback_entry_fields(self):
        entry = RollbackEntry(
            filepath="/tmp/test.txt",
            backup_path="/tmp/.nyx/rollback/backup.bak",
            timestamp=1234567890.0,
            checksum="abc123",
            size=100,
        )
        assert entry.filepath == "/tmp/test.txt"
        assert entry.size == 100

    def test_patch_record_fields(self):
        record = PatchRecord(
            filepath="test.py",
            change_type=ChangeType.CREATE,
            timestamp=1234567890.0,
            diff_text="diff",
            summary="CREATE test.py",
            success=True,
        )
        assert record.filepath == "test.py"
        assert record.change_type == ChangeType.CREATE
        assert record.success

    def test_fuzzy_find_best_match_whitespace_indentation(self):
        """_find_best_match should match even if indentation and whitespaces differ slightly."""
        from nyx.diff_tool import _find_best_match
        original = [
            "def my_func(a, b):",
            "    print('hello')",
            "    return a + b"
        ]
        
        # Mismatched indentation and spacing
        search = [
            "def   my_func(a,   b):",
            "  print('hello')",
            "     return a+b"
        ]
        
        match = _find_best_match(original, search)
        assert match == (0, 3)

    def test_anchor_matching_finds_shifted_block(self):
        """Anchor matching should locate a block far from the stale expected line."""
        from nyx.diff_tool import _find_best_hunk_match

        original = [f"prefix {i}" for i in range(250)]
        original += ["def target():", "    value = 1", "    return value"]
        original += [f"suffix {i}" for i in range(250)]

        match = _find_best_hunk_match(
            original,
            ["def target():", "    value = 1", "    return value"],
            expected_idx=5,
        )

        assert match == 250

    def test_propose_write_rejects_invalid_python_before_writing(self):
        """PatchTool should syntax-check generated Python before touching disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "bad.py"
            tool = PatchTool(project_dir=tmpdir)

            success, msg = tool.propose_write(str(filepath), "def broken(:\n    pass\n")

            assert not success
            assert "syntax" in msg.lower()
            assert not filepath.exists()
