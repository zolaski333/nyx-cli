"""
Nyx — Test loop: auto-detect, run, parse failures, and auto-correct.

Provides:
- Test discovery (pytest, unittest, npm, etc.)
- Test execution with output capture
- Failure parsing (extract file, line, error message)
- Auto-correction loop: run → parse → fix → re-run
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

SNAPSHOT_EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    ".env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "dist",
    "build",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


def _python_executable() -> str:
    """Return a shell-safe path to the current Python interpreter."""
    exe = sys.executable or "python"
    if " " in exe:
        return f'"{exe}"'
    return exe


def _normalise_python_command(command: str) -> str:
    """Make common Python command prefixes portable across platforms."""
    stripped = command.lstrip()
    prefix_len = len(command) - len(stripped)
    prefix = command[:prefix_len]
    for old in ("python3 ", "python "):
        if stripped.startswith(old):
            return prefix + _python_executable() + stripped[len(old) - 1:]
    return command


@dataclass
class TestFailure:
    """A single test failure with location and message."""
    __test__ = False
    file: str = ""
    line: int = 0
    test_name: str = ""
    error_type: str = ""
    message: str = ""
    raw: str = ""

    def summary(self) -> str:
        parts = []
        if self.test_name:
            parts.append(self.test_name)
        if self.file:
            loc = f"{self.file}:{self.line}" if self.line else self.file
            parts.append(f"({loc})")
        if self.error_type:
            parts.append(f"[{self.error_type}]")
        if self.message:
            parts.append(self.message[:200])
        return " ".join(parts)


@dataclass
class TestResult:
    """Result of a test run."""
    __test__ = False
    success: bool
    command: str
    stdout: str = ""
    stderr: str = ""
    failures: list[TestFailure] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    total: int = 0
    duration_ms: float = 0.0
    raw_output: str = ""

    @property
    def summary(self) -> str:
        if self.success:
            return f"✅ All {self.total} tests passed ({self.duration_ms:.0f}ms)"
        return (
            f"❌ {self.failed}/{self.total} tests failed "
            f"({self.passed} passed, {self.duration_ms:.0f}ms)"
        )


# ---------------------------------------------------------------------------
# Test discovery
# ---------------------------------------------------------------------------


def discover_test_commands(root: str | Path | None = None) -> list[str]:
    """
    Discover available test commands for the project.
    Returns a list of shell commands to run tests.
    """
    root = Path(root).resolve() if root else Path.cwd().resolve()
    
    # Identify indicators
    has_package_json = (root / "package.json").exists()
    has_cargo = (root / "Cargo.toml").exists()
    has_go = (root / "go.mod").exists()
    
    # Python indicators
    pyproject = root / "pyproject.toml"
    setup_cfg = root / "setup.cfg"
    pytest_ini = root / "pytest.ini"
    has_python = (
        pyproject.exists()
        or setup_cfg.exists()
        or pytest_ini.exists()
        or (root / "requirements.txt").exists()
        or (root / "setup.py").exists()
        or any(root.glob("*.py"))
    )

    commands: list[str] = []
    py = _python_executable()

    # Priority 1: JS/TS
    if has_package_json:
        commands.append("npm test 2>&1")

    # Priority 2: Rust
    if has_cargo:
        commands.append("cargo test 2>&1")

    # Priority 3: Go
    if has_go:
        commands.append("go test ./... 2>&1")

    # Priority 4: Python configured tests
    if has_python:
        if pyproject.exists():
            content = pyproject.read_text(encoding="utf-8", errors="ignore")
            if "[tool.pytest" in content:
                commands.append(f"{py} -m pytest -v --tb=short 2>&1")
                commands.append(f"{py} -m pytest -v --tb=long 2>&1")
        if setup_cfg.exists():
            content = setup_cfg.read_text(encoding="utf-8", errors="ignore")
            if "[tool:pytest]" in content or "[pytest]" in content:
                commands.append(f"{py} -m pytest -v --tb=short 2>&1")
        if pytest_ini.exists():
            commands.append(f"{py} -m pytest -v --tb=short 2>&1")

        # General python test fallbacks
        commands.append(f"{py} -m pytest -v --tb=short 2>&1")
        commands.append(f"{py} -m unittest discover -v 2>&1")

    # Makefile
    makefile = root / "Makefile"
    if makefile.exists():
        content = makefile.read_text(encoding="utf-8", errors="ignore")
        if any(line.startswith("test") for line in content.splitlines()):
            commands.append("make test 2>&1")

    # Tox
    if (root / "tox.ini").exists():
        commands.append("tox 2>&1")

    # Final fallback if absolutely nothing was appended
    if not commands:
        commands.append(f"{py} -m pytest -v --tb=short 2>&1")
        commands.append(f"{py} -m unittest discover -v 2>&1")
        if (root / "package.json").exists() or not list(root.glob("*.py")):
            commands.append("npm test 2>&1")

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for cmd in commands:
        if cmd not in seen:
            seen.add(cmd)
            deduped.append(cmd)
    return deduped


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------


def run_tests(
    command: str | None = None,
    root: str | Path | None = None,
    timeout: int = 120,
) -> TestResult:
    """
    Run tests and return structured results.

    Note:
        This function uses shell=True to run test commands. This is acceptable
        for an experimental agentic CLI to allow flexible shell pipelines and
        composability of test runners, but means command inputs must be trusted.

    Args:
        command: Specific test command. If None, auto-discover.
        root: Project root directory.
        timeout: Timeout in seconds.

    Returns:
        TestResult with parsed failures.
    """
    root = Path(root).resolve() if root else Path.cwd().resolve()
    t_start = time.time()

    # Auto-discover if no command given
    if not command:
        candidates = discover_test_commands(root)
        if not candidates:
            return TestResult(
                success=False,
                command="",
                stdout="",
                stderr="",
                raw_output="No test commands discovered.",
                failures=[],
            )
        command = candidates[0]

    command = _normalise_python_command(command)

    try:
        from nyx.config import Config
        config = Config.load()
        if config.sandbox_use_docker:
            import shutil
            if shutil.which("docker") or shutil.which("podman"):
                docker_bin = "docker" if shutil.which("docker") else "podman"
                escaped = command.replace("'", "'\\''")
                quoted_cmd = f"'{escaped}'"
                escaped_root = str(root).replace("'", "'\\''")
                command = f"{docker_bin} run --rm -v '{escaped_root}':/workspace -w /workspace {config.sandbox_docker_image} sh -c {quoted_cmd}"
                logger.info("Running tests in Docker sandbox: %s", command)
    except Exception as e:
        logger.debug("Failed to check Docker config: %s", e)

    logger.info("Running tests: %s (in %s)", command, root)

    env = os.environ.copy()
    bin_dirs = []
    for venv_name in (".venv", "venv", "env", ".env"):
        venv_bin = Path(root) / venv_name / ("Scripts" if os.name == "nt" else "bin")
        if venv_bin.exists():
            bin_dirs.append(str(venv_bin))
            break
    node_bin = Path(root) / "node_modules" / ".bin"
    if node_bin.exists():
        bin_dirs.append(str(node_bin))
    if bin_dirs:
        env["PATH"] = os.pathsep.join(bin_dirs) + os.pathsep + env.get("PATH", "")

    try:
        # We run the command with shell=True to allow complex/composite commands
        # and shell features in test execution (e.g. environment variable interpolation, piping).
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(root),
            env=env,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        raw = stdout + "\n" + stderr
    except subprocess.TimeoutExpired:
        return TestResult(
            success=False,
            command=command,
            stdout="",
            stderr="",
            raw_output=f"Tests timed out after {timeout}s",
            failures=[TestFailure(test_name="(timeout)", message=f"Timed out after {timeout}s")],
            duration_ms=(time.time() - t_start) * 1000,
        )
    except Exception as e:
        return TestResult(
            success=False,
            command=command,
            stdout="",
            stderr=str(e),
            raw_output=str(e),
            failures=[TestFailure(test_name="(error)", message=str(e))],
            duration_ms=(time.time() - t_start) * 1000,
        )

    duration = (time.time() - t_start) * 1000

    # Parse results
    failures = _parse_failures(raw, root)
    passed, failed, total = _parse_counts(raw)

    success = proc.returncode == 0 and failed == 0

    return TestResult(
        success=success,
        command=command,
        stdout=stdout,
        stderr=stderr,
        failures=failures,
        passed=passed,
        failed=failed,
        total=total or (passed + failed),
        duration_ms=duration,
        raw_output=raw,
    )


# ---------------------------------------------------------------------------
# Failure parsing
# ---------------------------------------------------------------------------

# Regex patterns for common test frameworks
FAILURE_PATTERNS: list[re.Pattern] = [
    # Pytest:  FAILED tests/test_agent.py::test_name - AssertionError: message
    # Also:    FAILED test_fail.py::test_should_fail - assert (1 + 1) == 3
    re.compile(
        r"FAILED\s+"
        r"(?P<file>[^\s]+?)"
        r"(?:::?(?P<test_name>[^\s-]+))?"
        r"\s*-\s*"
        r"(?:(?P<error_type>\w+):\s*)?"
        r"(?P<message>.+)"
    ),
    # Pytest verbose:  test_name - file:line: error
    re.compile(
        r"(?P<test_name>[^\s]+)\s+"
        r"(?P<file>[^\s]+?):"
        r"(?P<line>\d+):\s*"
        r"(?P<error_type>\w+Error|\w+):\s*"
        r"(?P<message>.+)"
    ),
    # Pytest short:  test_name - AssertionError: message
    re.compile(
        r"(?P<test_name>[^\s]+)\s+-\s+"
        r"(?P<error_type>\w+):\s*"
        r"(?P<message>.+)"
    ),
    # Unittest:  ERROR: test_name (module.file) - message
    re.compile(
        r"(?:ERROR|FAIL):\s+"
        r"(?P<test_name>[^\s]+)"
        r"\s+\((?P<file>[^)]+)\)"
        r"(?:\s*-\s*(?P<message>.+))?"
    ),
    # Generic:  File ".../file.py", line N, in test_name
    re.compile(
        r'File\s+"(?P<file>[^"]+)",\s+line\s+(?P<line>\d+)'
        r'(?:,\s+in\s+(?P<test_name>\w+))?'
    ),
    # Generic error line:  ErrorType: message (at file:line)
    re.compile(
        r"(?P<error_type>\w+Error|\w+Exception):\s+"
        r"(?P<message>.+?)\s*"
        r"\(at\s+(?P<file>[^:]+):(?P<line>\d+)\)"
    ),
    # Go test:  --- FAIL: TestName (file_test.go:N)
    re.compile(
        r"---\s+FAIL:\s+(?P<test_name>\w+)"
        r"\s+\((?P<file>[^:]+):(?P<line>\d+)\)"
    ),
    # JavaScript:  FAIL test/file.test.js - test name
    re.compile(
        r"FAIL\s+(?P<file>[^\s]+)"
        r"(?:\s+-\s+(?P<test_name>.+))?"
    ),
]


def _parse_failures(raw_output: str, root: Path) -> list[TestFailure]:
    """Parse test failures from raw output."""
    failures: list[TestFailure] = []
    seen: set[str] = set()

    lines = raw_output.splitlines()

    for line in lines:
        for pattern in FAILURE_PATTERNS:
            m = pattern.search(line)
            if m:
                gd = m.groupdict()
                failure = TestFailure(
                    file=_clean_file_path(gd.get("file", "") or "", root),
                    line=int(gd["line"]) if gd.get("line") else 0,
                    test_name=gd.get("test_name", "") or "",
                    error_type=gd.get("error_type", "") or "",
                    message=gd.get("message", "") or "",
                    raw=line.strip(),
                )
                # Deduplicate
                key = f"{failure.file}:{failure.line}:{failure.test_name}"
                if key not in seen:
                    seen.add(key)
                    failures.append(failure)
                break

    return failures


def _clean_file_path(file_path: str, root: Path) -> str:
    """Clean and relativize a file path from test output."""
    # Remove quotes and whitespace
    file_path = file_path.strip().strip("\"'")
    # Try to make relative to root
    try:
        p = Path(file_path)
        if p.is_absolute():
            try:
                return str(p.relative_to(root))
            except ValueError:
                return file_path
        return file_path
    except Exception:
        return file_path


def _parse_counts(raw_output: str) -> tuple[int, int, int]:
    """Parse passed/failed/total counts from test output."""
    passed = 0
    failed = 0
    total = 0

    # Pytest summary (handle both orderings):
    #   "3 passed, 2 failed in 0.45s"
    #   "1 failed, 1 passed in 0.01s"
    # Try failed-first pattern first to avoid partial matches
    m = re.search(
        r"(?P<failed>\d+)\s+failed,\s+(?P<passed>\d+)\s+passed",
        raw_output,
    )
    if m:
        failed = int(m.group("failed"))
        passed = int(m.group("passed"))
        total = passed + failed
        return passed, failed, total

    # Pytest summary:  X passed, Y failed
    m = re.search(
        r"(?P<passed>\d+)\s+passed"
        r"(?:,\s+(?P<failed>\d+)\s+failed)?"
        r"(?:,\s+(?P<total>\d+)\s+total)?",
        raw_output,
    )
    if m:
        passed = int(m.group("passed"))
        failed = int(m.group("failed") or 0)
        total = int(m.group("total") or 0) or (passed + failed)
        return passed, failed, total

    # Unittest:  Ran N tests in 0.45s - FAILED (failures=M)
    m = re.search(r"Ran\s+(?P<total>\d+)\s+tests?", raw_output)
    if m:
        total = int(m.group("total"))
        failed_m = re.search(r"FAILED\s+\(failures=(?P<failed>\d+)", raw_output)
        if failed_m:
            failed = int(failed_m.group("failed"))
        passed = total - failed
        return passed, failed, total

    # Go test:  ok pkg  0.123s  or  FAIL pkg  0.123s
    m = re.search(r"^(?:ok|FAIL)\s+\S+\s+[\d.]+s", raw_output, re.MULTILINE)
    if m:
        # Count individual test results
        passed = len(re.findall(r"---\s+PASS:", raw_output))
        failed = len(re.findall(r"---\s+FAIL:", raw_output))
        total = passed + failed
        return passed, failed, total

    return passed, failed, total


# ---------------------------------------------------------------------------
# Auto-correction loop
# ---------------------------------------------------------------------------


@dataclass
class CorrectionResult:
    """Result of a test correction cycle."""
    success: bool = False
    iterations: int = 0
    final_result: TestResult | None = None
    corrections: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    attempts: list["CorrectionAttempt"] = field(default_factory=list)
    rolled_back: bool = False


@dataclass
class CorrectionAttempt:
    """Structured metadata for one correction attempt."""
    iteration: int
    correction: str = ""
    changed_files: list[str] = field(default_factory=list)
    before_score: int = 0
    after_score: int = 0
    before_signature: str = ""
    after_signature: str = ""
    rolled_back: bool = False
    stalled: bool = False


@dataclass
class _FileMetadata:
    size: int
    mtime_ns: int
    partial_hash: str | None = None


def _get_partial_hash(path: Path, size: int) -> str | None:
    try:
        with open(path, "rb") as f:
            if size <= 2048:
                content = f.read()
            else:
                head = f.read(1024)
                f.seek(max(0, size - 1024))
                tail = f.read(1024)
                content = head + tail
            import hashlib
            return hashlib.sha256(content).hexdigest()
    except OSError:
        return None


@dataclass
class _WorkspaceSnapshot:
    root: Path
    files: dict[str, bytes | None]
    skipped_files: dict[str, _FileMetadata] = field(default_factory=dict)
    truncated: bool = False

    @classmethod
    def capture(
        cls,
        root: Path,
        *,
        max_files: int = 2500,
        max_total_bytes: int = 25_000_000,
        max_file_bytes: int = 2_000_000,
    ) -> "_WorkspaceSnapshot":
        files: dict[str, bytes | None] = {}
        skipped_files: dict[str, _FileMetadata] = {}
        total_bytes = 0
        truncated = False
        for path in _iter_snapshot_files(root):
            rel = path.relative_to(root).as_posix()
            try:
                stat_res = path.stat()
                size = stat_res.st_size
                mtime_ns = stat_res.st_mtime_ns
            except OSError:
                skipped_files[rel] = _FileMetadata(size=-1, mtime_ns=-1)
                continue

            if len(files) >= max_files or total_bytes + size > max_total_bytes or size > max_file_bytes:
                truncated = True
                partial_hash = _get_partial_hash(path, size)
                skipped_files[rel] = _FileMetadata(size=size, mtime_ns=mtime_ns, partial_hash=partial_hash)
                continue

            try:
                files[rel] = path.read_bytes()
                total_bytes += size
            except OSError:
                partial_hash = _get_partial_hash(path, size)
                skipped_files[rel] = _FileMetadata(size=size, mtime_ns=mtime_ns, partial_hash=partial_hash)
                continue
        return cls(root=root, files=files, skipped_files=skipped_files, truncated=truncated)

    def changed_files(self, other: "_WorkspaceSnapshot") -> list[str]:
        all_names = (
            set(self.files)
            | set(self.skipped_files)
            | set(other.files)
            | set(other.skipped_files)
        )
        changed = []
        for name in all_names:
            in_self_files = name in self.files
            in_self_skipped = name in self.skipped_files
            in_other_files = name in other.files
            in_other_skipped = name in other.skipped_files

            present_self = in_self_files or in_self_skipped
            present_other = in_other_files or in_other_skipped
            if present_self != present_other:
                changed.append(name)
                continue

            if in_self_files and in_other_files:
                if self.files[name] != other.files[name]:
                    changed.append(name)
            elif in_self_skipped and in_other_skipped:
                if self.skipped_files[name] != other.skipped_files[name]:
                    changed.append(name)
            else:
                # One is in files, other is in skipped_files. Definitely changed.
                changed.append(name)

        return sorted(changed)

    def restore(self, changed_files: list[str]) -> list[str]:
        restored: list[str] = []
        for rel in changed_files:
            target = (self.root / rel).resolve()
            try:
                target.relative_to(self.root)
            except ValueError:
                continue
            if rel in self.skipped_files:
                logger.warning("Skipping restore of %s because it was not captured in the snapshot.", rel)
                continue
            original = self.files.get(rel)
            try:
                if original is None:
                    if target.exists() and target.is_file():
                        target.unlink()
                        _remove_empty_parents(target.parent, self.root)
                        restored.append(rel)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(original)
                    restored.append(rel)
            except OSError:
                logger.warning("Failed to restore %s", target)
        return restored


def _iter_snapshot_files(root: Path):
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SNAPSHOT_EXCLUDED_DIRS]
        current_path = Path(current)
        if ".nyx" in current_path.parts:
            # Runtime audit/patch artifacts are not part of source repair quality.
            continue
        for filename in files:
            path = current_path / filename
            if path.is_file():
                yield path


def _remove_empty_parents(path: Path, root: Path) -> None:
    while path != root:
        try:
            path.rmdir()
        except OSError:
            return
        path = path.parent


def _failure_signature(test_result: TestResult) -> str:
    if test_result.success:
        return "success"
    if test_result.failures:
        parts = [
            f"{f.file}:{f.line}:{f.test_name}:{f.error_type}:{f.message[:120]}"
            for f in test_result.failures[:20]
        ]
        return "|".join(parts)
    return (test_result.raw_output or "")[-1000:]


def _failure_score(test_result: TestResult) -> int:
    if test_result.success:
        return 0
    if test_result.failed:
        return test_result.failed * 1000 + max(0, len(test_result.failures))
    if test_result.failures:
        return len(test_result.failures) * 1000
    return 999_999


def _auto_correct_loop_legacy(
    fix_function: Callable,
    root: str | Path | None = None,
    test_command: str | None = None,
    max_iterations: int = 5,
    timeout: int = 120,
) -> CorrectionResult:
    """
    Run a test → fix → re-run loop until all tests pass or max iterations reached.

    Args:
        fix_function: A callable that receives (failures, raw_output, [history]) and
                      returns a description of what was fixed (empty string = no fix).
        root: Project root directory.
        test_command: Specific test command. If None, auto-discover.
        max_iterations: Maximum number of fix iterations.
        timeout: Timeout per test run in seconds.

    Returns:
        CorrectionResult with the outcome.
    """
    root = Path(root).resolve() if root else Path.cwd().resolve()
    result = CorrectionResult()
    history: list[dict] = []

    for iteration in range(1, max_iterations + 1):
        logger.info("Test iteration %d/%d", iteration, max_iterations)

        # Run tests
        test_result = run_tests(command=test_command, root=root, timeout=timeout)
        result.iterations = iteration
        result.final_result = test_result

        if test_result.success:
            result.success = True
            logger.info("All tests passed after %d iterations!", iteration)
            return result

        # Record failure history from previous iteration if corrections were applied
        if iteration > 1 and result.corrections:
            prev_correction = result.corrections[-1]
            failure_summaries = [f.summary() for f in test_result.failures]
            history.append({
                "iteration": iteration - 1,
                "correction": prev_correction,
                "failure_summary": "; ".join(failure_summaries) if failure_summaries else "Unknown test failure"
            })

        if not test_result.failures:
            result.errors.append(
                f"Iteration {iteration}: Tests failed but no parseable failures found."
            )
            # Try raw output anyway
            try:
                fix_desc = fix_function([], test_result.raw_output, history)
            except TypeError:
                fix_desc = fix_function([], test_result.raw_output)

            if fix_desc:
                result.corrections.append(f"Iteration {iteration}: {fix_desc}")
            else:
                break
            continue

        # Call fix function with failures, raw_output, and history
        try:
            fix_desc = fix_function(test_result.failures, test_result.raw_output, history)
        except TypeError:
            fix_desc = fix_function(test_result.failures, test_result.raw_output)

        if fix_desc:
            result.corrections.append(f"Iteration {iteration}: {fix_desc}")
        else:
            result.errors.append(
                f"Iteration {iteration}: Fix function returned no changes."
            )
            break

    # Max iterations reached or no fix possible
    if not result.success:
        result.errors.append(
            f"Max iterations ({max_iterations}) reached or fix stalled."
        )

    return result


def auto_correct_loop(
    fix_function: Callable,
    root: str | Path | None = None,
    test_command: str | None = None,
    max_iterations: int = 5,
    timeout: int = 120,
    fix_timeout: int | None = None,
    require_changes: bool = False,
    rollback_on_regression: bool = True,
    stop_on_stall: bool = False,
) -> CorrectionResult:
    """
    Run a test -> fix -> re-run loop until all tests pass or max iterations is reached.

    The loop records structured attempts, detects whether files changed, measures
    before/after failure progress, and can restore changed files when a fix makes
    the test result worse.
    """
    root = Path(root).resolve() if root else Path.cwd().resolve()
    result = CorrectionResult()
    history: list[dict] = []
    seen_signatures: set[str] = set()

    for iteration in range(1, max_iterations + 1):
        logger.info("Test iteration %d/%d", iteration, max_iterations)

        test_result = run_tests(command=test_command, root=root, timeout=timeout)
        result.iterations = iteration
        result.final_result = test_result

        if test_result.success:
            result.success = True
            logger.info("All tests passed after %d iterations!", iteration)
            return result

        before_signature = _failure_signature(test_result)
        before_score = _failure_score(test_result)
        before_snapshot = _WorkspaceSnapshot.capture(root)

        if not test_result.failures:
            result.errors.append(
                f"Iteration {iteration}: Tests failed but no parseable failures found."
            )

        fix_desc = _call_fix_function(
            fix_function,
            test_result.failures,
            test_result.raw_output,
            history,
            fix_timeout,
        )

        if not fix_desc:
            result.errors.append(
                f"Iteration {iteration}: Fix function returned no changes."
            )
            break

        after_snapshot = _WorkspaceSnapshot.capture(root)
        changed_files = before_snapshot.changed_files(after_snapshot)
        after_result = run_tests(command=test_command, root=root, timeout=timeout)
        result.final_result = after_result
        after_signature = _failure_signature(after_result)
        after_score = _failure_score(after_result)
        stalled = after_signature == before_signature and after_score >= before_score

        attempt = CorrectionAttempt(
            iteration=iteration,
            correction=fix_desc,
            changed_files=changed_files,
            before_score=before_score,
            after_score=after_score,
            before_signature=before_signature,
            after_signature=after_signature,
            stalled=stalled,
        )

        if require_changes and not changed_files:
            result.errors.append(
                f"Iteration {iteration}: Fix reported changes but no source files changed."
            )
            attempt.stalled = True
            result.attempts.append(attempt)
            if stop_on_stall:
                break
            continue

        if rollback_on_regression and changed_files and after_score > before_score:
            has_uncaptured = any(
                rel in before_snapshot.skipped_files or rel in after_snapshot.skipped_files
                for rel in changed_files
            )
            if has_uncaptured:
                logger.warning(
                    "Regression detected, but automatic rollback refused because some changed files were not captured in the snapshot."
                )
                result.errors.append(
                    f"Iteration {iteration}: Regression detected, but automatic rollback refused because some changed files were not captured in the snapshot."
                )
            else:
                restored = before_snapshot.restore(changed_files)
                attempt.rolled_back = bool(restored)
                result.rolled_back = result.rolled_back or attempt.rolled_back
                result.errors.append(
                    f"Iteration {iteration}: Correction regressed tests; rolled back {len(restored)} file(s)."
                )
                result.final_result = run_tests(command=test_command, root=root, timeout=timeout)
            result.attempts.append(attempt)
            break

        result.corrections.append(
            f"Iteration {iteration}: {fix_desc}"
            + (f" ({len(changed_files)} file(s) changed)" if changed_files else "")
        )
        result.attempts.append(attempt)

        failure_summaries = [f.summary() for f in after_result.failures]
        history.append({
            "iteration": iteration,
            "correction": fix_desc,
            "failure_summary": "; ".join(failure_summaries) if failure_summaries else "All tests passed",
            "changed_files": changed_files,
            "before_score": before_score,
            "after_score": after_score,
        })

        if after_result.success:
            result.success = True
            logger.info("All tests passed after %d iterations!", iteration)
            return result

        repeated = after_signature in seen_signatures
        seen_signatures.add(before_signature)
        seen_signatures.add(after_signature)
        if stop_on_stall and (stalled or repeated):
            result.errors.append(
                f"Iteration {iteration}: Correction stalled with the same failure signature."
            )
            break

    if not result.success:
        result.errors.append(
            f"Max iterations ({max_iterations}) reached or fix stalled."
        )

    return result


def _call_fix_function_inner(
    fix_function: Callable,
    failures: list[TestFailure],
    raw_output: str,
    history: list[dict],
    fix_timeout: int | None,
) -> str:
    import inspect

    try:
        sig = inspect.signature(fix_function)
        has_var_positional = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
        pos_params = [
            p for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        max_pos = len(pos_params)
    except (ValueError, TypeError):
        has_var_positional = True
        max_pos = 4

    if has_var_positional or max_pos >= 4:
        try:
            return str(fix_function(failures, raw_output, history, fix_timeout))
        except TypeError as e:
            if e.__traceback__ and e.__traceback__.tb_next is not None:
                raise

    if has_var_positional or max_pos >= 3:
        try:
            return str(fix_function(failures, raw_output, history))
        except TypeError as e:
            if e.__traceback__ and e.__traceback__.tb_next is not None:
                raise

    return str(fix_function(failures, raw_output))


def _multiprocess_fix_worker(
    queue,
    fix_function: Callable,
    failures: list[TestFailure],
    raw_output: str,
    history: list[dict],
    fix_timeout: int | None,
) -> None:
    try:
        res = _call_fix_function_inner(fix_function, failures, raw_output, history, fix_timeout)
        queue.put({"ok": True, "result": res})
    except BaseException as e:
        queue.put({"ok": False, "error": e})


def _call_fix_function(
    fix_function: Callable,
    failures: list[TestFailure],
    raw_output: str,
    history: list[dict],
    fix_timeout: int | None,
) -> str:
    if fix_timeout is not None and fix_timeout > 0:
        import pickle
        is_picklable = False
        try:
            pickle.dumps(fix_function)
            is_picklable = True
        except Exception:
            pass

        if is_picklable:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            queue = ctx.Queue()
            p = ctx.Process(
                target=_multiprocess_fix_worker,
                args=(queue, fix_function, failures, raw_output, history, fix_timeout),
            )
            p.start()
            p.join(timeout=fix_timeout)
            if p.is_alive():
                logger.warning("Fix function timed out after %s seconds and was terminated.", fix_timeout)
                p.terminate()
                p.join(timeout=1)
                if p.is_alive():
                    p.kill()
                return ""
            try:
                val = queue.get_nowait()
                if val["ok"]:
                    return val["result"]
                else:
                    raise val["error"]
            except Exception as e:
                logger.error("Error running fix function in subprocess: %s", e)
                return ""
        else:
            logger.warning(
                "Fix function is not picklable. Enforcing synchronous execution without hard timeout to prevent thread leak."
            )
            return _call_fix_function_inner(fix_function, failures, raw_output, history, fix_timeout)
    else:
        return _call_fix_function_inner(fix_function, failures, raw_output, history, fix_timeout)


def format_failures_for_llm(failures: list[TestFailure], raw_output: str) -> str:
    """
    Format test failures into a prompt-friendly string for an LLM to fix.
    """
    parts: list[str] = []
    parts.append("The following test failures were detected:\n")

    for i, f in enumerate(failures, 1):
        parts.append(f"--- Failure #{i} ---")
        if f.test_name:
            parts.append(f"  Test: {f.test_name}")
        if f.file:
            parts.append(f"  File: {f.file}:{f.line}" if f.line else f"  File: {f.file}")
        if f.error_type:
            parts.append(f"  Error: {f.error_type}")
        if f.message:
            parts.append(f"  Message: {f.message[:300]}")
        parts.append("")

    # Include relevant snippets from raw output
    parts.append("--- Raw output (last 30 lines) ---")
    lines = raw_output.splitlines()
    tail = lines[-30:] if len(lines) > 30 else lines
    parts.extend(tail)

    parts.append("\n---")
    parts.append("Please fix the code to resolve these test failures.")
    return "\n".join(parts)
