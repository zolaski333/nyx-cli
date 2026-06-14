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


def auto_correct_loop(
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
