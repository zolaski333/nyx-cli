"""
Nyx — Repository map tool.

Provides a structured overview of the current repository:
- Directory tree (top-level + key subdirectories)
- Important files (config, manifest, CI, etc.)
- Git status (branch, changes, last commit)
- Available test suites and commands
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_git(args: list[str], cwd: str | Path | None = None) -> str:
    """Run a git command and return stdout (or empty string on failure)."""
    try:
        proc = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=cwd or ".",
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""


def _run_command(cmd: list[str], cwd: str | Path | None = None) -> str:
    """Run an arbitrary command and return stdout (or empty string on failure)."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=cwd or ".")
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""


# ---------------------------------------------------------------------------
# RepoMap
# ---------------------------------------------------------------------------

IMPORTANT_FILE_PATTERNS: list[str] = [
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "Cargo.toml",
    "package.json",
    "go.mod",
    "Gemfile",
    "composer.json",
    "Makefile",
    "CMakeLists.txt",
    "Dockerfile",
    "docker-compose.yml",
    ".env.example",
    "config.json",
    "config.example.json",
    ".gitignore",
    ".github/workflows/*.yml",
    ".github/workflows/*.yaml",
    "README.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "requirements.txt",
    "Pipfile",
    "poetry.lock",
    "tsconfig.json",
    "webpack.config.js",
    "vite.config.ts",
    "next.config.js",
    "ruff.toml",
    ".editorconfig",
    ".pre-commit-config.yaml",
    "renovate.json",
    ".gitlab-ci.yml",
    "Jenkinsfile",
]

TEST_DIR_PATTERNS: list[str] = [
    "tests",
    "test",
    "spec",
    "__tests__",
]

TEST_FILE_PATTERNS: list[str] = [
    "test_*.py",
    "*_test.py",
    "*.test.js",
    "*.test.ts",
    "*.spec.js",
    "*.spec.ts",
    "*_test.go",
    "*_spec.rb",
    "test_*.rs",
    "*_test.rs",
]


def _find_files_by_patterns(
    root: Path,
    patterns: list[str],
    max_results: int = 50,
) -> list[Path]:
    """Find files matching glob patterns, limited to max_results."""
    found: list[Path] = []
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        for m in matches:
            if m not in found:
                found.append(m)
                if len(found) >= max_results:
                    return found
    return found


def _get_directory_tree(root: Path, max_depth: int = 3, max_items: int = 30) -> str:
    """Build an indented directory tree string."""
    lines: list[str] = []
    root_name = root.name or str(root)

    def _walk(dir_path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        if len(lines) >= max_items + 5:  # +5 for header
            return

        try:
            entries = sorted(
                [e for e in dir_path.iterdir() if not e.name.startswith(".")],
                key=lambda x: (not x.is_dir(), x.name.lower()),
            )
        except PermissionError:
            lines.append("  " * depth + "  [permission denied]")
            return

        for entry in entries:
            if len(lines) >= max_items + 5:
                break
            indent = "  " * (depth + 1)
            if entry.is_dir():
                lines.append(f"{indent}📁 {entry.name}/")
                _walk(entry, depth + 1)
            else:
                lines.append(f"{indent}📄 {entry.name}")

    lines.append(f"📁 {root_name}/")
    _walk(root, 0)
    return "\n".join(lines)


def _get_git_status(root: Path) -> dict[str, Any]:
    """Get git status information."""
    cwd = str(root)
    return {
        "branch": _run_git(["branch", "--show-current"], cwd),
        "last_commit": _run_git(["log", "--oneline", "-1"], cwd),
        "last_commit_full": _run_git(
            ["log", "-1", "--format=%H%n%an%n%ae%n%ai%n%s"], cwd
        ),
        "status": _run_git(["status", "--short"], cwd),
        "ahead_behind": _run_git(
            ["rev-list", "--count", "--left-right", "@{upstream}...HEAD"], cwd
        ),
        "stash_count": _run_git(["stash", "list"], cwd),
        "tags": _run_git(["tag", "--points-at", "HEAD"], cwd),
        "has_remote": bool(_run_git(["remote", "-v"], cwd)),
        "root_relative": _run_git(["rev-parse", "--show-prefix"], cwd) or ".",
    }


def _get_test_info(root: Path) -> dict[str, Any]:
    """Discover available test suites and commands."""
    info: dict[str, Any] = {
        "test_dirs": [],
        "test_files": [],
        "available_commands": [],
        "framework": "unknown",
    }

    # Find test directories
    for pattern in TEST_DIR_PATTERNS:
        d = root / pattern
        if d.is_dir():
            info["test_dirs"].append(str(d.relative_to(root)))

    # Find test files (optimized to avoid scanning virtualenvs, node_modules, etc.)
    import fnmatch
    ignored_dirs = {
        ".git", ".venv", "node_modules", ".nyx", "__pycache__",
        ".pytest_cache", ".nyx_memory", "build", "dist", "nyx.egg-info",
        "venv", "env", ".env",
    }
    test_files_found = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored directories in-place
        dirnames[:] = [d for d in dirnames if d not in ignored_dirs and not d.startswith(".")]
        for filename in filenames:
            for pattern in TEST_FILE_PATTERNS:
                if fnmatch.fnmatch(filename, pattern):
                    filepath = Path(dirpath) / filename
                    try:
                        rel = str(filepath.relative_to(root))
                        if rel not in test_files_found:
                            test_files_found.append(rel)
                    except ValueError:
                        continue
                    break
    info["test_files"] = sorted(test_files_found)

    # Detect test framework from config files
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        content = pyproject.read_text(encoding="utf-8", errors="ignore")
        if "[tool.pytest" in content:
            info["framework"] = "pytest"
        elif "[tool.unittest" in content:
            info["framework"] = "unittest"

    # Check for package.json test scripts
    pkg_json = root / "package.json"
    if pkg_json.exists():
        import json
        try:
            data = json.loads(pkg_json.read_text(encoding="utf-8", errors="ignore"))
            scripts = data.get("scripts", {})
            for name, cmd in scripts.items():
                if "test" in name.lower() or "test" in cmd.lower():
                    info["available_commands"].append(f"npm run {name}")
        except (json.JSONDecodeError, OSError):
            pass

    # Check for Makefile test targets
    makefile = root / "Makefile"
    if makefile.exists():
        content = makefile.read_text(encoding="utf-8", errors="ignore")
        for line in content.splitlines():
            if line.startswith("test") or "test" in line.split(":")[0:1]:
                info["available_commands"].append(f"make {line.split(':')[0].strip()}")

    # Common test commands
    if info["framework"] == "pytest":
        info["available_commands"].insert(0, "pytest")
        info["available_commands"].append("pytest -v")
        info["available_commands"].append("pytest --tb=short")
    elif (root / "package.json").exists():
        if not info["available_commands"]:
            info["available_commands"].append("npm test")

    # Check for tox
    if (root / "tox.ini").exists() or (root / "tox.ini").exists():
        info["available_commands"].append("tox")

    return info


def _get_project_language(root: Path) -> str:
    """Detect the primary programming language of the project."""
    config_files = {
        "pyproject.toml": "Python",
        "setup.py": "Python",
        "Cargo.toml": "Rust",
        "package.json": "JavaScript/TypeScript",
        "go.mod": "Go",
        "Gemfile": "Ruby",
        "composer.json": "PHP",
        "CMakeLists.txt": "C/C++",
        "build.gradle": "Java/Kotlin",
        "pom.xml": "Java",
        "Project.toml": "Julia",
    }
    for filename, lang in config_files.items():
        if (root / filename).exists():
            return lang

    # Fallback: check file extensions
    ext_counts: dict[str, int] = {}
    for f in root.rglob("*"):
        if f.is_file() and f.suffix:
            ext_counts[f.suffix] = ext_counts.get(f.suffix, 0) + 1

    ext_map = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".rs": "Rust",
        ".go": "Go",
        ".java": "Java",
        ".rb": "Ruby",
        ".php": "PHP",
        ".c": "C",
        ".cpp": "C++",
        ".cs": "C#",
    }
    best_lang = "Unknown"
    best_count = 0
    for ext, lang in ext_map.items():
        count = ext_counts.get(ext, 0)
        if count > best_count:
            best_count = count
            best_lang = lang

    return best_lang


def build_repo_map(root: str | Path | None = None) -> str:
    """
    Build a complete repository map as a formatted string.

    Args:
        root: Project root directory. Defaults to current working directory.

    Returns:
        A formatted string containing the repository map.
    """
    root = Path(root).resolve() if root else Path.cwd().resolve()

    if not root.is_dir():
        return f"Error: '{root}' is not a valid directory."

    parts: list[str] = []
    parts.append("=" * 60)
    parts.append(f"📋 REPOSITORY MAP: {root.name}")
    parts.append("=" * 60)

    # -- Basic info --
    lang = _get_project_language(root)
    parts.append(f"\n🔤 Language: {lang}")
    parts.append(f"📍 Path: {root}")

    # -- Directory tree --
    parts.append(f"\n📂 Directory Structure:")
    parts.append(_get_directory_tree(root))

    # -- Important files --
    important = _find_files_by_patterns(root, IMPORTANT_FILE_PATTERNS)
    if important:
        parts.append(f"\n⭐ Important Files:")
        for f in important:
            try:
                rel = f.relative_to(root)
                parts.append(f"  • {rel}")
            except ValueError:
                parts.append(f"  • {f.name}")

    # -- Git status --
    git_info = _get_git_status(root)
    if git_info["branch"]:
        parts.append(f"\n🌿 Git Status:")
        parts.append(f"  Branch: {git_info['branch']}")
        if git_info["last_commit"]:
            parts.append(f"  Last commit: {git_info['last_commit']}")
        if git_info["status"]:
            changes = git_info["status"].splitlines()
            parts.append(f"  Uncommitted changes ({len(changes)}):")
            for change in changes[:20]:
                parts.append(f"    {change}")
        if git_info["ahead_behind"]:
            parts.append(f"  Ahead/behind remote: {git_info['ahead_behind']}")
        if git_info["tags"]:
            parts.append(f"  Tags: {git_info['tags']}")
    else:
        parts.append("\n🌿 Git Status: Not a git repository (or git not available)")

    # -- Test info --
    test_info = _get_test_info(root)
    parts.append(f"\n🧪 Tests:")
    parts.append(f"  Framework: {test_info['framework']}")
    if test_info["test_dirs"]:
        parts.append(f"  Test directories: {', '.join(test_info['test_dirs'])}")
    if test_info["test_files"]:
        parts.append(f"  Test files ({len(test_info['test_files'])}):")
        for tf in test_info["test_files"][:15]:
            parts.append(f"    • {tf}")
        if len(test_info["test_files"]) > 15:
            parts.append(f"    ... and {len(test_info['test_files']) - 15} more")
    if test_info["available_commands"]:
        parts.append(f"  Available commands:")
        for cmd in test_info["available_commands"]:
            parts.append(f"    • {cmd}")

    parts.append("\n" + "=" * 60)
    return "\n".join(parts)


def build_repo_map_short(root: str | Path | None = None) -> str:
    """
    Build a concise one-line summary of the repository.
    Useful for quick context injection.
    """
    root = Path(root).resolve() if root else Path.cwd().resolve()
    lang = _get_project_language(root)
    branch = _run_git(["branch", "--show-current"], str(root))
    test_info = _get_test_info(root)

    parts = [
        f"📋 {root.name} ({lang})",
    ]
    if branch:
        parts.append(f"🌿 {branch}")
    if test_info["test_files"]:
        parts.append(f"🧪 {len(test_info['test_files'])} tests")
    return " | ".join(parts)