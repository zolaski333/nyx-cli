"""
Nyx — Repository map tool.

Provides a structured overview of the current repository:
- Directory tree (top-level + key subdirectories)
- Important files (config, manifest, CI, etc.)
- Git status (branch, changes, last commit)
- Available test suites and commands
"""
from __future__ import annotations

import ast
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_git(args: list[str], cwd: str | Path | None = None) -> str:
    """Run a git command and return stdout (or empty string on failure)."""
    safe_cwd = str(Path(cwd or ".").resolve())
    try:
        proc = subprocess.run(
            ["git", "-c", f"safe.directory={safe_cwd}", *args],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=safe_cwd,
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
            info["test_dirs"].append(d.relative_to(root).as_posix())

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
                        rel = filepath.relative_to(root).as_posix()
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


def _format_func(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Format a function or method signature including async, decorators, and arguments."""
    is_async = isinstance(node, ast.AsyncFunctionDef)
    decorators = []
    for dec in node.decorator_list:
        try:
            decorators.append("@" + ast.unparse(dec))
        except Exception:
            pass
    
    dec_str = " ".join(decorators) + " " if decorators else ""
    async_str = "async " if is_async else ""
    
    try:
        args_str = ast.unparse(node.args)
    except Exception:
        args_str = ""
        
    return f"{dec_str}{async_str}{node.name}({args_str})"


def _format_class(node: ast.ClassDef) -> str:
    """Format a class definition including its base classes (inheritance)."""
    bases = []
    for base in node.bases:
        try:
            bases.append(ast.unparse(base))
        except Exception:
            pass
    bases_str = f" ({', '.join(bases)})" if bases else ""
    return f"Class: {node.name}{bases_str}"


def _get_regex_symbols(file_path: Path) -> list[str]:
    """Extract class, struct, type and function definitions from non-Python files using regex."""
    ext = file_path.suffix.lower()
    symbols = []
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    lines = content.splitlines()

    # JS/TS patterns
    if ext in (".js", ".ts", ".jsx", ".tsx"):
        class_pat = re.compile(r"^\s*(?:export\s+)?(?:default\s+)?class\s+(\w+)")
        fn_pat1 = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)")
        fn_pat2 = re.compile(r"^\s*(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>")
        method_pat = re.compile(r"^\s*(?:private|protected|public|async)?\s*(\w+)\s*\(([^)]*)\)\s*(?::|{)")

        in_class = False
        for line in lines:
            line_str = line.strip()
            cm = class_pat.match(line_str)
            if cm:
                symbols.append(f"    - Class: {cm.group(1)}")
                in_class = True
                continue

            fm1 = fn_pat1.match(line_str)
            if fm1:
                symbols.append(f"    - Function: {fm1.group(1)}({fm1.group(2)})")
                continue

            fm2 = fn_pat2.match(line_str)
            if fm2:
                symbols.append(f"    - Function: {fm2.group(1)}({fm2.group(2)})")
                continue

            if in_class and not line_str.startswith("}"):
                mm = method_pat.match(line_str)
                if mm and mm.group(1) not in ("if", "for", "while", "switch", "catch", "constructor"):
                    symbols.append(f"      • {mm.group(1)}({mm.group(2)})")

            if line_str.startswith("}"):
                in_class = False

    # Rust patterns
    elif ext == ".rs":
        struct_pat = re.compile(r"^\s*(?:pub\s+)?(?:struct|enum|trait)\s+(\w+)")
        fn_pat = re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*\(([^)]*)\)")
        impl_pat = re.compile(r"^\s*impl(?:\s+<\w+>)?\s+(\w+)")

        in_impl = False
        for line in lines:
            line_str = line.strip()
            sm = struct_pat.match(line_str)
            if sm:
                symbols.append(f"    - Struct/Enum/Trait: {sm.group(1)}")
                continue

            im = impl_pat.match(line_str)
            if im:
                symbols.append(f"    - impl {im.group(1)}")
                in_impl = True
                continue

            fm = fn_pat.match(line_str)
            if fm:
                if in_impl:
                    symbols.append(f"      • fn {fm.group(1)}({fm.group(2)})")
                else:
                    symbols.append(f"    - fn {fm.group(1)}({fm.group(2)})")

            if line_str.startswith("}"):
                in_impl = False

    # Go patterns
    elif ext == ".go":
        type_pat = re.compile(r"^\s*type\s+(\w+)\s+(?:struct|interface)")
        fn_pat = re.compile(r"^\s*func\s+(\w+)\s*\(([^)]*)\)")
        method_pat = re.compile(r"^\s*func\s*\(\s*\w+\s*\*?(\w+)\s*\)\s*(\w+)\s*\(([^)]*)\)")

        for line in lines:
            line_str = line.strip()
            tm = type_pat.match(line_str)
            if tm:
                symbols.append(f"    - Type: {tm.group(1)}")
                continue

            mm = method_pat.match(line_str)
            if mm:
                symbols.append(f"      • func ({mm.group(1)}) {mm.group(2)}({mm.group(3)})")
                continue

            fm = fn_pat.match(line_str)
            if fm:
                symbols.append(f"    - func {fm.group(1)}({fm.group(2)})")

    # C/C++ patterns
    elif ext in (".cpp", ".hpp", ".c", ".h"):
        class_pat = re.compile(r"^\s*(?:class|struct)\s+(\w+)")
        fn_pat = re.compile(r"^\s*(?:\w+\s+)?(?:async\s+)?(\w+)\s*\(([^)]*)\)\s*(?:const)?\s*(?:{|;)")

        for line in lines:
            line_str = line.strip()
            cm = class_pat.match(line_str)
            if cm:
                symbols.append(f"    - Class/Struct: {cm.group(1)}")
                continue

            fm = fn_pat.match(line_str)
            if fm:
                if fm.group(1) not in ("if", "for", "while", "switch", "catch", "return"):
                    symbols.append(f"    - Function: {fm.group(1)}({fm.group(2)})")

    return symbols


def _get_ast_symbols(root: Path, max_files: int = 15) -> str:
    """Scan key source files (Python, JS/TS, Go, Rust, C++) and extract semantic structures.

    Prioritizes files modified in Git and limits scan to max_files.
    """
    ignored_dirs = {
        ".git", ".venv", "node_modules", ".nyx", "__pycache__",
        ".pytest_cache", ".nyx_memory", "build", "dist", "nyx.egg-info",
        "venv", "env", ".env", "tests", "test",
    }
    allowed_extensions = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".cpp", ".hpp", ".c", ".h"
    }

    # 1. Identify modified files via Git
    modified_files = set()
    git_status = _run_git(["status", "--short"], root)
    if git_status:
        for line in git_status.splitlines():
            if len(line) > 3:
                path_str = line[3:].strip()
                modified_files.add((root / path_str).resolve())

    # 2. Collect all source files
    source_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ignored_dirs and not d.startswith(".")]
        for filename in filenames:
            file_path = Path(dirpath) / filename
            if file_path.suffix.lower() in allowed_extensions and filename != "__init__.py":
                source_files.append(file_path.resolve())

    # 3. Sort files: modified first, then alphabetically
    def sort_key(p: Path) -> tuple[int, str]:
        is_mod = 0 if p in modified_files else 1
        return (is_mod, str(p).lower())

    source_files.sort(key=sort_key)

    if not source_files:
        return ""

    lines = ["\n🧬 Semantic Symbols:"]
    files_to_scan = source_files[:max_files]

    for source_file in files_to_scan:
        try:
            rel_path = source_file.relative_to(root).as_posix()
        except ValueError:
            rel_path = source_file.name

        try:
            file_symbols = []
            if source_file.suffix.lower() == ".py":
                content = source_file.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(content, filename=str(source_file))

                for node in ast.iter_child_nodes(tree):
                    if isinstance(node, ast.ClassDef):
                        class_str = f"    - {_format_class(node)}"
                        file_symbols.append(class_str)
                        for child in ast.iter_child_nodes(node):
                            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                if child.name == "__init__" or not child.name.startswith("_"):
                                    file_symbols.append(f"      • {_format_func(child)}")
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if not node.name.startswith("_"):
                            file_symbols.append(f"    - Function: {_format_func(node)}")
            else:
                # Use regex parser for other languages
                file_symbols = _get_regex_symbols(source_file)

            if file_symbols:
                lines.append(f"  • {rel_path}:")
                lines.extend(file_symbols[:15])
                if len(file_symbols) > 15:
                    lines.append(f"    ... and {len(file_symbols) - 15} more symbols")
        except Exception as e:
            logger.debug("Could not parse symbols for %s: %s", source_file, e)

    if len(source_files) > max_files:
        lines.append(f"  (Semantic scan limited to first {max_files} of {len(source_files)} code files)")

    return "\n".join(lines)


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

    # Index static dependencies
    try:
        index_dependencies(root)
    except Exception as e:
        logger.debug("Failed to index dependencies: %s", e)

    parts: list[str] = []
    parts.append("=" * 60)
    parts.append(f"📋 REPOSITORY MAP: {root.name}")
    parts.append("=" * 60)

    # -- Basic info --
    lang = _get_project_language(root)
    parts.append(f"\n🔤 Language: {lang}")
    parts.append(f"📍 Path: {root}")

    # -- Directory tree --
    parts.append("\n📂 Directory Structure:")
    parts.append(_get_directory_tree(root))

    # -- Important files --
    important = _find_files_by_patterns(root, IMPORTANT_FILE_PATTERNS)
    if important:
        parts.append("\n⭐ Important Files:")
        for f in important:
            try:
                rel = f.relative_to(root).as_posix()
                parts.append(f"  • {rel}")
            except ValueError:
                parts.append(f"  • {f.name}")

    # -- Git status --
    git_info = _get_git_status(root)
    if git_info["branch"]:
        parts.append("\n🌿 Git Status:")
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
    parts.append("\n🧪 Tests:")
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
        parts.append("  Available commands:")
        for cmd in test_info["available_commands"]:
            parts.append(f"    • {cmd}")

    # -- AST symbols --
    ast_symbols = _get_ast_symbols(root)
    if ast_symbols:
        parts.append(ast_symbols)

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


def index_dependencies(root: str | Path | None = None) -> dict[str, list[str]]:
    """Index file-level dependencies by parsing imports/requires and save to .nyx/repo_graph.json."""
    import re
    import json
    if root is None:
        root = Path.cwd()
    else:
        root = Path(root).resolve()

    graph = {}
    
    # Define regex for imports per extension
    patterns = {
        ".py": [
            re.compile(r"^\s*import\s+([\w\.]+)"),
            re.compile(r"^\s*from\s+([\w\.]+)\s+import")
        ],
        ".js": [
            re.compile(r"import\s+.*\s+from\s+['\"]([^'\"]+)['\"]"),
            re.compile(r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)")
        ],
        ".ts": [
            re.compile(r"import\s+.*\s+from\s+['\"]([^'\"]+)['\"]"),
            re.compile(r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)")
        ],
        ".jsx": [
            re.compile(r"import\s+.*\s+from\s+['\"]([^'\"]+)['\"]"),
            re.compile(r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)")
        ],
        ".tsx": [
            re.compile(r"import\s+.*\s+from\s+['\"]([^'\"]+)['\"]"),
            re.compile(r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)")
        ],
        ".rs": [
            re.compile(r"^\s*(?:pub\s+)?use\s+([\w:]+)")
        ],
        ".go": [
            re.compile(r"import\s+['\"]([^'\"]+)['\"]"),
            re.compile(r"^\s*['\"]([^'\"]+)['\"]") # for multiline import blocks
        ]
    }

    # Helper to walk files recursively
    for dirpath, _, filenames in os.walk(root):
        dir_parts = Path(dirpath).parts
        if any(part.startswith(".") or part in ("node_modules", "venv", "__pycache__", "build", "dist") for part in dir_parts):
            continue
        
        for fname in filenames:
            fpath = Path(dirpath) / fname
            ext = fpath.suffix.lower()
            if ext not in patterns:
                continue
            
            rel_path = fpath.relative_to(root).as_posix()
            imports = []
            
            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore")
                lines = content.splitlines()
                
                in_go_import = False
                for line in lines:
                    line_stripped = line.strip()
                    if ext == ".go":
                        if line_stripped.startswith("import ("):
                            in_go_import = True
                            continue
                        elif in_go_import and line_stripped.startswith(")"):
                            in_go_import = False
                            continue
                    
                    for pat in patterns[ext]:
                        if ext == ".go" and in_go_import:
                            m = patterns[".go"][1].match(line_stripped)
                            if m:
                                imports.append(m.group(1))
                                break
                        else:
                            m = pat.search(line)
                            if m:
                                imports.append(m.group(1))
                                break
            except Exception as e:
                logger.warning("Error indexing dependencies for %s: %s", rel_path, e)
                continue
                
            if imports:
                graph[rel_path] = sorted(list(set(imports)))

    try:
        nyx_dir = root / ".nyx"
        nyx_dir.mkdir(exist_ok=True)
        graph_file = nyx_dir / "repo_graph.json"
        graph_file.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Could not save repo_graph.json: %s", e)

    return graph
