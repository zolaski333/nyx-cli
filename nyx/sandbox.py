"""
Nyx — Sandbox with project root and secure path resolution.

Provides:
  - A project root directory that acts as a sandbox boundary.
  - Secure path resolution that prevents path traversal attacks.
  - Automatic chdir to the project root for shell commands.
  - Path allow/deny lists for fine-grained filesystem access control.
"""
from __future__ import annotations

import logging
import os
import fnmatch
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PathTraversalError(ValueError):
    """Raised when a path resolves outside the allowed sandbox."""

    def __init__(self, path: str, resolved: str, root: str):
        self.path = path
        self.resolved = resolved
        self.root = root
        super().__init__(f"Path traversal blocked: '{path}' resolves to '{resolved}' outside project root '{root}'")


class SandboxDenyPathError(PermissionError):
    """Raised when a path matches an explicit sandbox deny rule."""

    def __init__(self, path: str, pattern: str):
        self.path = path
        self.pattern = pattern
        super().__init__(f"Path denied by sandbox rule: '{path}' matches '{pattern}'")


class Sandbox:
    """Project sandbox that enforces path boundaries and manages working directory."""

    def __init__(
        self,
        project_root: str | Path | None = None,
        allow_paths: list[str] | None = None,
        deny_paths: list[str] | None = None,
        auto_chdir: bool = True,
        use_docker: bool = False,
        docker_image: str = "python:3.11-slim-buster",
    ):
        self._root: Path | None = None
        self._original_cwd: Path = self._safe_cwd()
        self._auto_chdir = auto_chdir
        self._allow_extra: list[Path] = []
        self._deny_patterns: list[str] = deny_paths or []
        self._deny_resolved: list[Path] = []
        self.use_docker = use_docker
        self.docker_image = docker_image

        if project_root:
            self.set_root(project_root)

        if allow_paths:
            for p in allow_paths:
                resolved = Path(p).resolve()
                if resolved.exists() or resolved.parent.exists():
                    self._allow_extra.append(resolved)

        if deny_paths:
            for p in deny_paths:
                try:
                    self._deny_resolved.append(Path(p).resolve())
                except OSError:
                    logger.debug("Could not resolve deny path pattern: %s", p)

    @staticmethod
    def _safe_cwd() -> Path:
        """Get the current working directory safely, even if it has been deleted."""
        try:
            return Path.cwd().resolve()
        except FileNotFoundError:
            # CWD was deleted; fall back to a reasonable default
            return Path("C:/").resolve() if os.name == "nt" else Path("/").resolve()

    # ------------------------------------------------------------------
    # Root management
    # ------------------------------------------------------------------

    @property
    def root(self) -> Path | None:
        return self._root

    @property
    def root_str(self) -> str:
        return str(self._root) if self._root else ""

    def set_root(self, path: str | Path) -> None:
        """Set the project root directory. Creates it if it doesn't exist."""
        p = Path(path).resolve()
        p.mkdir(parents=True, exist_ok=True)
        self._root = p
        logger.info("Sandbox root set to: %s", p)

    def chdir(self, path: str | Path | None = None) -> None:
        """Change to a directory within the sandbox. If None, go to root."""
        target = self._root if path is None else self.resolve(path)
        if target is None:
            target = self._safe_cwd()
        os.chdir(target)
        logger.debug("Changed directory to: %s", target)

    def restore_cwd(self) -> None:
        """Restore the original working directory."""
        try:
            if self._original_cwd.exists():
                os.chdir(self._original_cwd)
                logger.debug("Restored working directory to: %s", self._original_cwd)
        except OSError as e:
            logger.debug("Could not restore working directory: %s", e)

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def resolve(self, path: str | Path, for_write: bool = False) -> Path:
        """Resolve a path safely within the sandbox.

        Args:
            path: The path to resolve (absolute or relative).
            for_write: If True, the path is being used for writing.

        Returns:
            Resolved absolute Path guaranteed to be within the sandbox.

        Raises:
            PathTraversalError: If the resolved path is outside the sandbox.
        """
        p = Path(path)

        # If it's already absolute, resolve it directly
        if p.is_absolute():
            resolved = p.resolve()
        else:
            # Relative paths are resolved against the sandbox root
            if self._root:
                resolved = (self._root / p).resolve()
            else:
                resolved = p.resolve()

        # If no sandbox root is set, allow everything
        if self._root is None:
            self._check_denied(resolved)
            return resolved

        # Check if the resolved path is within the sandbox root
        try:
            resolved.relative_to(self._root)
            self._check_denied(resolved)
            return resolved
        except ValueError:
            pass

        # Check if the path is in the allowed extras list
        for allowed in self._allow_extra:
            try:
                resolved.relative_to(allowed)
                self._check_denied(resolved)
                return resolved
            except ValueError:
                pass

        # Path traversal detected
        raise PathTraversalError(str(path), str(resolved), str(self._root))

    def _check_denied(self, resolved: Path) -> None:
        resolved_str = str(resolved)
        resolved_norm = resolved_str.lower() if os.name == "nt" else resolved_str

        for denied in self._deny_resolved:
            try:
                resolved.relative_to(denied)
                raise SandboxDenyPathError(resolved_str, str(denied))
            except ValueError:
                pass

        for pattern in self._deny_patterns:
            pattern_norm = pattern.lower() if os.name == "nt" else pattern
            if fnmatch.fnmatch(resolved_norm, pattern_norm):
                raise SandboxDenyPathError(resolved_str, pattern)

    def is_within_sandbox(self, path: str | Path) -> bool:
        """Check if a path is within the sandbox without raising."""
        try:
            self.resolve(path)
            return True
        except PathTraversalError:
            return False

    def safe_read_path(self, path: str | Path) -> Path:
        """Resolve a path for reading without escaping the sandbox.

        Reads are data exfiltration risks in an agentic CLI: source files, SSH
        keys, local config and environment-adjacent files can all be sent to a
        model.  Keep the same boundary as writes unless the user configured an
        explicit allow_paths entry.
        """
        return self.resolve(path, for_write=False)

    # ------------------------------------------------------------------
    # Command execution helpers
    # ------------------------------------------------------------------

    def is_docker_available(self) -> bool:
        """Check if docker or podman is available on the system."""
        import shutil
        return bool(shutil.which("docker") or shutil.which("podman"))

    def prepare_command(self, command: str) -> str:
        """Prepare a shell command for execution within the sandbox.

        If use_docker is enabled and docker/podman is available, wraps the command
        to run inside a Docker container. Local subprocess execution must pass
        cwd explicitly instead of relying on shell string prefixing.
        """
        if self.use_docker and self.is_docker_available():
            import shutil
            docker_bin = "docker" if shutil.which("docker") else "podman"
            root_dir = str(self._root) if self._root else os.getcwd()
            quoted_cmd = shlex_quote(command)
            return f'{docker_bin} run --rm -v {shlex_quote(root_dir)}:/workspace -w /workspace {self.docker_image} sh -c {quoted_cmd}'

        return command

    def to_dict(self) -> dict[str, Any]:
        """Serialize sandbox config for display/logging."""
        return {
            "root": self.root_str,
            "auto_chdir": self._auto_chdir,
            "allow_paths": [str(p) for p in self._allow_extra],
            "deny_patterns": list(self._deny_patterns),
            "use_docker": self.use_docker,
            "docker_image": self.docker_image,
        }


def shlex_quote(s: str) -> str:
    """Simple shell quoting for a path string."""
    if os.name == "nt":
        # Windows cmd.exe uses double-quote style
        return '"' + s.replace('"', '""') + '"'
    # Replace single quotes with end-quote, escaped quote, begin-quote
    escaped = s.replace("'", "'\\''")
    return f"'{escaped}'"

