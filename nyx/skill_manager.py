"""
Skill system: discover Python skills statically and execute them safely.

A skill can be a single Python file or a package. Discovery reads metadata
without importing the skill module. Execution happens on demand, preferably in
a worker process with a hard timeout.
"""
from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nyx.providers.base import ToolDefinition


SKILL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
DEFAULT_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {}}


@dataclass
class SkillResult:
    """Structured result from a skill execution."""

    name: str
    output: str = ""
    status: str = "completed"
    error: str | None = None
    error_type: str | None = None
    duration_seconds: float = 0.0
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and self.status == "completed"

    def to_text(self) -> str:
        if self.ok:
            suffix = "\n[Skill output truncated]" if self.truncated else ""
            return f"{self.output}{suffix}"
        return f"[Skill:{self.name}] {self.status} ({self.error_type}): {self.error}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "output": self.output,
            "status": self.status,
            "error": self.error,
            "error_type": self.error_type,
            "duration_seconds": self.duration_seconds,
            "truncated": self.truncated,
        }


@dataclass
class Skill:
    """A discovered skill ready to expose as a tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    file_path: str
    entrypoint: str = "execute"

    def to_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=f"skill_{self.name}",
            description=self.description,
            parameters=self.parameters,
        )


class SkillManager:
    """Discovers and manages skills."""

    def __init__(
        self,
        skills_dir: str = "",
        *,
        process_isolation: bool = True,
        default_timeout_seconds: float | None = 30,
        max_output_chars: int = 20000,
    ) -> None:
        self._skills: dict[str, Skill] = {}
        self._skills_dir = skills_dir
        self.process_isolation = process_isolation
        self.default_timeout_seconds = default_timeout_seconds
        self.max_output_chars = max_output_chars

    def discover(self, skills_dir: str | None = None) -> list[Skill]:
        """Scan the skills directory and load valid skill metadata.

        Discovery does not import skill modules, so top-level Python code does
        not run while Nyx starts up. Python code runs only when a skill is
        executed.
        """
        directory = skills_dir or self._skills_dir
        if not directory:
            return []
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            return []

        found: list[Skill] = []
        seen_names: set[str] = set()
        print(f"  ! Discovering trusted local Python skills from {root}")

        for entry in sorted(root.iterdir(), key=lambda p: p.name):
            skill = self._discover_entry(entry, root)
            if not skill:
                continue
            if skill.name in seen_names:
                print(f"  ! Skill '{skill.name}': duplicate name, skipping {entry.name}.")
                continue
            seen_names.add(skill.name)
            found.append(skill)
            print(f"  ok Skill '{skill.name}' discovered")

        self._skills = {s.name: s for s in found}
        return found

    def _discover_entry(self, entry: Path, root: Path) -> Skill | None:
        try:
            resolved = entry.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            print(f"  ! Skill '{entry.name}': path escapes skills directory, skipping.")
            return None

        if entry.is_file() and entry.suffix == ".py" and not entry.name.startswith("_"):
            return self._load_metadata(entry.stem, entry)

        if entry.is_dir():
            manifest = entry / "skill.json"
            init_file = entry / "__init__.py"
            if manifest.is_file():
                return self._load_manifest(entry.name, manifest, init_file if init_file.is_file() else None)
            if init_file.is_file():
                return self._load_metadata(entry.name, init_file)

        return None

    def _load_manifest(self, fallback_name: str, manifest_path: Path, init_file: Path | None) -> Skill | None:
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  x Skill '{fallback_name}': manifest load error - {e}")
            return None

        file_name = data.get("file", "__init__.py")
        file_path = (manifest_path.parent / file_name).resolve()
        try:
            file_path.relative_to(manifest_path.parent.resolve())
        except (OSError, ValueError):
            print(f"  ! Skill '{fallback_name}': manifest file escapes package directory, skipping.")
            return None
        if not file_path.is_file():
            if init_file:
                file_path = init_file
            else:
                print(f"  ! Skill '{fallback_name}': entry file missing, skipping.")
                return None

        return self._validate_skill(
            fallback_name=fallback_name,
            file_path=file_path,
            name=data.get("name", fallback_name),
            description=data.get("description", ""),
            parameters=data.get("parameters", DEFAULT_PARAMETERS),
            entrypoint=data.get("entrypoint", "execute"),
        )

    def _load_metadata(self, fallback_name: str, file_path: Path) -> Skill | None:
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        except Exception as e:
            print(f"  x Skill '{fallback_name}': metadata parse error - {e}")
            return None

        metadata: dict[str, Any] = {}
        execute_found = False
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {"name", "description", "parameters"}:
                        try:
                            metadata[target.id] = ast.literal_eval(node.value)
                        except Exception:
                            print(f"  ! Skill '{fallback_name}': metadata '{target.id}' must be a literal, skipping.")
                            return None
            elif isinstance(node, ast.FunctionDef) and node.name == "execute":
                execute_found = True

        if not execute_found:
            print(f"  ! Skill '{fallback_name}': missing function 'execute(arguments)'.")
            return None

        return self._validate_skill(
            fallback_name=fallback_name,
            file_path=file_path,
            name=metadata.get("name", fallback_name),
            description=metadata.get("description", ""),
            parameters=metadata.get("parameters", DEFAULT_PARAMETERS),
            entrypoint="execute",
        )

    def _validate_skill(
        self,
        *,
        fallback_name: str,
        file_path: Path,
        name: Any,
        description: Any,
        parameters: Any,
        entrypoint: Any,
    ) -> Skill | None:
        if not isinstance(name, str) or not SKILL_NAME_RE.match(name):
            print(f"  ! Skill '{fallback_name}': invalid name, skipping.")
            return None
        if not isinstance(description, str) or not description.strip():
            print(f"  ! Skill '{name}': no description set, skipping.")
            return None
        if not self._is_valid_parameters(parameters):
            print(f"  ! Skill '{name}': invalid JSON schema parameters, skipping.")
            return None
        if not isinstance(entrypoint, str) or not SKILL_NAME_RE.match(entrypoint):
            print(f"  ! Skill '{name}': invalid entrypoint, skipping.")
            return None

        return Skill(
            name=name,
            description=description,
            parameters=parameters,
            file_path=str(file_path),
            entrypoint=entrypoint,
        )

    @staticmethod
    def _is_valid_parameters(parameters: Any) -> bool:
        if not isinstance(parameters, dict):
            return False
        if parameters.get("type", "object") != "object":
            return False
        properties = parameters.get("properties", {})
        if not isinstance(properties, dict):
            return False
        required = parameters.get("required", [])
        return isinstance(required, list)

    def get_skill(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def get_tool_definitions(self) -> list[ToolDefinition]:
        return [s.to_definition() for s in self._skills.values()]

    def execute_skill_result(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> SkillResult:
        skill = self._skills.get(name)
        if not skill:
            return SkillResult(name=name, status="failed", error=f"Unknown skill: {name}", error_type="unknown_skill")

        started = time.monotonic()
        timeout = timeout_seconds if timeout_seconds is not None else self.default_timeout_seconds
        if self.process_isolation:
            from nyx.skill_worker import run_skill_in_process

            result = run_skill_in_process(skill=skill, arguments=arguments, timeout_seconds=timeout)
        else:
            from nyx.skill_worker import run_skill_in_current_process

            result = run_skill_in_current_process(skill=skill, arguments=arguments)

        result.duration_seconds = result.duration_seconds or (time.monotonic() - started)
        return self._truncate_result(result)

    def execute_skill(self, name: str, arguments: dict[str, Any]) -> str:
        return self.execute_skill_result(name, arguments).to_text()

    def _truncate_result(self, result: SkillResult) -> SkillResult:
        if self.max_output_chars > 0 and len(result.output) > self.max_output_chars:
            result.output = result.output[: self.max_output_chars]
            result.truncated = True
        return result
