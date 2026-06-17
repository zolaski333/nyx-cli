"""
Skill system — dynamically loads Python skills from a directory.

A skill is a Python file (or package) in the skills/ directory that exports:
- `name: str` (required)
- `description: str` (required)
- `parameters: dict` (JSON Schema, required)
- `execute(arguments: dict) -> str` (required)

Skills are registered as tools the agent can call.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nyx.providers.base import ToolDefinition


@dataclass
class Skill:
    """A loaded skill ready for execution."""
    name: str
    description: str
    parameters: dict[str, Any]
    execute_fn: Any = None
    file_path: str = ""
    module: Any = None

    def to_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=f"skill_{self.name}",
            description=self.description,
            parameters=self.parameters,
        )

    def execute(self, arguments: dict[str, Any]) -> str:
        if self.execute_fn:
            try:
                result = self.execute_fn(arguments)
                return str(result)
            except Exception as e:
                return f"[Skill:{self.name}] Error: {e}"
        return f"[Skill:{self.name}] Not loaded."


class SkillManager:
    """Discovers and manages skills."""

    def __init__(self, skills_dir: str = "") -> None:
        self._skills: dict[str, Skill] = {}
        self._skills_dir = skills_dir

    def discover(self, skills_dir: str | None = None) -> list[Skill]:
        """Scan the skills directory and load all valid skills.

        Skills are trusted local Python code. Importing them executes module
        top-level code, so only point this manager at directories you control.
        """
        directory = skills_dir or self._skills_dir
        if not directory or not os.path.isdir(directory):
            return []
        root = Path(directory).resolve()

        found: list[Skill] = []
        print(f"  ! Loading trusted local Python skills from {root}")
        sys.path.insert(0, directory)

        for entry in sorted(os.listdir(directory)):
            path = os.path.join(directory, entry)
            try:
                resolved = Path(path).resolve()
                resolved.relative_to(root)
            except (OSError, ValueError):
                print(f"  ! Skill '{entry}': path escapes skills directory, skipping.")
                continue

            # Single-file skill: <name>.py
            if entry.endswith(".py") and not entry.startswith("_"):
                mod_name = entry[:-3]
                skill = self._load_from_file(mod_name, path)
                if skill:
                    found.append(skill)

            # Package skill: <name>/__init__.py
            elif os.path.isdir(path) and os.path.isfile(os.path.join(path, "__init__.py")):
                skill = self._load_from_file(entry, os.path.join(path, "__init__.py"))
                if skill:
                    found.append(skill)

        sys.path.pop(0)
        self._skills = {s.name: s for s in found}
        return found

    def _load_from_file(self, mod_name: str, file_path: str) -> Skill | None:
        try:
            spec = importlib.util.spec_from_file_location(mod_name, file_path)
            if not spec or not spec.loader:
                return None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"  x Skill '{mod_name}': load error - {e}")
            return None

        # Validate required exports
        name = getattr(mod, "name", mod_name)
        description = getattr(mod, "description", "")
        parameters = getattr(mod, "parameters", {"type": "object", "properties": {}})
        execute_fn = getattr(mod, "execute", None)

        if not description:
            print(f"  ! Skill '{name}': no description set, skipping.")
            return None
        if not execute_fn or not callable(execute_fn):
            print(f"  ! Skill '{name}': missing callable 'execute(arguments)'.")
            return None

        skill = Skill(
            name=name,
            description=description,
            parameters=parameters,
            execute_fn=execute_fn,
            file_path=file_path,
            module=mod,
        )
        print(f"  ok Skill '{name}' loaded")
        return skill

    def get_skill(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def get_tool_definitions(self) -> list[ToolDefinition]:
        return [s.to_definition() for s in self._skills.values()]

    def execute_skill(self, name: str, arguments: dict[str, Any]) -> str:
        skill = self._skills.get(name)
        if not skill:
            return f"[Skill] Unknown skill: {name}"
        return skill.execute(arguments)
