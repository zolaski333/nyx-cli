"""
Nyx — Granular permission model for shell and file operations.

Defines permission levels, rules, and a permission checker that the agent
consults before executing any shell command or file operation.

Permission levels (from most restrictive to least):
  - deny:     Always blocked, no override.
  - prompt:   Requires interactive user approval each time.
  - allow:    Allowed without prompting.

Configuration is loaded from the Nyx config file under the "permissions" key.
"""
from __future__ import annotations

import fnmatch
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Permission level
# ---------------------------------------------------------------------------


class PermissionLevel(Enum):
    """Granular permission levels for operations."""

    DENY = "deny"        # Always blocked
    PROMPT = "prompt"    # Requires user approval
    ALLOW = "allow"      # Allowed without prompting

    @classmethod
    def from_str(cls, s: str) -> PermissionLevel:
        norm = s.strip().lower()
        for level in cls:
            if level.value == norm:
                return level
        logger.warning("Unknown permission level '%s', defaulting to PROMPT", s)
        return cls.PROMPT


# ---------------------------------------------------------------------------
# Permission rule
# ---------------------------------------------------------------------------


@dataclass
class PermissionRule:
    """A single permission rule matching a pattern."""

    pattern: str            # Glob or regex pattern
    level: PermissionLevel  # What to do when matched
    description: str = ""   # Human-readable description
    is_regex: bool = False  # If True, pattern is a regex; otherwise glob

    def matches(self, target: str) -> bool:
        """Check if this rule matches the given target string."""
        if self.is_regex:
            try:
                return bool(re.search(self.pattern, target))
            except re.error:
                logger.warning("Invalid regex pattern: %s", self.pattern)
                return False
        # Glob matching (case-insensitive on the pattern)
        return fnmatch.fnmatch(target.lower(), self.pattern.lower())


# ---------------------------------------------------------------------------
# Permission category
# ---------------------------------------------------------------------------


@dataclass
class PermissionCategory:
    """A category of permissions (e.g. 'shell', 'filesystem')."""

    name: str
    default_level: PermissionLevel = PermissionLevel.ALLOW
    rules: list[PermissionRule] = field(default_factory=list)

    def check(self, target: str) -> PermissionLevel:
        """Check a target against rules. Most specific match wins (DENY > PROMPT > ALLOW)."""
        # Rules are evaluated in order; last matching rule wins (allows override)
        matched_level: PermissionLevel | None = None
        for rule in self.rules:
            if rule.matches(target):
                matched_level = rule.level
                logger.debug("Rule '%s' matched '%s' → %s", rule.pattern, target, rule.level.value)

        if matched_level is not None:
            return matched_level
        return self.default_level


# ---------------------------------------------------------------------------
# Permission manager
# ---------------------------------------------------------------------------


class PermissionManager:
    """Central permission manager for all operations."""

    def __init__(self, config: dict | None = None):
        self.categories: dict[str, PermissionCategory] = {}
        self._approval_callback: Callable[[str, str, str], tuple[bool, str]] | None = None
        self._load_defaults()
        if config:
            self._load_config(config)

    def set_approval_callback(
        self,
        callback: Callable[[str, str, str], tuple[bool, str]] | None,
    ) -> None:
        """Set the callback for interactive approval prompts.

        The callback receives (category_name, operation_description, target)
        and returns (approved: bool, reason: str).
        """
        self._approval_callback = callback

    # ------------------------------------------------------------------
    # Default rules
    # ------------------------------------------------------------------

    def _load_defaults(self) -> None:
        """Load sensible default permission rules."""
        # -- Shell commands --
        shell = PermissionCategory(
            name="shell",
            default_level=PermissionLevel.ALLOW,
            rules=[
                # Destructive / system-level commands → prompt
                PermissionRule("rm *", PermissionLevel.PROMPT, "File deletion"),
                PermissionRule("rmdir *", PermissionLevel.PROMPT, "Directory deletion"),
                PermissionRule("mv *", PermissionLevel.PROMPT, "File move/rename"),
                PermissionRule("cp *", PermissionLevel.PROMPT, "File copy"),
                PermissionRule("chmod *", PermissionLevel.PROMPT, "Permission change"),
                PermissionRule("chown *", PermissionLevel.PROMPT, "Ownership change"),
                PermissionRule("dd *", PermissionLevel.PROMPT, "Raw disk write"),
                PermissionRule("sudo *", PermissionLevel.PROMPT, "Superuser execution"),
                PermissionRule("su *", PermissionLevel.PROMPT, "User switch"),
                PermissionRule("passwd *", PermissionLevel.PROMPT, "Password change"),
                PermissionRule("kill *", PermissionLevel.PROMPT, "Process kill"),
                PermissionRule("mkfs*", PermissionLevel.PROMPT, "Filesystem creation"),
                PermissionRule("fdisk *", PermissionLevel.PROMPT, "Disk partitioning"),
                PermissionRule("mount *", PermissionLevel.PROMPT, "Filesystem mount"),
                PermissionRule("umount *", PermissionLevel.PROMPT, "Filesystem unmount"),
                PermissionRule("iptables *", PermissionLevel.PROMPT, "Firewall change"),
                PermissionRule("wget *", PermissionLevel.PROMPT, "Remote download"),
                PermissionRule("curl *", PermissionLevel.PROMPT, "Remote request"),
                PermissionRule("apt *", PermissionLevel.PROMPT, "Package manager"),
                PermissionRule("apt-get *", PermissionLevel.PROMPT, "Package manager"),
                PermissionRule("yum *", PermissionLevel.PROMPT, "Package manager"),
                PermissionRule("dnf *", PermissionLevel.PROMPT, "Package manager"),
                PermissionRule("pacman *", PermissionLevel.PROMPT, "Package manager"),
                PermissionRule("pip install *", PermissionLevel.PROMPT, "Python package install"),
                PermissionRule("npm install *", PermissionLevel.PROMPT, "Node package install"),
                PermissionRule("git push *", PermissionLevel.PROMPT, "Git push"),
                PermissionRule("git reset *", PermissionLevel.PROMPT, "Git reset"),
                PermissionRule("git rebase *", PermissionLevel.PROMPT, "Git rebase"),
                PermissionRule("git merge *", PermissionLevel.PROMPT, "Git merge"),
                PermissionRule("git cherry-pick *", PermissionLevel.PROMPT, "Git cherry-pick"),
                PermissionRule("docker *", PermissionLevel.PROMPT, "Docker operation"),
                PermissionRule("systemctl *", PermissionLevel.PROMPT, "System service control"),
                PermissionRule("journalctl *", PermissionLevel.PROMPT, "Journal control"),

                # Shell operators that indicate destructive intent
                PermissionRule("* > *", PermissionLevel.PROMPT, "Output redirection (write)", is_regex=False),
                PermissionRule("* >> *", PermissionLevel.PROMPT, "Output redirection (append)", is_regex=False),

                # Explicitly denied commands (use regex for precise matching)
                PermissionRule(r"^rm\s+-rf\s+/\s*$", PermissionLevel.DENY, "Root deletion", is_regex=True),
                PermissionRule(r"^rm\s+-rf\s+/\*\s*$", PermissionLevel.DENY, "Recursive root deletion", is_regex=True),
                PermissionRule(r"^dd\s+if=.*\s+of=/\s*", PermissionLevel.DENY, "Raw write to root", is_regex=True),
            ],
        )
        import os
        if os.name == "nt":
            shell.rules.extend([
                # Windows commands → prompt
                PermissionRule("del *", PermissionLevel.PROMPT, "File deletion"),
                PermissionRule("rd *", PermissionLevel.PROMPT, "Directory deletion"),
                PermissionRule("move *", PermissionLevel.PROMPT, "File move/rename"),
                PermissionRule("copy *", PermissionLevel.PROMPT, "File copy"),
                PermissionRule("xcopy *", PermissionLevel.PROMPT, "File copy"),
                PermissionRule("robocopy *", PermissionLevel.PROMPT, "File copy"),
                PermissionRule("powershell *", PermissionLevel.PROMPT, "PowerShell execution"),
                PermissionRule("pwsh *", PermissionLevel.PROMPT, "PowerShell execution"),
                PermissionRule("cmd *", PermissionLevel.PROMPT, "Command execution"),
                PermissionRule("taskkill *", PermissionLevel.PROMPT, "Process kill"),
                PermissionRule("net *", PermissionLevel.PROMPT, "Network/user management"),
                PermissionRule("sc *", PermissionLevel.PROMPT, "Service control"),
                PermissionRule("reg *", PermissionLevel.PROMPT, "Registry operation"),
                PermissionRule("icacls *", PermissionLevel.PROMPT, "Permission change"),
                PermissionRule("takeown *", PermissionLevel.PROMPT, "Ownership change"),
                # Explicitly denied commands
                PermissionRule("format *", PermissionLevel.DENY, "Disk format"),
            ])


        # -- Filesystem operations --
        fs = PermissionCategory(
            name="filesystem",
            default_level=PermissionLevel.ALLOW,
            rules=[
                # Writing outside project root → prompt
                PermissionRule("/etc/*", PermissionLevel.PROMPT, "System config file"),
                PermissionRule("/usr/*", PermissionLevel.PROMPT, "System file"),
                PermissionRule("/bin/*", PermissionLevel.PROMPT, "System binary"),
                PermissionRule("/sbin/*", PermissionLevel.PROMPT, "System binary"),
                PermissionRule("/var/*", PermissionLevel.PROMPT, "System data"),
                PermissionRule("/dev/*", PermissionLevel.PROMPT, "Device file"),
                PermissionRule("/proc/*", PermissionLevel.PROMPT, "Process file"),
                PermissionRule("/sys/*", PermissionLevel.PROMPT, "System file"),
                PermissionRule("/boot/*", PermissionLevel.PROMPT, "Boot file"),
                PermissionRule("/root/*", PermissionLevel.PROMPT, "Root home"),
                PermissionRule("/home/*", PermissionLevel.PROMPT, "User home"),

                # Deny writing to critical system paths
                PermissionRule("/etc/shadow", PermissionLevel.DENY, "Password file"),
                PermissionRule("/etc/sudoers*", PermissionLevel.DENY, "Sudoers file"),
                PermissionRule("/etc/passwd", PermissionLevel.DENY, "Password file"),
            ],
        )
        if os.name == "nt":
            fs.rules.extend([
                PermissionRule("C:\\Windows\\*", PermissionLevel.PROMPT, "Windows system dir"),
                PermissionRule("C:\\Program Files\\*", PermissionLevel.PROMPT, "Program Files"),
                PermissionRule("C:\\Program Files (x86)\\*", PermissionLevel.PROMPT, "Program Files"),
                PermissionRule("C:\\Users\\*", PermissionLevel.PROMPT, "User directory"),
                PermissionRule("C:\\Users\\*\\AppData\\Local\\Temp\\*", PermissionLevel.ALLOW, "Temp folder"),
                PermissionRule("C:\\Windows\\System32\\*", PermissionLevel.DENY, "System32 files"),
            ])

        self.categories["shell"] = shell
        self.categories["filesystem"] = fs

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_config(self, config: dict) -> None:
        """Load permission rules from configuration dict."""
        for cat_name, cat_config in config.items():
            category = self.categories.get(cat_name)
            if category is None:
                category = PermissionCategory(name=cat_name)
                self.categories[cat_name] = category

            # Override default level
            if "default" in cat_config:
                category.default_level = PermissionLevel.from_str(cat_config["default"])

            # Add rules
            for rule_config in cat_config.get("rules", []):
                rule = PermissionRule(
                    pattern=rule_config["pattern"],
                    level=PermissionLevel.from_str(rule_config.get("level", "prompt")),
                    description=rule_config.get("description", ""),
                    is_regex=rule_config.get("is_regex", False),
                )
                category.rules.append(rule)

    # ------------------------------------------------------------------
    # Permission checks
    # ------------------------------------------------------------------

    def check_shell(self, command: str) -> PermissionLevel:
        """Check permission level for a shell command."""
        cat = self.categories.get("shell")
        if cat is None:
            return PermissionLevel.ALLOW
        return cat.check(command)

    def check_file_write(self, path: str) -> PermissionLevel:
        """Check permission level for writing to a file path."""
        cat = self.categories.get("filesystem")
        if cat is None:
            return PermissionLevel.ALLOW
        return cat.check(path)

    def check_file_read(self, path: str) -> PermissionLevel:
        """Check permission level for reading a file path."""
        cat = self.categories.get("filesystem")
        if cat is None:
            return PermissionLevel.ALLOW
        # Reading is generally allowed; use same rules but default to ALLOW
        level = cat.check(path)
        # Downgrade DENY to PROMPT for reads (reading is less dangerous)
        if level == PermissionLevel.DENY:
            return PermissionLevel.PROMPT
        return level

    # ------------------------------------------------------------------
    # Approval flow
    # ------------------------------------------------------------------

    def request_approval(
        self,
        category: str,
        description: str,
        target: str,
    ) -> tuple[bool, str]:
        """Request user approval for an operation.

        Returns (approved: bool, reason: str).
        If no callback is configured, the operation is denied by default.
        """
        if self._approval_callback:
            return self._approval_callback(category, description, target)
        logger.warning(
            "No approval callback configured; denying %s: %s (%s)",
            category, target, description,
        )
        return False, "No approval mechanism configured."

    def authorize_shell(self, command: str) -> tuple[bool, str, PermissionLevel]:
        """Authorize a shell command. Returns (approved, reason, level)."""
        level = self.check_shell(command)
        if level == PermissionLevel.DENY:
            return False, f"Command is explicitly denied by security policy: {command[:200]}", level
        if level == PermissionLevel.PROMPT:
            return self.request_approval("shell", "Shell command execution", command[:200]) + (level,)
        return True, "", level

    def authorize_file_write(self, path: str) -> tuple[bool, str, PermissionLevel]:
        """Authorize a file write operation. Returns (approved, reason, level)."""
        level = self.check_file_write(path)
        if level == PermissionLevel.DENY:
            return False, f"Writing to this path is explicitly denied: {path}", level
        if level == PermissionLevel.PROMPT:
            return self.request_approval("filesystem", "File write operation", path) + (level,)
        return True, "", level

    def authorize_file_read(self, path: str) -> tuple[bool, str, PermissionLevel]:
        """Authorize a file read operation. Returns (approved, reason, level)."""
        level = self.check_file_read(path)
        if level == PermissionLevel.DENY:
            return False, f"Reading this path is explicitly denied: {path}", level
        if level == PermissionLevel.PROMPT:
            return self.request_approval("filesystem", "File read operation", path) + (level,)
        return True, "", level

    def to_dict(self) -> dict:
        """Serialize permissions to a dict for config/display."""
        result = {}
        for name, cat in self.categories.items():
            result[name] = {
                "default": cat.default_level.value,
                "rules": [
                    {
                        "pattern": r.pattern,
                        "level": r.level.value,
                        "description": r.description,
                        "is_regex": r.is_regex,
                    }
                    for r in cat.rules
                ],
            }
        return result