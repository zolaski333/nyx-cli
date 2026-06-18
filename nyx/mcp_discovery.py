"""
Nyx — MCP auto-discovery module.
Scans the project repository for technology signatures and configures relevant MCP servers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def discover_mcp_servers(root: str | Path) -> dict[str, dict[str, Any]]:
    """Scan the project directory for technological signatures and return suggested MCP servers."""
    root = Path(root).resolve()
    suggestions = {}

    # 1. Git detection
    if (root / ".git").exists():
        suggestions["git"] = {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-git"],
            "description": "Git MCP server for repository analysis and commits."
        }

    # 2. SQLite detection
    sqlite_files = []
    # Find any .db, .sqlite, .sqlite3 files in project root (not deep recursive to avoid virtualenvs/node_modules)
    for p in root.glob("*"):
        if p.is_file() and p.suffix.lower() in (".db", ".sqlite", ".sqlite3"):
            sqlite_files.append(p)
            
    if sqlite_files:
        db_path = str(sqlite_files[0])
        suggestions["sqlite"] = {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-sqlite", "--db", db_path],
            "description": f"SQLite MCP server configured for database '{sqlite_files[0].name}'."
        }

    # 3. AWS configuration detection
    aws_config = Path.home() / ".aws" / "config"
    if aws_config.exists() or (root / ".aws").exists():
        suggestions["aws"] = {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-aws"],
            "description": "AWS MCP server for Cloud Control API."
        }

    return suggestions


def run_interactive_discovery(root: str | Path, config_dict: dict[str, Any]) -> dict[str, Any]:
    """Interactively ask the user if they want to enable discovered MCP servers.
    
    Modifies config_dict['mcp_servers'] in-place.
    """
    root = Path(root).resolve()
    discovered = discover_mcp_servers(root)
    if not discovered:
        return config_dict

    mcp_servers = config_dict.setdefault("mcp_servers", {})

    print("\n🔍 Nyx MCP Auto-Discovery:")
    has_new_suggestions = False
    for name, spec in discovered.items():
        if name in mcp_servers:
            continue
        has_new_suggestions = True
        desc = spec.get("description", "")
        print(f"  • Found signature for {name.upper()}: {desc}")
        try:
            response = input(f"    Would you like to enable the {name} MCP server? [y/N]: ").strip().lower()
            if response in ("y", "yes"):
                server_config = {
                    "command": spec["command"],
                    "args": spec["args"]
                }
                mcp_servers[name] = server_config
                print(f"    Added {name} MCP server configuration.")
        except (KeyboardInterrupt, EOFError):
            print("\n    Auto-discovery skipped.")
            break

    if has_new_suggestions:
        print()
    return config_dict
