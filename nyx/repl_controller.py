"""Shared REPL command controller for ANSI and Rich frontends."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Protocol, TYPE_CHECKING

from nyx.config import Config, DEFAULT_USER_CONFIG_PATH
from nyx.providers import get_provider

if TYPE_CHECKING:
    from nyx.agent import Agent


class ReplUI(Protocol):
    """UI adapter used by the shared REPL controller."""

    def setup(self, agent: "Agent", config: Config) -> None: ...
    def read_input(self) -> str: ...
    def append_history(self, text: str) -> None: ...
    def show_bye(self) -> None: ...
    def show_help(self) -> None: ...
    def show_context_cleared(self) -> None: ...
    def show_agent_reset(self) -> None: ...
    def show_model(self, model: str) -> None: ...
    def show_model_changed(self, model: str) -> None: ...
    def show_status(self, message: str, *, success: bool = False) -> None: ...
    def show_mode_status(self, mode: str, autonomy: str) -> None: ...
    def show_autonomy_status(self, autonomy: str) -> None: ...
    def show_config_status(self, config: Config, paths: list[tuple[str, Path]]) -> None: ...
    def show_config_saved(self, path: Path) -> None: ...
    def show_config_set(self, key: str, value: Any, path: Path) -> None: ...
    def show_config_error(self, message: str) -> None: ...
    def show_tools(self, agent: "Agent", page: int) -> None: ...
    def show_memory(self, agent: "Agent", page: int) -> None: ...
    def show_conversations(self, agent: "Agent", page: int) -> None: ...
    def show_switched_conversation(self, title: str) -> None: ...
    def show_multiple_conversation_matches(self, matches: list[Any]) -> None: ...
    def show_conversation_not_found(self, conv_id: str) -> None: ...
    def make_on_token(self) -> Callable[[str], None]: ...
    def before_agent_response(self, *, stream: bool) -> None: ...
    def show_agent_result(self, result: str, *, stream: bool) -> None: ...
    def show_error(self, error: Exception) -> None: ...


def project_config_path(project_dir: str | None = None) -> Path:
    root = Path(project_dir or Path.cwd())
    return root / ".nyx" / "config.json"


def parse_config_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def set_nested(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = data
    parts = [p for p in dotted_key.split(".") if p]
    if not parts:
        raise ValueError("Config key cannot be empty.")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def get_nested(data: dict[str, Any], dotted_key: str) -> Any:
    current: Any = data
    for part in [p for p in dotted_key.split(".") if p]:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_key)
        current = current[part]
    return current


def load_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def write_config_file(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def get_paginated_arg(user_input: str, command: str) -> int:
    rest = user_input[len(command):].strip()
    try:
        return int(rest.split()[0]) if rest else 1
    except (ValueError, IndexError):
        return 1


def _apply_config_change(agent: "Agent", config: Config, key: str, value: Any) -> None:
    if key == "model":
        config.model = value
        agent.provider = get_provider(config)
    elif key == "provider":
        config.provider = value
        agent.provider = get_provider(config)
    elif key == "agent.mode":
        agent.switch_mode(value)
    elif key == "agent.autonomy":
        agent.switch_autonomy(value)


def _handle_config_command(agent: "Agent", config: Config, ui: ReplUI, user_input: str) -> bool:
    args_list = user_input[7:].strip().split()
    if not args_list:
        ui.show_config_status(
            config,
            [
                ("User (Global)", DEFAULT_USER_CONFIG_PATH),
                ("Project (Local)", project_config_path(config.project_dir)),
            ],
        )
        return True

    subcmd = args_list[0].lower()
    if subcmd == "save":
        use_global = "--global" in args_list or "-g" in args_list
        path = DEFAULT_USER_CONFIG_PATH if use_global else project_config_path(config.project_dir)
        try:
            data = load_config_file(path)
            data["provider"] = config.provider
            data["model"] = config.model
            set_nested(data, "agent.mode", config.agent_mode)
            set_nested(data, "agent.autonomy", config.agent_autonomy)
            write_config_file(path, data)
            ui.show_config_saved(path)
        except Exception as e:
            ui.show_config_error(f"Failed to save config: {e}")
        return True

    if subcmd == "set":
        use_global = False
        key_val_args = []
        for arg in args_list[1:]:
            if arg in ("--global", "-g"):
                use_global = True
            else:
                key_val_args.append(arg)

        if len(key_val_args) < 2:
            ui.show_config_error("Error: `/config set [--global] <key> <value>` requires a key and a value.")
            return True

        key = key_val_args[0]
        value_text = " ".join(key_val_args[1:])
        path = DEFAULT_USER_CONFIG_PATH if use_global else project_config_path(config.project_dir)
        try:
            data = load_config_file(path)
            value = parse_config_value(value_text)
            set_nested(data, key, value)
            write_config_file(path, data)
            ui.show_config_set(key, value, path)
            _apply_config_change(agent, config, key, value)
        except Exception as e:
            ui.show_config_error(f"Failed to update config: {e}")
        return True

    ui.show_config_error(f"Unknown config subcommand: {subcmd}. Valid options: save, set")
    return True


def handle_repl_command(agent: "Agent", config: Config, ui: ReplUI, user_input: str) -> bool:
    """Handle one slash command. Returns True when the input was consumed."""
    if user_input in {"/help", "/?"}:
        ui.show_help()
        return True
    if user_input == "/clear":
        agent.reset_context()
        ui.show_context_cleared()
        return True
    if user_input == "/reset":
        agent.shutdown()
        agent.reset_context()
        ui.show_agent_reset()
        return True
    if user_input == "/model":
        ui.show_model(config.model)
        return True
    if user_input.startswith("/model "):
        config.model = user_input[7:].strip()
        agent.provider = get_provider(config)
        ui.show_model_changed(config.model)
        return True
    if user_input.startswith("/mode"):
        rest = user_input[5:].strip()
        if not rest:
            ui.show_mode_status(config.agent_mode, config.agent_autonomy)
        else:
            msg = agent.switch_mode(rest)
            ui.show_status(msg, success="switched" in msg)
        return True
    if user_input.startswith("/autonomy"):
        rest = user_input[9:].strip()
        if not rest:
            ui.show_autonomy_status(config.agent_autonomy)
        else:
            msg = agent.switch_autonomy(rest)
            ui.show_status(msg, success="switched" in msg)
        return True
    if user_input.startswith("/config"):
        return _handle_config_command(agent, config, ui, user_input)
    if user_input.startswith("/tools"):
        ui.show_tools(agent, get_paginated_arg(user_input, "/tools"))
        return True
    if user_input.startswith("/memory"):
        ui.show_memory(agent, get_paginated_arg(user_input, "/memory"))
        return True
    if user_input.startswith("/conversations"):
        ui.show_conversations(agent, get_paginated_arg(user_input, "/conversations"))
        return True
    if user_input.startswith("/switch "):
        conv_id = user_input[8:].strip()
        if agent.memory.switch_to(conv_id):
            conv = agent.memory.current
            agent.load_conversation_history()
            ui.show_switched_conversation(conv.title if conv else conv_id)
            return True

        matches = [c for c in agent.memory.conversations.values() if c.id.startswith(conv_id)]
        if len(matches) == 1:
            agent.memory.switch_to(matches[0].id)
            agent.load_conversation_history()
            ui.show_switched_conversation(matches[0].title)
        elif len(matches) > 1:
            ui.show_multiple_conversation_matches(matches)
        else:
            ui.show_conversation_not_found(conv_id)
        return True
    return False


def run_interactive_repl(agent: "Agent", config: Config, ui: ReplUI) -> None:
    """Run the shared interactive REPL loop."""
    ui.setup(agent, config)

    while True:
        try:
            user_input = ui.read_input().strip()
        except (EOFError, KeyboardInterrupt):
            ui.show_bye()
            agent.memory.save_all()
            break

        if not user_input:
            continue

        ui.append_history(user_input)

        if user_input in {"/exit", "/quit", "/q"}:
            ui.show_bye()
            agent.memory.save_all()
            break

        if handle_repl_command(agent, config, ui, user_input):
            continue

        ui.before_agent_response(stream=config.stream)
        on_token = ui.make_on_token()

        try:
            result = agent.run(user_input, on_token=on_token)
            ui.show_agent_result(result, stream=config.stream)
        except Exception as e:
            ui.show_error(e)
