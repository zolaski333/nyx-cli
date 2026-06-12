"""
Subagent system — spawns child agents that report back.

A subagent is a self-contained LLM conversation with its own system prompt.
It can be given a task, run independently, and return results.
Useful for parallel research, code generation, or complex subtasks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

from nyx.config import Config
from nyx.providers.base import BaseLLMProvider, ToolCall
from nyx.providers import get_provider

if TYPE_CHECKING:
    from nyx.tools import ToolContext


@dataclass
class SubagentResult:
    """Result from a subagent execution."""
    task: str
    output: str
    tokens_used: int = 0
    error: str | None = None


class Subagent:
    """A subagent that can execute a task independently."""

    def __init__(
        self,
        name: str,
        system_prompt: str = "",
        provider: BaseLLMProvider | None = None,
        config: Config | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.5,
        tools: list | None = None,
        tool_executor: Callable[[ToolCall], str] | None = None,
        context: ToolContext | None = None,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt or (
            "You are a focused subagent. Complete the assigned task concisely "
            "and return only the result. Do not ask questions."
        )
        self._provider = provider
        self._config = config
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._tools = tools  # Controlled tool subset (None = no tools)
        self._tool_executor = tool_executor
        self.context = context

    def execute(
        self,
        task: str,
        context: str = "",
        tools: list | None = None,
    ) -> SubagentResult:
        """Run the subagent with a given task.

        Args:
            task: The task description for the subagent.
            context: Optional context to prepend.
            tools: Optional tool list override. If None, uses self._tools.

        Returns:
            SubagentResult with the output or error.
        """
        provider = self._provider
        if not provider and self._config:
            try:
                provider = get_provider(self._config)
            except Exception:
                pass
        if not provider:
            return SubagentResult(
                task=task,
                output="",
                error="No LLM provider configured for subagent.",
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}"})
        messages.append({"role": "user", "content": task})

        # Use provided tools, or instance tools, or none
        effective_tools = tools if tools is not None else self._tools

        # Maximum depth of tool calls
        max_steps = 10
        total_tokens = 0
        use_anthropic_format = self._config.provider == "anthropic" if self._config else False

        try:
            for step in range(max_steps):
                response = provider.chat(
                    messages=messages,
                    tools=effective_tools or None,
                    stream=False,
                )
                
                if response.usage:
                    total_tokens += response.usage.get("total_tokens", 0) or response.usage.get("prompt_tokens", 0) + response.usage.get("completion_tokens", 0)
                
                # Append assistant response
                content_str = response.content or ""
                assistant_msg = {
                    "role": "assistant",
                    "content": content_str,
                }
                if response.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                        for tc in response.tool_calls
                    ]
                messages.append(assistant_msg)

                # If no tool calls, this is the final answer!
                if not response.tool_calls or not effective_tools:
                    return SubagentResult(
                        task=task,
                        output=response.content,
                        tokens_used=total_tokens,
                    )

                # Execute all tool calls in this turn
                for tc in response.tool_calls:
                    if self.context:
                        from nyx.tools import execute_tool
                        tool_result = execute_tool(tc, self.context)
                    else:
                        tool_result = self._execute_tool_call(tc, effective_tools)

                    # Append tool result to messages
                    if use_anthropic_format:
                        messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tc.id,
                                    "content": [{"type": "text", "text": tool_result}],
                                }
                            ],
                        })
                    else:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.name,
                            "content": tool_result,
                        })
                
                # If finish tool was called, break early and return its result
                finish_calls = [tc for tc in response.tool_calls if tc.name == "finish"]
                if finish_calls:
                    finish_args = finish_calls[0].arguments
                    summary = finish_args.get("summary", "")
                    result = finish_args.get("result", "")
                    output_val = f"Summary: {summary}\nResult: {result}" if (summary or result) else (response.content or "Task completed.")
                    return SubagentResult(
                        task=task,
                        output=output_val,
                        tokens_used=total_tokens,
                    )

            # If we reached max_steps, return the last response content
            return SubagentResult(
                task=task,
                output=messages[-1].get("content") or "Max reasoning steps reached.",
                tokens_used=total_tokens,
            )

        except Exception as e:
            return SubagentResult(task=task, output="", error=str(e), tokens_used=total_tokens)

    def _execute_tool_call(self, tc, tools: list) -> str:
        """Execute a single tool call for the subagent.

        This is a simplified execution — subagents get read-only + controlled tools.
        """
        if self._tool_executor:
            return self._tool_executor(tc)

        import subprocess
        from pathlib import Path

        name = tc.name
        args = tc.arguments

        try:
            if name == "read_file":
                path = args.get("path", "")
                project_dir = self._config.project_dir if self._config and self._config.project_dir else None
                resolved = Path(project_dir) / path if project_dir else Path(path)
                if resolved.exists():
                    return resolved.read_text(encoding="utf-8", errors="ignore")[:5000]
                return f"File not found: {path}"

            if name == "list_files":
                path = args.get("path", ".")
                recursive = args.get("recursive", False)
                project_dir = self._config.project_dir if self._config and self._config.project_dir else None
                resolved = Path(project_dir) / path if project_dir else Path(path)
                if not resolved.exists() or not resolved.is_dir():
                    return f"Directory not found: {path}"
                from nyx.tools import IGNORED_DIRS
                if recursive:
                    files = []
                    for f in resolved.rglob("*"):
                        rel_parts = f.relative_to(resolved).parts
                        if any(part in IGNORED_DIRS for part in rel_parts):
                            continue
                        files.append(str(f.relative_to(resolved)))
                else:
                    files = []
                    for f in resolved.iterdir():
                        if f.name in IGNORED_DIRS:
                            continue
                        files.append(f.name)
                return "\n".join(sorted(files)) if files else "(empty directory)"

            if name == "search_code":
                from nyx.search_code import search_code as _sc
                pattern = args.get("pattern", "")
                if not pattern:
                    return "No pattern provided."
                project_dir = self._config.project_dir if self._config and self._config.project_dir else None
                root = Path(project_dir).resolve() if project_dir else Path(".").resolve()
                result = _sc(pattern, root=root)
                return result.formatted(max_results=20, context_lines=2)

            if name == "repo_map":
                from nyx.repo_map import build_repo_map
                return build_repo_map()

            if name == "web_search":
                from nyx.web_search import search_web, format_search_results
                query = args.get("query", "")
                max_results = min(args.get("max_results", 5), 10)
                results = search_web(query, "duckduckgo", max_results)
                return format_search_results(results)

            if name == "web_fetch":
                from nyx.web_search import fetch_page
                return fetch_page(args.get("url", ""), mode=args.get("mode", "clean"))

            if name == "memory_recall":
                return "[Subagent] Memory recall not available in subagent context."

            if name == "execute_command":
                command = args.get("command", "").strip()
                timeout = args.get("timeout", 30)
                if not command:
                    return "Empty command."
                # Resolve directory
                cwd = self._config.project_dir if self._config and self._config.project_dir else None
                env = os.environ.copy()
                search_dir = cwd if cwd else "."
                bin_dirs = []
                for venv_name in (".venv", "venv", "env", ".env"):
                    venv_bin = Path(search_dir) / venv_name / ("Scripts" if os.name == "nt" else "bin")
                    if venv_bin.exists():
                        bin_dirs.append(str(venv_bin))
                        break
                node_bin = Path(search_dir) / "node_modules" / ".bin"
                if node_bin.exists():
                    bin_dirs.append(str(node_bin))
                if bin_dirs:
                    env["PATH"] = os.pathsep.join(bin_dirs) + os.pathsep + env.get("PATH", "")

                proc = subprocess.run(
                    command, shell=True, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env
                )
                out = proc.stdout or ""
                err = proc.stderr or ""
                from nyx.tools import truncate_output
                if proc.returncode != 0:
                    trunc_out = truncate_output(out, max_lines=200, head_lines=50, tail_lines=100)
                    trunc_err = truncate_output(err, max_lines=200, head_lines=50, tail_lines=100)
                    return f"Exit code: {proc.returncode}\nstdout:\n{trunc_out}\nstderr:\n{trunc_err}"
                trunc_out = truncate_output(out, max_lines=200, head_lines=50, tail_lines=100)
                return trunc_out or "(no output)"

            if name == "write_file":
                path = args.get("path", "")
                content = args.get("content", "")
                project_dir = self._config.project_dir if self._config and self._config.project_dir else None
                if project_dir:
                    resolved = Path(project_dir) / path
                else:
                    resolved = Path(path)
                
                # Ensure directories exist
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_text(content, encoding="utf-8")
                return f"File written: {path}"

            if name == "apply_diff":
                path = args.get("path", "")
                diff_text = args.get("content", "")  # sometimes key 'content' is used or 'diff'
                if not diff_text:
                    diff_text = args.get("diff", "")
                
                project_dir = self._config.project_dir if self._config and self._config.project_dir else None
                if project_dir:
                    resolved = Path(project_dir) / path
                else:
                    resolved = Path(path)
                
                # Check format of diff
                from nyx.diff_tool import parse_unified_diff, parse_search_replace, _apply_unified_diff_to_content, _apply_search_replace_to_content
                original = ""
                if resolved.exists():
                    original = resolved.read_text(encoding="utf-8")
                
                # Quick search-replace or unified diff reconstruction
                proposed = None
                if "<<<<<<< SEARCH" in diff_text:
                    proposed = _apply_search_replace_to_content(original, diff_text)
                else:
                    proposed = _apply_unified_diff_to_content(original, diff_text)
                
                if proposed is None:
                    # Fallback to writing content directly if it's full content instead of a diff
                    proposed = diff_text
                
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_text(proposed, encoding="utf-8")
                return f"File written: {path}"

            if name == "append_file":
                path = args.get("path", "")
                content = args.get("content", "")
                project_dir = self._config.project_dir if self._config and self._config.project_dir else None
                if project_dir:
                    resolved = Path(project_dir) / path
                else:
                    resolved = Path(path)
                
                resolved.parent.mkdir(parents=True, exist_ok=True)
                with open(resolved, "a", encoding="utf-8") as f:
                    f.write(content)
                return f"Content appended to: {path}"

            if name == "run_tests":
                from nyx.test_loop import run_tests as _rt
                command = args.get("command", "")
                project_dir = self._config.project_dir if self._config and self._config.project_dir else None
                root = Path(project_dir).resolve() if project_dir else Path(".").resolve()
                result = _rt(command=command if command else None, root=root)
                return result.summary

            if name == "finish":
                return "[Subagent task complete]"

            return f"Unknown tool: {name}"

        except Exception as e:
            return f"Tool '{name}' error: {e}"


class SubagentManager:
    """Manages spawning and tracking subagents."""

    def __init__(self, config: Config | None = None) -> None:
        self._config = config
        self._subagents: dict[str, Subagent] = {}
        self._results: dict[str, list[SubagentResult]] = {}
        self._default_tools: list | None = None  # Controlled tool subset
        self._tool_executor: Callable[[ToolCall], str] | None = None
        self._progress_callback: Callable[[str, int, int], None] | None = None
        self._context: ToolContext | None = None

    def set_progress_callback(self, callback: Callable[[str, int, int], None] | None) -> None:
        """Set a callback for progress updates: (label, current, total)."""
        self._progress_callback = callback

    def set_default_tools(self, tools: list | None) -> None:
        """Set the default tool subset for all spawned subagents."""
        self._default_tools = tools

    def set_tool_executor(self, executor: Callable[[ToolCall], str] | None) -> None:
        """Set the shared tool executor used by spawned subagents."""
        self._tool_executor = executor
        for agent in self._subagents.values():
            agent._tool_executor = executor

    def set_context(self, context: ToolContext | None) -> None:
        """Set the tool execution context for all subagents spawned by this manager."""
        self._context = context

    def spawn(
        self,
        name: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.5,
        tools: list | None = None,
    ) -> Subagent:
        """Create a new subagent by name.

        Args:
            name: Unique name for the subagent.
            system_prompt: Custom system prompt.
            max_tokens: Maximum tokens for responses.
            temperature: LLM temperature.
            tools: Optional tool subset. Falls back to default_tools if None.

        Returns:
            Subagent instance.
        """
        if name in self._subagents:
            return self._subagents[name]
        effective_tools = tools if tools is not None else self._default_tools
        agent = Subagent(
            name=name,
            system_prompt=system_prompt,
            config=self._config,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=effective_tools,
            tool_executor=self._tool_executor,
            context=self._context,
        )
        self._subagents[name] = agent
        return agent

    def run(
        self,
        name: str,
        task: str,
        context: str = "",
    ) -> SubagentResult:
        """Run a subagent (auto-spawn if needed)."""
        agent = self._subagents.get(name) or self.spawn(name)
        if self._progress_callback:
            self._progress_callback(name, 0, 1)
        result = agent.execute(task=task, context=context)
        if self._progress_callback:
            self._progress_callback(name, 1, 1)
        self._results.setdefault(name, []).append(result)
        return result

    def run_parallel(self, tasks: list[tuple[str, str, str]]) -> list[SubagentResult]:
        """Run multiple subagents sequentially with different tasks.
        (True parallel would require asyncio — for now, sequential.)
        """
        results = []
        for name, task, context in tasks:
            result = self.run(name, task, context)
            results.append(result)
        return results

    def get_results(self, name: str) -> list[SubagentResult]:
        return self._results.get(name, [])

    def summarise_all(self) -> str:
        """Return a text summary of all subagent results."""
        parts = []
        for name, results in self._results.items():
            parts.append(f"=== Subagent: {name} ===")
            for i, r in enumerate(results, 1):
                status = "✓" if not r.error else "✗"
                parts.append(f"  Task #{i} [{status}]: {r.task[:80]}")
                if r.error:
                    parts.append(f"    Error: {r.error}")
                else:
                    parts.append(f"    Output: {r.output[:200]}...")
        return "\n".join(parts)

    def clear(self) -> None:
        self._subagents.clear()
        self._results.clear()
