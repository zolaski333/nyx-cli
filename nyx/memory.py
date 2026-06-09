"""
Nyx — Persistent memory with automatic context summarization.

Saves conversations to disk, tracks token usage, and automatically
summarises old context when the window gets too large.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nyx.providers.base import BaseLLMProvider

DEFAULT_MEMORY_DIR = Path(__file__).resolve().parent.parent / ".nyx_memory"


@dataclass
class ConversationEntry:
    """A single entry in the conversation history."""
    role: str
    content: str
    timestamp: float = 0.0
    token_count: int = 0
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp or time.time(),
            "token_count": self.token_count,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConversationEntry":
        return cls(
            role=d["role"],
            content=d["content"],
            timestamp=d.get("timestamp", 0.0),
            token_count=d.get("token_count", 0),
            summary=d.get("summary", ""),
        )


@dataclass
class Conversation:
    """A full conversation with metadata."""
    id: str
    title: str = "Untitled"
    created_at: float = 0.0
    updated_at: float = 0.0
    entries: list[ConversationEntry] = field(default_factory=list)
    summary: str = ""
    total_tokens: int = 0
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at or time.time(),
            "updated_at": time.time(),
            "entries": [e.to_dict() for e in self.entries],
            "summary": self.summary,
            "total_tokens": self.total_tokens,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Conversation":
        return cls(
            id=d["id"],
            title=d.get("title", "Untitled"),
            created_at=d.get("created_at", 0.0),
            updated_at=d.get("updated_at", 0.0),
            entries=[ConversationEntry.from_dict(e) for e in d.get("entries", [])],
            summary=d.get("summary", ""),
            total_tokens=d.get("total_tokens", 0),
            model=d.get("model", ""),
        )


def _estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars ≈ 1 token)."""
    return len(text) // 4


class MemoryManager:
    """
    Manages conversation persistence, context window, and summarisation.

    Features:
    - Auto-saves conversations to disk as JSON
    - Tracks token usage per conversation
    - Automatically summarises old entries when context exceeds max_tokens
    - Multiple conversation support (switch between them)
    """

    def __init__(
        self,
        memory_dir: str | Path | None = None,
        max_context_tokens: int = 32000,
        provider: BaseLLMProvider | None = None,
        auto_summarise: bool = True,
    ) -> None:
        self._dir = Path(memory_dir) if memory_dir else DEFAULT_MEMORY_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self.max_context_tokens = max_context_tokens
        self._provider = provider
        self.auto_summarise = auto_summarise

        # Current conversation
        self._current_id: str = ""
        self._conversations: dict[str, Conversation] = {}

        # Load existing conversations index
        self._load_index()

        # Create a new conversation if none loaded
        if not self._current_id:
            self.new_conversation()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _index_path(self) -> Path:
        return self._dir / "index.json"

    def _load_index(self) -> None:
        """Load the conversation index from disk."""
        idx = self._index_path()
        if idx.exists():
            try:
                with idx.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                conv_ids = data.get("conversations", [])
                current = data.get("current", "")
                for cid in conv_ids:
                    conv_path = self._conv_path(cid)
                    if conv_path.exists():
                        with conv_path.open("r", encoding="utf-8") as f:
                            self._conversations[cid] = Conversation.from_dict(json.load(f))
                if current and current in self._conversations:
                    self._current_id = current
            except (json.JSONDecodeError, KeyError, OSError):
                pass

    def _save_index(self) -> None:
        """Save the conversation index to disk."""
        data = {
            "conversations": list(self._conversations.keys()),
            "current": self._current_id,
            "updated_at": time.time(),
        }
        with self._index_path().open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _conv_path(self, conv_id: str) -> Path:
        return self._dir / f"{conv_id}.json"

    # ------------------------------------------------------------------
    # Conversation management
    # ------------------------------------------------------------------

    def new_conversation(self, title: str = "") -> str:
        """Create a new conversation and switch to it."""
        import uuid
        conv_id = uuid.uuid4().hex[:12]
        conv = Conversation(
            id=conv_id,
            title=title or f"Conversation {time.strftime('%Y-%m-%d %H:%M')}",
            created_at=time.time(),
            updated_at=time.time(),
        )
        self._conversations[conv_id] = conv
        self._current_id = conv_id
        self._save_index()
        return conv_id

    @property
    def current(self) -> Conversation | None:
        return self._conversations.get(self._current_id)

    def switch_to(self, conv_id: str) -> bool:
        """Switch to an existing conversation."""
        if conv_id in self._conversations:
            self._current_id = conv_id
            self._save_index()
            return True
        return False

    def list_conversations(self) -> list[dict[str, Any]]:
        """Return a summary list of all conversations."""
        return [
            {
                "id": c.id,
                "title": c.title,
                "created_at": c.created_at,
                "updated_at": c.updated_at,
                "entry_count": len(c.entries),
                "total_tokens": c.total_tokens,
                "summary": c.summary[:100] if c.summary else "",
            }
            for c in sorted(
                self._conversations.values(),
                key=lambda x: x.updated_at,
                reverse=True,
            )
        ]

    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation."""
        if conv_id not in self._conversations:
            return False
        del self._conversations[conv_id]
        conv_path = self._conv_path(conv_id)
        if conv_path.exists():
            conv_path.unlink()
        if self._current_id == conv_id:
            self._current_id = next(iter(self._conversations.keys()), "")
            if not self._current_id:
                self.new_conversation()
        self._save_index()
        return True

    # ------------------------------------------------------------------
    # Entry management
    # ------------------------------------------------------------------

    def add_entry(self, role: str, content: str, token_count: int = 0) -> None:
        """Add an entry to the current conversation."""
        conv = self.current
        if not conv:
            return
        entry = ConversationEntry(
            role=role,
            content=content,
            timestamp=time.time(),
            token_count=token_count or _estimate_tokens(content),
        )
        conv.entries.append(entry)
        conv.total_tokens += entry.token_count
        conv.updated_at = time.time()
        self._auto_save()

        # Auto-summarise if context is too large
        if self.auto_summarise and conv.total_tokens > self.max_context_tokens:
            self._summarise_old_entries()

    def get_context_messages(
        self,
        max_tokens: int = 32000,
        include_summary: bool = True,
    ) -> list[dict[str, str]]:
        """
        Build a messages list from the current conversation.
        If include_summary is True and a summary exists, prepend it as a system message.
        """
        conv = self.current
        if not conv:
            return []

        messages: list[dict[str, str]] = []

        # Add summary as system message if available
        if include_summary and conv.summary:
            messages.append({
                "role": "system",
                "content": f"[Previous conversation summary]\n{conv.summary}",
            })

        # Add entries, fitting within token budget
        token_budget = max_tokens - _estimate_tokens(conv.summary) if include_summary else max_tokens
        selected: list[ConversationEntry] = []
        running_tokens = 0

        # Take the most recent entries that fit
        for entry in reversed(conv.entries):
            if running_tokens + entry.token_count > token_budget:
                break
            selected.insert(0, entry)
            running_tokens += entry.token_count

        for entry in selected:
            messages.append({"role": entry.role, "content": entry.content})

        return messages

    # ------------------------------------------------------------------
    # Summarisation
    # ------------------------------------------------------------------

    def _summarise_old_entries(self) -> None:
        """Summarise older entries to free up context window."""
        conv = self.current
        if not conv or len(conv.entries) < 4:
            return

        # Keep the last N entries, summarise everything before
        keep_count = max(4, int(len(conv.entries) * 0.3))
        to_summarise = conv.entries[:-keep_count]
        to_keep = conv.entries[-keep_count:]

        if not to_summarise:
            return

        # Build a summary text from old entries
        old_text = "\n".join(
            f"[{e.role}] {e.content[:500]}"
            for e in to_summarise
        )

        summary_text = self._generate_summary(old_text, conv.summary)

        # Replace entries with summary
        conv.summary = summary_text
        conv.entries = to_keep
        conv.total_tokens = sum(e.token_count for e in to_keep)
        self._auto_save()

    def _generate_summary(self, old_text: str, existing_summary: str) -> str:
        """Use the LLM to generate a summary, or fallback to a simple compression."""
        if self._provider:
            try:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a conversation summariser. Condense the following "
                            "conversation into a concise summary (max 3 sentences) that "
                            "captures the key points, decisions, and context needed to continue."
                        ),
                    },
                ]
                if existing_summary:
                    messages.append({
                        "role": "user",
                        "content": f"Existing summary: {existing_summary}\n\nNew conversation to incorporate:\n{old_text}",
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": f"Summarise this conversation:\n{old_text}",
                    })

                response = self._provider.chat(messages=messages, stream=False)
                if response.content:
                    return response.content.strip()
            except Exception:
                pass

        # Fallback: simple truncation summarisation
        lines = old_text.split("\n")
        if len(lines) > 20:
            return "\n".join(lines[:5]) + "\n...\n" + "\n".join(lines[-5:])
        return old_text[:1000]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _auto_save(self) -> None:
        """Save the current conversation to disk."""
        conv = self.current
        if not conv:
            return
        path = self._conv_path(conv.id)
        with path.open("w", encoding="utf-8") as f:
            json.dump(conv.to_dict(), f, ensure_ascii=False, indent=2)
        self._save_index()

    def save_all(self) -> None:
        """Save all conversations to disk."""
        for conv in self._conversations.values():
            path = self._conv_path(conv.id)
            with path.open("w", encoding="utf-8") as f:
                json.dump(conv.to_dict(), f, ensure_ascii=False, indent=2)
        self._save_index()

    def clear_current(self) -> None:
        """Clear the current conversation (keep it saved)."""
        conv = self.current
        if conv:
            conv.entries.clear()
            conv.total_tokens = 0
            conv.summary = ""
            self._auto_save()