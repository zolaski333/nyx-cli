"""
Nyx — Persistent memory with automatic context summarization.

Saves conversations to disk, tracks token usage, and automatically
summarises old context when the window gets too large.

Performance optimizations:
- Buffered disk writes: saves are debounced (batched) to avoid I/O on every operation
- Lazy loading: conversations are loaded on-demand, not all at startup
- Thread-safe save scheduling
"""
from __future__ import annotations

import json
import os
import threading
import time
import hashlib
import math
import re
import urllib.request
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nyx.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)

if os.name == "nt":
    DEFAULT_MEMORY_DIR = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "nyx" / "memory"
else:
    DEFAULT_MEMORY_DIR = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "nyx" / "memory"


# ---------------------------------------------------------------------------
# Embedding and BM25 Search Utilities (Zero-Dependency)
# ---------------------------------------------------------------------------

def _get_embedding(text: str, provider: str, api_key: str) -> list[float] | None:
    """Fetch embeddings from OpenAI or OpenRouter APIs using standard urllib."""
    if not api_key or provider not in ("openai", "openrouter"):
        return None
    url = "https://api.openai.com/v1/embeddings" if provider == "openai" else "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "input": text,
        "model": "text-embedding-3-small" if provider == "openai" else "openai/text-embedding-3-small"
    }
    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=8) as response:
            res = json.loads(response.read().decode("utf-8"))
            return res["data"][0]["embedding"]
    except Exception as e:
        logger.debug("Failed to get embedding: %s", e)
        return None


def _text_checksum(text: str) -> str:
    """Compute a stable checksum for embedding caching."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calculate the cosine similarity between two vectors."""
    if len(v1) != len(v2) or not v1:
        return 0.0
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    return dot_product / (norm_v1 * norm_v2) if norm_v1 > 0 and norm_v2 > 0 else 0.0


def _tokenize(text: str) -> list[str]:
    """Tokenize a string into words for lexical search."""
    return re.findall(r"\w+", text.lower())


class BM25:
    """A lightweight BM25 implementation for local lexical fallback search."""
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avg_doc_len = sum(len(doc) for doc in corpus) / max(1, self.corpus_size)
        self.doc_lens = [len(doc) for doc in corpus]
        self.df = {}
        for doc in corpus:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1
        self.idf = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
        self.doc_term_freqs = []
        for doc in corpus:
            tf = {}
            for term in doc:
                tf[term] = tf.get(term, 0) + 1
            self.doc_term_freqs.append(tf)

    def score(self, query: list[str], doc_idx: int) -> float:
        tf = self.doc_term_freqs[doc_idx]
        doc_len = self.doc_lens[doc_idx]
        score = 0.0
        for term in query:
            if term not in tf:
                continue
            freq = tf[term]
            num = freq * (self.k1 + 1)
            den = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += self.idf.get(term, 0.0) * (num / den)
        return score



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
    """Accurate token estimation with tiktoken fallback, default to division by 4 for backwards compatibility."""
    if not text:
        return 0
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text, disallowed_special=()))
    except ImportError:
        # Backward-compatible default: 4 characters per token
        return max(1, len(text) // 4) if text else 0




class MemoryManager:
    """
    Manages conversation persistence, context window, and summarisation.

    Features:
    - Auto-saves conversations to disk as JSON (buffered/debounced)
    - Tracks token usage per conversation
    - Automatically summarises old entries when context exceeds max_tokens
    - Multiple conversation support (switch between them)
    - Lazy loading: conversations are loaded on first access
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

        # Buffered save mechanism
        self._save_lock = threading.Lock()
        self._save_timer: threading.Timer | None = None
        self._save_delay = 2.0  # Debounce delay in seconds
        self._dirty_conversations: set[str] = set()
        self._dirty_index = False

        # Load existing conversations index (lazy — only loads metadata)
        self._load_index()

        # Load embeddings cache
        self._embeddings_cache_path = self._dir / "embeddings.json"
        self._embeddings_cache: dict[str, list[float]] = {}
        self._load_embeddings_cache()

        # Create a new conversation if none loaded
        if not self._current_id:
            self.new_conversation()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _index_path(self) -> Path:
        return self._dir / "index.json"

    def _load_index(self) -> None:
        """Load the conversation index from disk (lazy — only metadata)."""
        idx = self._index_path()
        if idx.exists():
            try:
                with idx.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                conv_ids = data.get("conversations", [])
                current = data.get("current", "")
                # Only load the current conversation fully; others are lazy-loaded
                if current and current in conv_ids:
                    self._load_conversation(current)
                for cid in conv_ids:
                    if cid != current and cid not in self._conversations:
                        # Create a placeholder with just the ID
                        self._conversations[cid] = Conversation(id=cid)
                if current and current in self._conversations:
                    self._current_id = current
            except (json.JSONDecodeError, KeyError, OSError):
                pass

    def _load_conversation(self, conv_id: str) -> bool:
        """Fully load a single conversation from disk."""
        conv_path = self._conv_path(conv_id)
        if conv_path.exists():
            try:
                with conv_path.open("r", encoding="utf-8") as f:
                    self._conversations[conv_id] = Conversation.from_dict(json.load(f))
                return True
            except (json.JSONDecodeError, KeyError, OSError):
                pass
        return False

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
    # Buffered save mechanism
    # ------------------------------------------------------------------

    def _schedule_save(self, conv_id: str | None = None) -> None:
        """Schedule a buffered save. Multiple calls within the debounce window
        are coalesced into a single disk write."""
        with self._save_lock:
            if conv_id:
                self._dirty_conversations.add(conv_id)
            self._dirty_index = True

            # Cancel any pending timer
            if self._save_timer is not None:
                self._save_timer.cancel()
                self._save_timer = None

            # Schedule a new timer
            self._save_timer = threading.Timer(self._save_delay, self._flush_saves)
            self._save_timer.daemon = True
            self._save_timer.start()

    def _flush_saves(self) -> None:
        """Flush all pending saves to disk."""
        with self._save_lock:
            dirty_convs = self._dirty_conversations.copy()
            dirty_idx = self._dirty_index
            self._dirty_conversations.clear()
            self._dirty_index = False
            self._save_timer = None

        # Write dirty conversations
        for cid in dirty_convs:
            conv = self._conversations.get(cid)
            if conv and conv.entries:  # Only save non-empty conversations
                path = self._conv_path(cid)
                try:
                    with path.open("w", encoding="utf-8") as f:
                        json.dump(conv.to_dict(), f, ensure_ascii=False, indent=2)
                except OSError:
                    pass

        # Write index if dirty
        if dirty_idx:
            self._save_index()

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
        self._schedule_save(conv_id)
        return conv_id

    @property
    def current(self) -> Conversation | None:
        return self._conversations.get(self._current_id)

    @property
    def conversations(self) -> dict[str, Conversation]:
        """Expose all conversations (lazy-loads any that are placeholders)."""
        for cid, conv in list(self._conversations.items()):
            if not conv.entries and not conv.title.startswith("Conversation "):
                # This is a placeholder — try to load fully
                self._load_conversation(cid)
        return dict(self._conversations)

    def switch_to(self, conv_id: str) -> bool:
        """Switch to an existing conversation."""
        if conv_id in self._conversations:
            # Lazy-load if needed
            conv = self._conversations[conv_id]
            if not conv.entries and not conv.title.startswith("Conversation "):
                self._load_conversation(conv_id)
            self._current_id = conv_id
            self._schedule_save()
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
        self._schedule_save()
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
        self._schedule_save(conv.id)

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
            role = entry.role
            if role == "memory":
                role = "system"
            messages.append({"role": role, "content": entry.content})

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

        # Only replace entries if we got a valid summary back
        if summary_text and summary_text.strip():
            conv.summary = summary_text
            conv.entries = to_keep
            conv.total_tokens = sum(e.token_count for e in to_keep)
            self._schedule_save(conv.id)

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

    def _notes_path(self) -> Path:
        return self._dir / "notes.json"

    def _save_note(self, content: str, tags: str = "") -> None:
        """Save a note to the dedicated notes store (persistent across conversations)."""
        notes = self._load_notes()
        notes.append({
            "content": content,
            "tags": tags,
            "timestamp": time.time(),
        })
        with self._notes_path().open("w", encoding="utf-8") as f:
            json.dump(notes, f, ensure_ascii=False, indent=2)

    def _load_notes(self) -> list[dict[str, Any]]:
        """Load saved notes from the notes store."""
        path = self._notes_path()
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return []

    def save_all(self) -> None:
        """Save all conversations to disk immediately (flush)."""
        # Cancel any pending timer and flush immediately
        with self._save_lock:
            if self._save_timer is not None:
                self._save_timer.cancel()
                self._save_timer = None
            self._dirty_conversations.update(self._conversations.keys())
            self._dirty_index = True

        self._flush_saves()

    def clear_current(self) -> None:
        """Clear the current conversation (keep it saved)."""
        conv = self.current
        if conv:
            conv.entries.clear()
            conv.total_tokens = 0
            conv.summary = ""
            self._schedule_save(conv.id)

    # ------------------------------------------------------------------
    # Embeddings Cache & Semantic Recall
    # ------------------------------------------------------------------

    def _load_embeddings_cache(self) -> None:
        """Load cached embeddings from disk."""
        if self._embeddings_cache_path.exists():
            try:
                with self._embeddings_cache_path.open("r", encoding="utf-8") as f:
                    self._embeddings_cache = json.load(f)
            except Exception:
                self._embeddings_cache = {}

    def _save_embeddings_cache(self) -> None:
        """Save cached embeddings to disk."""
        try:
            with self._embeddings_cache_path.open("w", encoding="utf-8") as f:
                json.dump(self._embeddings_cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def recall_memories(self, query: str, limit: int = 10, config: Any = None) -> list[dict[str, Any]]:
        """Recall relevant past memories using hybrid Semantic Embeddings and BM25 local fallback.

        Each memory is returned as a dict with: type, title, content, score, and metadata.
        """
        # 1. Collect all documents from conversations and notes
        docs = []

        # Load notes
        notes = self._load_notes()
        for note in notes:
            docs.append({
                "type": "note",
                "title": f"Saved Note (tags: {note.get('tags', 'none')})",
                "content": note["content"],
                "metadata": {"tags": note.get("tags", ""), "timestamp": note.get("timestamp", 0.0)}
            })

        # Load conversations (lazy-load all)
        for cid in list(self._conversations.keys()):
            # Placeholders logic: trigger lazy loading
            self.conversations

        for cid, conv in self._conversations.items():
            # Add conversation summary if present
            if conv.summary:
                docs.append({
                    "type": "conversation_summary",
                    "title": f"Summary: {conv.title}",
                    "content": conv.summary,
                    "metadata": {"conv_id": conv.id, "title": conv.title, "updated_at": conv.updated_at}
                })

            # Add messages
            for entry in conv.entries:
                if len(entry.content) > 30:  # ignore very short messages
                    docs.append({
                        "type": "message",
                        "title": f"Message in '{conv.title}' ({entry.role})",
                        "content": entry.content,
                        "metadata": {"conv_id": conv.id, "title": conv.title, "role": entry.role, "timestamp": entry.timestamp}
                    })

        if not docs:
            return []

        # 2. Try to get embedding for query
        query_emb = None
        provider = None
        api_key = None
        if config:
            provider = config.provider
            api_key = config.get_api_key()
        elif self._provider:
            provider = getattr(self._provider, "provider", None) or getattr(self._provider, "name", None)
            api_key = getattr(self._provider, "api_key", None)

        if provider and api_key:
            query_emb = _get_embedding(query, provider, api_key)

        # 3. Calculate scores
        results = []
        if query_emb:
            logger.debug("Performing semantic similarity search for query: '%s'", query)
            dirty_cache = False
            for doc in docs:
                txt = doc["content"]
                chk = _text_checksum(txt)
                doc_emb = self._embeddings_cache.get(chk)
                if not doc_emb:
                    doc_emb = _get_embedding(txt, provider, api_key)
                    if doc_emb:
                        self._embeddings_cache[chk] = doc_emb
                        dirty_cache = True

                if doc_emb:
                    score = _cosine_similarity(query_emb, doc_emb)
                else:
                    score = 0.0
                results.append((score, doc))

            if dirty_cache:
                self._save_embeddings_cache()
        else:
            logger.debug("Performing BM25 lexical search fallback for query: '%s'", query)
            corpus = [_tokenize(doc["content"]) for doc in docs]
            bm25 = BM25(corpus)
            query_tokens = _tokenize(query)
            for idx, doc in enumerate(docs):
                score = bm25.score(query_tokens, idx)
                results.append((score, doc))

        # Sort by score descending and return
        results.sort(key=lambda x: x[0], reverse=True)

        min_score = 0.3 if query_emb else 0.01

        return [
            {
                "type": doc["type"],
                "title": doc["title"],
                "content": doc["content"],
                "score": round(score, 4),
                "metadata": doc["metadata"]
            }
            for score, doc in results
            if score >= min_score
        ][:limit]
