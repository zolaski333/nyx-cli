"""Tests for Nyx memory system."""
from __future__ import annotations

import tempfile
from pathlib import Path

from nyx.memory import MemoryManager, Conversation, ConversationEntry, _estimate_tokens


class TestMemoryManager:
    """Test the persistent memory system."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.memory = MemoryManager(memory_dir=self.tmpdir, auto_summarise=False)

    def test_new_conversation(self):
        conv_id = self.memory.new_conversation("Test conversation")
        assert self.memory.current is not None
        assert self.memory.current.id == conv_id
        assert self.memory.current.title == "Test conversation"

    def test_add_entry(self):
        self.memory.add_entry("user", "hello")
        assert self.memory.current is not None
        assert len(self.memory.current.entries) == 1
        assert self.memory.current.entries[0].role == "user"
        assert self.memory.current.entries[0].content == "hello"

    def test_list_conversations(self):
        # MemoryManager creates a default conversation on init
        # So we should have 1 + 2 new = 3
        self.memory.new_conversation("Conv 1")
        self.memory.new_conversation("Conv 2")
        convs = self.memory.list_conversations()
        assert len(convs) == 3

    def test_switch_conversation(self):
        id1 = self.memory.new_conversation("First")
        id2 = self.memory.new_conversation("Second")
        assert self.memory.current.id == id2
        assert self.memory.switch_to(id1) is True
        assert self.memory.current.id == id1

    def test_switch_nonexistent(self):
        assert self.memory.switch_to("nonexistent") is False

    def test_delete_conversation(self):
        id1 = self.memory.new_conversation("To delete")
        assert self.memory.delete_conversation(id1) is True
        assert self.memory.current is not None  # Should auto-create new

    def test_delete_nonexistent(self):
        assert self.memory.delete_conversation("nonexistent") is False

    def test_clear_current(self):
        self.memory.add_entry("user", "hello")
        self.memory.clear_current()
        assert self.memory.current is not None
        assert len(self.memory.current.entries) == 0
        assert self.memory.current.total_tokens == 0

    def test_get_context_messages(self):
        self.memory.add_entry("user", "hello")
        self.memory.add_entry("assistant", "hi there")
        messages = self.memory.get_context_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_get_context_messages_with_summary(self):
        self.memory.add_entry("user", "hello")
        conv = self.memory.current
        assert conv is not None
        conv.summary = "Previous conversation summary"
        messages = self.memory.get_context_messages(include_summary=True)
        assert any("Previous conversation summary" in m.get("content", "") for m in messages)

    def test_save_and_load_persistence(self):
        self.memory.add_entry("user", "persistent message")
        conv_id = self.memory.current.id
        self.memory.save_all()

        # Create new memory manager pointing to same dir
        memory2 = MemoryManager(memory_dir=self.tmpdir, auto_summarise=False)
        assert memory2.switch_to(conv_id)
        assert memory2.current is not None
        assert len(memory2.current.entries) == 1
        assert memory2.current.entries[0].content == "persistent message"

    def test_estimate_tokens(self):
        assert _estimate_tokens("hello") == 1  # 4 chars = 1 token
        assert _estimate_tokens("a" * 100) == 25


class TestConversation:
    """Test Conversation serialization."""

    def test_to_dict_and_from_dict(self):
        conv = Conversation(
            id="test123",
            title="Test",
            entries=[
                ConversationEntry(role="user", content="hello"),
                ConversationEntry(role="assistant", content="world"),
            ],
        )
        d = conv.to_dict()
        assert d["id"] == "test123"
        assert len(d["entries"]) == 2

        restored = Conversation.from_dict(d)
        assert restored.id == "test123"
        assert len(restored.entries) == 2
        assert restored.entries[0].content == "hello"
        assert restored.entries[1].content == "world"