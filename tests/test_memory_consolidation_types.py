"""Test MemoryStore.consolidate() handles non-string tool call arguments.

Regression test for https://github.com/HKUDS/nanobot/issues/1042
When memory consolidation receives dict values instead of strings from the LLM
tool call response, it should serialize them to JSON instead of raising TypeError.
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.memory import MemoryStore
from nanobot.providers.base import LLMResponse, ToolCallRequest


def _make_session(message_count: int = 30, memory_window: int = 50):
    """Create a mock session with messages."""
    session = MagicMock()
    session.messages = [
        {"role": "user", "content": f"msg{i}", "timestamp": "2026-01-01 00:00"}
        for i in range(message_count)
    ]
    session.last_consolidated = 0
    return session


def _make_tool_response(daily_entries, memory_update):
    """Create an LLMResponse with a save_memory tool call."""
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCallRequest(
                id="call_1",
                name="save_memory",
                arguments={
                    "daily_entries": daily_entries,
                    "memory_update": memory_update,
                },
            )
        ],
    )


class TestMemoryConsolidationTypeHandling:
    """Test that consolidation handles various argument types correctly."""

    @pytest.mark.asyncio
    async def test_string_arguments_work(self, tmp_path: Path) -> None:
        """Normal case: LLM returns string arguments."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat = AsyncMock(
            return_value=_make_tool_response(
                daily_entries=["[2026-01-01] User discussed testing."],
                memory_update="# Memory\nUser likes testing.",
            )
        )
        session = _make_session(message_count=60)

        result = await store.consolidate(session, provider, "test-model", memory_window=50)

        assert result is True
        today_file = store.daily_file_for(datetime.now().strftime("%Y-%m-%d"))
        assert today_file.exists()
        assert "User discussed testing." in today_file.read_text()
        assert "User likes testing." in store.memory_file.read_text()

    @pytest.mark.asyncio
    async def test_dict_arguments_serialized_to_json(self, tmp_path: Path) -> None:
        """Issue #1042: LLM returns dict/list values — must not raise TypeError."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat = AsyncMock(
            return_value=_make_tool_response(
                daily_entries=[{"timestamp": "2026-01-01", "summary": "User discussed testing."}],
                memory_update={"facts": ["User likes testing"], "topics": ["testing"]},
            )
        )
        session = _make_session(message_count=60)

        result = await store.consolidate(session, provider, "test-model", memory_window=50)

        assert result is True
        today_file = store.daily_file_for(datetime.now().strftime("%Y-%m-%d"))
        history_content = today_file.read_text()
        entry = history_content.strip().split("] ", 1)[1]
        parsed = json.loads(entry)
        assert parsed["summary"] == "User discussed testing."

        memory_content = store.memory_file.read_text()
        parsed_mem = json.loads(memory_content)
        assert "User likes testing" in parsed_mem["facts"]

    @pytest.mark.asyncio
    async def test_string_arguments_as_raw_json(self, tmp_path: Path) -> None:
        """Some providers return arguments as a JSON string instead of parsed dict."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()

        # Simulate arguments being a JSON string (not yet parsed)
        response = LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="save_memory",
                    arguments=json.dumps({
                        "daily_entries": ["[2026-01-01] User discussed testing."],
                        "memory_update": "# Memory\nUser likes testing.",
                    }),
                )
            ],
        )
        provider.chat = AsyncMock(return_value=response)
        session = _make_session(message_count=60)

        result = await store.consolidate(session, provider, "test-model", memory_window=50)

        assert result is True
        today_file = store.daily_file_for(datetime.now().strftime("%Y-%m-%d"))
        assert "User discussed testing." in today_file.read_text()

    @pytest.mark.asyncio
    async def test_no_tool_call_returns_false(self, tmp_path: Path) -> None:
        """When LLM doesn't use the save_memory tool, return False."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat = AsyncMock(
            return_value=LLMResponse(content="I summarized the conversation.", tool_calls=[])
        )
        session = _make_session(message_count=60)

        result = await store.consolidate(session, provider, "test-model", memory_window=50)

        assert result is False
        today_file = store.daily_file_for(datetime.now().strftime("%Y-%m-%d"))
        assert not today_file.exists()

    @pytest.mark.asyncio
    async def test_skips_when_few_messages(self, tmp_path: Path) -> None:
        """Consolidation should be a no-op when messages < keep_count."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        session = _make_session(message_count=10)

        result = await store.consolidate(session, provider, "test-model", memory_window=50)

        assert result is True
        provider.chat.assert_not_called()
