"""Tests for /stop task cancellation."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_loop():
    """Create a minimal AgentLoop with mocked dependencies."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    workspace = MagicMock()
    workspace.__truediv__ = MagicMock(return_value=MagicMock())

    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.SubagentManager") as mock_sub_mgr:
        mock_sub_mgr.return_value.cancel_by_session = AsyncMock(return_value=0)
        loop = AgentLoop(bus=bus, provider=provider, workspace=workspace)
    return loop, bus


def _make_configured_loop(tmp_path, models: list[str] | None = None):
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from nanobot.config.schema import Config
    from nanobot.session.manager import SessionManager

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "anthropic/claude-opus-4-5"
    config = Config()
    config.agents.defaults.model = "anthropic/claude-opus-4-5"
    config.agents.defaults.available_models = models or ["openai/gpt-4.1"]
    config.providers.anthropic.api_key = "anthropic-key"
    config.providers.openai.api_key = "openai-key"

    with patch("nanobot.agent.loop.load_config", return_value=config), \
         patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SubagentManager") as mock_sub_mgr:
        mock_sub_mgr.return_value.cancel_by_session = AsyncMock(return_value=0)
        loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=tmp_path,
            model=config.agents.defaults.model,
            session_manager=SessionManager(tmp_path),
        )
    return loop, bus, config


def test_agent_loop_forwards_tool_timeout_to_subagent_manager():
    """AgentLoop should pass tool_timeout through to SubagentManager."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    workspace = MagicMock()
    workspace.__truediv__ = MagicMock(return_value=MagicMock())

    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.SubagentManager") as mock_sub_mgr:
        mock_sub_mgr.return_value.cancel_by_session = AsyncMock(return_value=0)
        AgentLoop(bus=bus, provider=provider, workspace=workspace, tool_timeout=77)

    assert mock_sub_mgr.call_args.kwargs["tool_timeout"] == 77


def test_make_archive_provider_allows_bedrock_model_without_api_key():
    from nanobot.config.schema import Config

    loop, _ = _make_loop()
    config = Config()

    provider = loop._make_archive_provider(config, "bedrock/anthropic.claude-3-5-sonnet-v2:0")

    assert provider is not None


class TestHandleStop:
    @pytest.mark.asyncio
    async def test_stop_no_active_task(self):
        from nanobot.bus.events import InboundMessage

        loop, bus = _make_loop()
        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/stop")
        await loop._handle_stop(msg)
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert "No active task" in out.content

    @pytest.mark.asyncio
    async def test_stop_cancels_active_task(self):
        from nanobot.bus.events import InboundMessage

        loop, bus = _make_loop()
        cancelled = asyncio.Event()

        async def slow_task():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = asyncio.create_task(slow_task())
        await asyncio.sleep(0)
        loop._active_tasks["test:c1"] = [task]

        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/stop")
        await loop._handle_stop(msg)

        assert cancelled.is_set()
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert "stopped" in out.content.lower()

    @pytest.mark.asyncio
    async def test_stop_cancels_multiple_tasks(self):
        from nanobot.bus.events import InboundMessage

        loop, bus = _make_loop()
        events = [asyncio.Event(), asyncio.Event()]

        async def slow(idx):
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                events[idx].set()
                raise

        tasks = [asyncio.create_task(slow(i)) for i in range(2)]
        await asyncio.sleep(0)
        loop._active_tasks["test:c1"] = tasks

        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/stop")
        await loop._handle_stop(msg)

        assert all(e.is_set() for e in events)
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert "2 task" in out.content


class TestDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_processes_and_publishes(self):
        from nanobot.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop()
        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="hello")
        loop._process_message = AsyncMock(
            return_value=OutboundMessage(channel="test", chat_id="c1", content="hi")
        )
        await loop._dispatch(msg)
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert out.content == "hi"

    @pytest.mark.asyncio
    async def test_processing_lock_serializes(self):
        from nanobot.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop()
        order = []

        async def mock_process(m, **kwargs):
            order.append(f"start-{m.content}")
            await asyncio.sleep(0.05)
            order.append(f"end-{m.content}")
            return OutboundMessage(channel="test", chat_id="c1", content=m.content)

        loop._process_message = mock_process
        msg1 = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="a")
        msg2 = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="b")

        t1 = asyncio.create_task(loop._dispatch(msg1))
        t2 = asyncio.create_task(loop._dispatch(msg2))
        await asyncio.gather(t1, t2)
        assert order == ["start-a", "end-a", "start-b", "end-b"]


class TestModelCommand:
    @pytest.mark.asyncio
    async def test_model_lists_current_and_available(self, tmp_path):
        from nanobot.bus.events import InboundMessage

        loop, _, _ = _make_configured_loop(tmp_path, models=["openai/gpt-4.1", "deepseek/deepseek-chat"])

        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/model")
        response = await loop._process_message(msg)

        assert response is not None
        assert "Current model: anthropic/claude-opus-4-5" in response.content
        assert "openai/gpt-4.1" in response.content
        assert "deepseek/deepseek-chat" in response.content

    @pytest.mark.asyncio
    async def test_model_switches_session_metadata(self, tmp_path):
        from nanobot.bus.events import InboundMessage

        loop, _, _ = _make_configured_loop(tmp_path)

        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/model openai/gpt-4.1")
        response = await loop._process_message(msg)
        session = loop.sessions.get_or_create("cli:c1")

        assert response is not None
        assert "Switched model to openai/gpt-4.1" in response.content
        assert session.metadata["model"] == "openai/gpt-4.1"

    @pytest.mark.asyncio
    async def test_model_rejects_unconfigured_target(self, tmp_path):
        from nanobot.bus.events import InboundMessage

        loop, _, _ = _make_configured_loop(tmp_path)

        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/model gemini/gemini-2.0-flash")
        response = await loop._process_message(msg)

        assert response is not None
        assert "Model not configured" in response.content

    @pytest.mark.asyncio
    async def test_session_model_used_for_provider_calls(self, tmp_path):
        from nanobot.bus.events import InboundMessage
        from nanobot.providers.base import LLMResponse

        loop, _, config = _make_configured_loop(tmp_path)
        session = loop.sessions.get_or_create("cli:c1")
        session.metadata["model"] = "openai/gpt-4.1"

        provider = MagicMock()
        provider.chat = AsyncMock(return_value=LLMResponse(content="ok", tool_calls=[]))
        loop.tools.get_definitions = MagicMock(return_value=[])

        with patch("nanobot.agent.loop.create_provider", return_value=provider) as mock_create_provider:
            msg = InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="hello")
            response = await loop._process_message(msg)

        assert response is not None
        assert response.content == "ok"
        mock_create_provider.assert_called_with(config, "openai/gpt-4.1")
        provider.chat.assert_awaited()
        assert provider.chat.await_args.kwargs["model"] == "openai/gpt-4.1"


class TestSubagentCancellation:
    @pytest.mark.asyncio
    async def test_cancel_by_session(self):
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        mgr = SubagentManager(provider=provider, workspace=MagicMock(), bus=bus)

        cancelled = asyncio.Event()

        async def slow():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = asyncio.create_task(slow())
        await asyncio.sleep(0)
        mgr._running_tasks["sub-1"] = task
        mgr._session_tasks["test:c1"] = {"sub-1"}

        count = await mgr.cancel_by_session("test:c1")
        assert count == 1
        assert cancelled.is_set()

    @pytest.mark.asyncio
    async def test_cancel_by_session_no_tasks(self):
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        mgr = SubagentManager(provider=provider, workspace=MagicMock(), bus=bus)
        assert await mgr.cancel_by_session("nonexistent") == 0
