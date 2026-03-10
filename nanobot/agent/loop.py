"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import re
import weakref
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.memory import (
    MemoryGetTool,
    MemoryIndex,
    MemoryRetrievalConfig,
    MemorySearchTool,
)
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.loader import load_config
from nanobot.providers.base import LLMProvider
from nanobot.providers.factory import create_provider
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig, MemoryToolConfig
    from nanobot.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 500

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        archive_model: str | None = None,
        max_iterations: int = 40,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        memory_window: int = 100,
        reasoning_effort: str | None = None,
        brave_api_key: str | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        memory_config: MemoryToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        tool_timeout: int = 30,
    ):
        from nanobot.config.schema import ExecToolConfig
        self.bus = bus
        self.channels_config = channels_config
        self.config = load_config()
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.archive_model = archive_model
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.reasoning_effort = reasoning_effort
        self.brave_api_key = brave_api_key
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.memory_config = memory_config
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self.context = ContextBuilder(workspace)
        self.memory_store = MemoryStore(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry(default_timeout=tool_timeout)
        self._memory_search_tool: MemorySearchTool | None = None
        self._memory_get_tool: MemoryGetTool | None = None
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_effort=reasoning_effort,
            brave_api_key=brave_api_key,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
            tool_timeout=tool_timeout,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._consolidating: set[str] = set()  # Session keys with consolidation in progress
        self._consolidation_counts: dict[str, int] = {}
        self._consolidation_tasks: set[asyncio.Task] = set()  # Strong refs to in-flight tasks
        self._consolidation_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._processing_lock = asyncio.Lock()
        self._register_default_tools()

    def _get_session_model(self, session: Session) -> str:
        """Return the active model for the session."""
        model = session.metadata.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()
        if self.model:
            return self.model
        return self.config.agents.defaults.model

    def _list_model_options(self) -> list[str]:
        """Return selectable models from config."""
        return self.config.get_available_models()

    def _handle_model_command(self, cmd: str, session: Session, msg: InboundMessage) -> OutboundMessage:
        """Handle `/model` listing and switching."""
        available_models = self._list_model_options()
        current_model = self._get_session_model(session)
        parts = msg.content.strip().split(maxsplit=1)

        if len(parts) == 1:
            options = "\n".join(
                f"{'•' if model != current_model else '✓'} {model}"
                for model in available_models
            ) or "(no models configured)"
            content = (
                f"Current model: {current_model}\n\n"
                "Available models:\n"
                f"{options}\n\n"
                "Use `/model <name>` to switch."
            )
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

        target_model = parts[1].strip()
        if not self.config.is_configured_model(target_model):
            content = (
                f"Model not configured: {target_model}\n"
                "Use `/model` to view the configured model list."
            )
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

        try:
            create_provider(self.config, target_model)
        except ValueError as exc:
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=str(exc))

        session.metadata["model"] = target_model
        provider_name = self.config.get_provider_name(target_model) or "auto"
        content = f"Switched model to {target_model} (provider: {provider_name})."
        self._save_direct_turn(session, msg.content, content)
        self.sessions.save(session)
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        memory_cfg = MemoryRetrievalConfig.from_obj(self.memory_config)
        if memory_cfg.enabled:
            try:
                index = MemoryIndex(workspace=self.workspace, cfg=memory_cfg)
                self._memory_get_tool = MemoryGetTool(workspace=self.workspace, cfg=memory_cfg, index=index)
                self.tools.register(self._memory_get_tool)
                self._memory_search_tool = MemorySearchTool(
                    workspace=self.workspace,
                    cfg=memory_cfg,
                    index=index,
                )
                self.tools.register(self._memory_search_tool)
                if not memory_cfg.embedding_model.strip():
                    logger.info("memory_search using keyword fallback: tools.memory.embedding_model is not configured")
            except Exception as e:
                logger.warning("Memory retrieval tools disabled: {}", e)
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            path_append=self.exec_config.path_append,
        ))
        self.tools.register(WebSearchTool(api_key=self.brave_api_key, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except Exception as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        *,
        provider: LLMProvider | None = None,
        model: str | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop. Returns (final_content, tools_used, messages)."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        while iteration < self.max_iterations:
            iteration += 1

            active_provider = provider or self.provider
            active_model = model or self.model

            response = await active_provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=active_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
            )

            if response.has_tool_calls:
                if on_progress:
                    thought = self._strip_think(response.content)
                    if thought:
                        await on_progress(thought)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                clean = self._strip_think(response.content)
                # Don't persist error responses to session history — they can
                # poison the context and cause permanent 400 loops (#1303).
                if response.finish_reason == "error":
                    logger.error("LLM returned error: {}", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                messages = self.context.add_assistant_message(
                    messages, clean, reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if msg.content.strip().lower() == "/stop":
                await self._handle_stop(msg)
            else:
                task = asyncio.create_task(self._dispatch(msg))
                self._active_tasks.setdefault(msg.session_key, []).append(task)
                task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    async def _handle_stop(self, msg: InboundMessage) -> None:
        """Cancel all active tasks and subagents for the session."""
        tasks = self._active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        sub_cancelled = await self.subagents.cancel_by_session(msg.session_key)
        total = cancelled + sub_cancelled
        content = f"⏹ Stopped {total} task(s)." if total else "No active task to stop."
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under the global lock."""
        async with self._processing_lock:
            try:
                response = await self._process_message(msg)
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))

    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            active_model = self._get_session_model(session)
            active_provider = self.provider
            if active_model != self.model:
                active_provider = create_provider(self.config, active_model)
            history = session.get_history(max_messages=self.memory_window)
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
            )
            final_content, _, all_msgs = await self._run_agent_loop(
                messages,
                provider=active_provider,
                model=active_model,
            )
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)
        if self._memory_search_tool:
            await self._memory_search_tool.maybe_sync(force=False)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            old_session = session
            fresh_session = Session(key=session.key)
            self.sessions.save(fresh_session)
            self.sessions.invalidate(session.key)

            async def _archive_previous_session() -> None:
                snapshot = old_session.messages[old_session.last_consolidated:]
                if not snapshot:
                    return
                temp = Session(key=old_session.key)
                temp.messages = list(snapshot)
                ok = await self._consolidate_memory(temp, archive_all=True)
                if not ok:
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content="Memory archival failed for previous session snapshot.",
                    ))

            self._schedule_consolidation(session.key, _archive_previous_session)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="New session started. Archiving previous context in background.",
            )
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new — Start a new conversation\n/model — Show or switch models\n/stop — Stop the current task\n/help — Show available commands")
        if cmd == "/model" or cmd.startswith("/model "):
            return self._handle_model_command(cmd, session, msg)

        pending = session.metadata.get("pending_memory_capture")
        if isinstance(pending, dict):
            pending_result = self._handle_pending_memory_capture(msg, session, pending)
            if pending_result is not None:
                self.sessions.save(session)
                return pending_result

        remember_result = self._handle_immediate_memory_capture(msg, session)
        if remember_result is not None:
            self.sessions.save(session)
            return remember_result

        unconsolidated = len(session.messages) - session.last_consolidated
        if (unconsolidated >= self.memory_window and session.key not in self._consolidating):
            async def _consolidate_session() -> None:
                await self._consolidate_memory(session)

            self._schedule_consolidation(session.key, _consolidate_session)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        active_model = self._get_session_model(session)
        active_provider = self.provider
        if active_model != self.model:
            active_provider = create_provider(self.config, active_model)
        self.subagents.provider = active_provider
        self.subagents.model = active_model

        history = session.get_history(max_messages=self.memory_window)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages,
            provider=active_provider,
            model=active_model,
            on_progress=on_progress or _bus_progress,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    def _mark_consolidation_started(self, session_key: str) -> None:
        self._consolidation_counts[session_key] = self._consolidation_counts.get(session_key, 0) + 1
        self._consolidating.add(session_key)

    def _mark_consolidation_finished(self, session_key: str) -> None:
        remaining = self._consolidation_counts.get(session_key, 0) - 1
        if remaining > 0:
            self._consolidation_counts[session_key] = remaining
            return
        self._consolidation_counts.pop(session_key, None)
        self._consolidating.discard(session_key)

    def _schedule_consolidation(
        self,
        session_key: str,
        job: Callable[[], Awaitable[None]],
    ) -> None:
        lock = self._consolidation_locks.setdefault(session_key, asyncio.Lock())
        self._mark_consolidation_started(session_key)

        async def _runner() -> None:
            try:
                async with lock:
                    await job()
            finally:
                self._mark_consolidation_finished(session_key)
                _task = asyncio.current_task()
                if _task is not None:
                    self._consolidation_tasks.discard(_task)

        _task = asyncio.create_task(_runner())
        self._consolidation_tasks.add(_task)

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool" and isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = []
                    for c in content:
                        if c.get("type") == "text" and isinstance(c.get("text"), str) and c["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                            continue  # Strip runtime context from multimodal messages
                        if (c.get("type") == "image_url"
                                and c.get("image_url", {}).get("url", "").startswith("data:image/")):
                            filtered.append({"type": "text", "text": "[image]"})
                        else:
                            filtered.append(c)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def _consolidate_memory(self, session, archive_all: bool = False) -> bool:
        """Delegate to MemoryStore.consolidate(). Returns True on success."""
        # Use archive_model (fast model) if configured, otherwise fall back to main model
        if self.archive_model:
            from nanobot.config.loader import load_config
            config = load_config()
            archive_provider = self._make_archive_provider(config, self.archive_model)
            if archive_provider:
                ok = await self.memory_store.consolidate(
                    session, archive_provider, self.archive_model,
                    archive_all=archive_all, memory_window=self.memory_window,
                )
                if ok and self._memory_search_tool:
                    await self._memory_search_tool.maybe_sync(force=False)
                return ok
            logger.warning("Failed to create archive provider, falling back to main model")

        ok = await self.memory_store.consolidate(
            session, self.provider, self.model,
            archive_all=archive_all, memory_window=self.memory_window,
        )
        if ok and self._memory_search_tool:
            await self._memory_search_tool.maybe_sync(force=False)
        return ok

    @staticmethod
    def _looks_like_memory_intent(text: str) -> bool:
        lowered = text.lower()
        patterns = [
            "记住这个", "记住：", "记住:", "记一下", "记下来", "帮我记住",
            "remember this", "remember:", "save this memory",
        ]
        return any(p in lowered for p in patterns)

    @staticmethod
    def _extract_memory_payload(text: str) -> str:
        trimmed = text.strip()
        splitters = [
            "记住这个：", "记住这个:", "记住：", "记住:", "remember this:", "remember:",
        ]
        lowered = trimmed.lower()
        for splitter in splitters:
            pos = lowered.find(splitter.lower())
            if pos >= 0:
                payload = trimmed[pos + len(splitter):].strip()
                if payload:
                    return payload
        return trimmed

    @staticmethod
    def _parse_memory_confirmation(text: str) -> str | None:
        lowered = text.strip().lower()
        if lowered in {"取消", "算了", "cancel", "no"}:
            return "cancel"
        long_answers = {"长期", "long", "long-term", "long term", "长期记忆", "1"}
        daily_answers = {"今天", "当日", "daily", "today", "短期", "2"}
        if lowered in long_answers:
            return "long_term"
        if lowered in daily_answers:
            return "daily"
        return None

    def _save_direct_turn(self, session: Session, user_text: str, assistant_text: str) -> None:
        session.add_message("user", user_text)
        session.add_message("assistant", assistant_text)

    def _handle_pending_memory_capture(
        self,
        msg: InboundMessage,
        session: Session,
        pending: dict[str, Any],
    ) -> OutboundMessage | None:
        choice = self._parse_memory_confirmation(msg.content)
        if choice == "cancel":
            session.metadata.pop("pending_memory_capture", None)
            content = "Memory save canceled."
            self._save_direct_turn(session, msg.content, content)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)
        if choice in {"long_term", "daily"}:
            payload = str(pending.get("text", "")).strip()
            if not payload:
                session.metadata.pop("pending_memory_capture", None)
                content = "Nothing to save."
                self._save_direct_turn(session, msg.content, content)
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)
            path = self.memory_store.save_immediate_memory(payload, target=choice)
            session.metadata.pop("pending_memory_capture", None)
            content = f"Saved to {path.relative_to(self.workspace)}."
            self._save_direct_turn(session, msg.content, content)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)
        if len(msg.content.strip()) <= 20:
            ask = "请确认写入位置：回复 `长期` 或 `今天`，或回复 `取消`。"
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=ask)
        session.metadata.pop("pending_memory_capture", None)
        return None

    def _handle_immediate_memory_capture(
        self,
        msg: InboundMessage,
        session: Session,
    ) -> OutboundMessage | None:
        if not self._looks_like_memory_intent(msg.content):
            return None
        payload = self._extract_memory_payload(msg.content)
        if not payload:
            content = "请提供要记住的内容。"
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

        target, confidence = MemoryStore.classify_target(payload)
        if confidence < 0.7:
            session.metadata["pending_memory_capture"] = {"text": payload, "suggested": target}
            ask = (
                "我识别到你想让我记住这条信息。"
                "请确认写入位置：回复 `长期`（写入 MEMORY.md）或 `今天`（写入当日文件）。"
            )
            self._save_direct_turn(session, msg.content, ask)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=ask)

        path = self.memory_store.save_immediate_memory(payload, target=target)
        content = f"Saved to {path.relative_to(self.workspace)}."
        self._save_direct_turn(session, msg.content, content)
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

    def _make_archive_provider(self, config, model: str):
        """Create a provider for the archive model."""
        try:
            return create_provider(
                config,
                model,
                set_global_api_base=False,
                allow_missing_standard_credentials=True,
            )
        except ValueError:
            return None

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""

    async def process_inbound(
        self,
        msg: InboundMessage,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a fully formed inbound message and return final content."""
        await self._connect_mcp()
        response = await self._process_message(msg, session_key=msg.session_key, on_progress=on_progress)
        return response.content if response else ""
