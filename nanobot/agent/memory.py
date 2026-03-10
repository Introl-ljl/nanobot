"""Memory system for persistent agent memory."""

from __future__ import annotations

import json
import re
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.utils.helpers import ensure_dir

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session


_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save the memory consolidation result to persistent storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "daily_entries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Short process notes to append into today's memory/YYYY-MM-DD.md. "
                        "Include key temporary decisions, debugging steps, and follow-ups.",
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "Full updated long-term memory as markdown. Include all existing "
                        "facts plus new ones. Return unchanged if nothing new.",
                    },
                    "history_entry": {
                        "type": "string",
                        "description": "Legacy compatibility field. Prefer daily_entries.",
                    },
                },
                "required": ["daily_entries", "memory_update"],
            },
        },
    }
]


class MemoryStore:
    """Memory with long-term MEMORY.md and daily process logs."""

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self.migration_marker = self.memory_dir / ".history_migrated_v2"
        self.migrate_legacy_history()

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def daily_file_for(self, day: str | None = None) -> Path:
        date_str = day or datetime.now().strftime("%Y-%m-%d")
        return self.memory_dir / f"{date_str}.md"

    def append_daily(self, entry: str, day: str | None = None) -> Path:
        target = self.daily_file_for(day)
        stamped = self._ensure_timestamp_prefix(entry.rstrip())
        with open(target, "a", encoding="utf-8") as f:
            f.write(stamped.rstrip() + "\n\n")
        return target

    def save_immediate_memory(self, content: str, target: str) -> Path:
        clean = content.strip()
        if target == "long_term":
            old = self.read_long_term().strip()
            if not old:
                new_content = f"- {clean}\n"
            else:
                new_content = old + ("\n" if not old.endswith("\n") else "") + f"- {clean}\n"
            self.write_long_term(new_content)
            return self.memory_file
        if target == "daily":
            return self.append_daily(clean)
        raise ValueError(f"Unsupported memory target: {target}")

    def append_history(self, entry: str) -> None:
        """Legacy append path (kept for compatibility)."""
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    def migrate_legacy_history(self) -> None:
        """One-time migration from HISTORY.md to memory/YYYY-MM-DD.md files."""
        if self.migration_marker.exists():
            return

        if not self.history_file.exists():
            self.migration_marker.write_text("no-history\n", encoding="utf-8")
            return

        raw = self.history_file.read_text(encoding="utf-8")
        entries = [entry.strip() for entry in re.split(r"\n\s*\n", raw) if entry.strip()]
        migrated = 0
        for entry in entries:
            day = self._extract_date(entry) or datetime.now().strftime("%Y-%m-%d")
            self.append_daily(entry, day=day)
            migrated += 1

        if migrated > 0:
            self.history_file.write_text(
                "# Legacy HISTORY.md migrated to daily memory files.\n"
                "# This file is retained for backward compatibility only.\n",
                encoding="utf-8",
            )
            logger.info("Migrated {} legacy history entries into daily memory files", migrated)
        self.migration_marker.write_text(f"migrated={migrated}\n", encoding="utf-8")

    @staticmethod
    def _extract_date(entry: str) -> str | None:
        match = re.match(r"\[(\d{4}-\d{2}-\d{2})[^\]]*\]", entry.strip())
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _ensure_timestamp_prefix(entry: str) -> str:
        if re.match(r"^\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\]", entry):
            return entry
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"[{now}] {entry}"

    @staticmethod
    def classify_target(text: str) -> tuple[str, float]:
        """Heuristic classifier for immediate memory routing."""
        lower = text.lower()
        long_markers = [
            "长期", "偏好", "习惯", "固定", "规则", "配置", "项目背景",
            "always", "prefer", "preference", "default",
        ]
        daily_markers = [
            "今天", "刚刚", "这次", "临时", "排障", "待跟进", "todo", "follow-up", "today",
        ]
        long_score = sum(1 for kw in long_markers if kw in lower)
        daily_score = sum(1 for kw in daily_markers if kw in lower)
        if "长期" in lower or "long-term" in lower:
            return "long_term", 0.95
        if "当日" in lower or "daily" in lower or "today" in lower:
            return "daily", 0.95
        if long_score > daily_score:
            return "long_term", 0.78
        if daily_score > long_score:
            return "daily", 0.78
        return "daily", 0.45

    @staticmethod
    def content_hash(content: str) -> str:
        return sha256(content.encode("utf-8")).hexdigest()

    async def consolidate(
        self,
        session: Session,
        provider: LLMProvider,
        model: str,
        *,
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> bool:
        """Consolidate old messages into MEMORY.md + memory/YYYY-MM-DD.md via LLM tool call.

        Returns True on success (including no-op), False on failure.
        """
        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info("Memory consolidation (archive_all): {} messages", len(session.messages))
        else:
            keep_count = memory_window // 2
            if len(session.messages) <= keep_count:
                return True
            if len(session.messages) - session.last_consolidated <= 0:
                return True
            old_messages = session.messages[session.last_consolidated:-keep_count]
            if not old_messages:
                return True
            logger.info("Memory consolidation: {} to consolidate, {} keep", len(old_messages), keep_count)

        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")

        current_memory = self.read_long_term()
        prompt = f"""Process this conversation and call the save_memory tool with your consolidation.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{chr(10).join(lines)}"""

        try:
            response = await provider.chat_with_retry(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Call the save_memory tool with your consolidation of the conversation."},
                    {"role": "user", "content": prompt},
                ],
                tools=_SAVE_MEMORY_TOOL,
                model=model,
            )

            if not response.has_tool_calls:
                logger.warning("Memory consolidation: LLM did not call save_memory, skipping")
                return False

            args = response.tool_calls[0].arguments
            # Some providers return arguments as a JSON string instead of dict
            if isinstance(args, str):
                args = json.loads(args)
            # Some providers return arguments as a list (handle edge case)
            if isinstance(args, list):
                if args and isinstance(args[0], dict):
                    args = args[0]
                else:
                    logger.warning("Memory consolidation: unexpected arguments as empty or non-dict list")
                    return False
            if not isinstance(args, dict):
                logger.warning("Memory consolidation: unexpected arguments type {}", type(args).__name__)
                return False

            daily_entries = args.get("daily_entries", [])
            if isinstance(daily_entries, list):
                for entry in daily_entries:
                    if not isinstance(entry, str):
                        entry = json.dumps(entry, ensure_ascii=False)
                    if entry.strip():
                        self.append_daily(entry)
            elif daily_entries:
                entry = daily_entries if isinstance(daily_entries, str) else json.dumps(daily_entries, ensure_ascii=False)
                self.append_daily(entry)

            if entry := args.get("history_entry"):
                # Backward compatibility for models that still output the old field.
                if not isinstance(entry, str):
                    entry = json.dumps(entry, ensure_ascii=False)
                if entry.strip():
                    self.append_daily(entry)
            if update := args.get("memory_update"):
                if not isinstance(update, str):
                    update = json.dumps(update, ensure_ascii=False)
                if update != current_memory:
                    self.write_long_term(update)

            session.last_consolidated = 0 if archive_all else len(session.messages) - keep_count
            logger.info("Memory consolidation done: {} messages, last_consolidated={}", len(session.messages), session.last_consolidated)
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return False
