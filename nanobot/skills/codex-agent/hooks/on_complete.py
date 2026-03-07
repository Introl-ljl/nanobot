#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys


def _run(args: list[str]) -> None:
    subprocess.run(args, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    if len(sys.argv) < 2:
        return 0

    try:
        payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        return 1

    if payload.get("type") != "agent-turn-complete":
        return 0

    channel = os.environ.get("CODEX_AGENT_CHANNEL", "")
    chat_id = os.environ.get("CODEX_AGENT_CHAT_ID", "")
    session = os.environ.get("CODEX_AGENT_SESSION", "")
    notify = os.environ.get("CODEX_AGENT_NOTIFY", "1") == "1"

    summary = payload.get("last-assistant-message", "") or "Turn complete."
    cwd = payload.get("cwd", "unknown")
    thread_id = payload.get("thread-id", "unknown")

    if notify and channel and chat_id:
        _run([
            "nanobot", "hooks", "notify",
            "--channel", channel,
            "--chat-id", chat_id,
            "--message", f"🔔 Codex 完成一轮\n📁 {cwd}\n💬 {summary}",
        ])

    if session and ":" in session:
        origin_channel, origin_chat_id = session.split(":", 1)
        _run([
            "nanobot", "hooks", "inject",
            "--system",
            "--channel", origin_channel,
            "--chat-id", origin_chat_id,
            "--sender-id", "codex-hook",
            "--message", f"[Codex Hook] Codex completed a turn. Review the result, decide whether to continue the tmux session, and only report done when quality is acceptable.\ncwd: {cwd}\nthread: {thread_id}\nsummary: {summary}",
        ])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
