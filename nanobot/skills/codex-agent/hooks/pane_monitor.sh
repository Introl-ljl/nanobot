#!/bin/sh
set -eu

SOCKET_DIR="${NANOBOT_TMUX_SOCKET_DIR:-${TMPDIR:-/tmp}/nanobot-tmux-sockets}"
SOCKET="${CODEX_AGENT_SOCKET:-$SOCKET_DIR/nanobot.sock}"
SESSION="${1:?usage: pane_monitor.sh <session>}"
INTERVAL="${CODEX_AGENT_MONITOR_INTERVAL:-5}"
PID_FILE="/tmp/nanobot_codex_monitor_${SESSION}.pid"
LAST_APPROVAL=""

echo $$ > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT INT TERM

while :; do
  if ! tmux -S "$SOCKET" has-session -t "$SESSION" 2>/dev/null; then
    exit 0
  fi

  OUTPUT=$(tmux -S "$SOCKET" capture-pane -p -J -t "$SESSION":0.0 -S -40 2>/dev/null || true)
  case "$OUTPUT" in
    *"Would you like to run"*|*"Press enter to confirm"*|*"approve this"*|*"allow this"*)
      CMD=$(printf '%s\n' "$OUTPUT" | sed -n 's/^.*\$ //p' | tail -n 1)
      KEY="$SESSION|$CMD"
      if [ "$KEY" != "$LAST_APPROVAL" ]; then
        LAST_APPROVAL="$KEY"
        if [ -n "${CODEX_AGENT_CHANNEL:-}" ] && [ -n "${CODEX_AGENT_CHAT_ID:-}" ] && [ "${CODEX_AGENT_NOTIFY:-1}" = "1" ]; then
          nanobot hooks notify --channel "$CODEX_AGENT_CHANNEL" --chat-id "$CODEX_AGENT_CHAT_ID" --message "⏸️ Codex 等待审批\n🔧 session: $SESSION\n📋 command: ${CMD:-unknown}" >/dev/null 2>&1 || true
        fi
        if [ -n "${CODEX_AGENT_SESSION:-}" ]; then
          ORIGIN_CHANNEL=${CODEX_AGENT_SESSION%%:*}
          ORIGIN_CHAT_ID=${CODEX_AGENT_SESSION#*:}
          nanobot hooks inject --system --channel "$ORIGIN_CHANNEL" --chat-id "$ORIGIN_CHAT_ID" --sender-id codex-monitor --message "[Codex Monitor] Codex is waiting for approval. Inspect the command and decide whether to approve in tmux.\nsession: $SESSION\ncommand: ${CMD:-unknown}" >/dev/null 2>&1 || true
        fi
      fi
      ;;
  esac
  sleep "$INTERVAL"
done
