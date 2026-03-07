#!/bin/sh
set -eu

SOCKET_DIR="${NANOBOT_TMUX_SOCKET_DIR:-${TMPDIR:-/tmp}/nanobot-tmux-sockets}"
SOCKET="$SOCKET_DIR/nanobot.sock"
SESSION="${1:?usage: stop_codex.sh <session>}"
PID_FILE="/tmp/nanobot_codex_monitor_${SESSION}.pid"

tmux -S "$SOCKET" kill-session -t "$SESSION" >/dev/null 2>&1 || true
if [ -f "$PID_FILE" ]; then
  kill "$(cat "$PID_FILE")" >/dev/null 2>&1 || true
  rm -f "$PID_FILE"
fi
