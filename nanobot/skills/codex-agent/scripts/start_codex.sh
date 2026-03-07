#!/bin/sh
set -eu

SOCKET_DIR="${NANOBOT_TMUX_SOCKET_DIR:-${TMPDIR:-/tmp}/nanobot-tmux-sockets}"
SOCKET="$SOCKET_DIR/nanobot.sock"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BASE_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
SESSION="${1:?usage: start_codex.sh <session> <workdir> [codex args ...]}"
WORKDIR="${2:?usage: start_codex.sh <session> <workdir> [codex args ...]}"
shift 2

mkdir -p "$SOCKET_DIR"
command -v tmux >/dev/null 2>&1 || { echo "tmux not found" >&2; exit 1; }
command -v codex >/dev/null 2>&1 || { echo "codex not found" >&2; exit 1; }
[ -d "$WORKDIR" ] || { echo "workdir not found: $WORKDIR" >&2; exit 1; }

tmux -S "$SOCKET" kill-session -t "$SESSION" >/dev/null 2>&1 || true
tmux -S "$SOCKET" new-session -d -s "$SESSION" -c "$WORKDIR"
tmux -S "$SOCKET" send-keys -t "$SESSION":0.0 "codex --no-alt-screen $*" Enter
"$BASE_DIR/hooks/pane_monitor.sh" "$SESSION" >/dev/null 2>&1 &

echo "session=$SESSION"
echo "socket=$SOCKET"
echo "workdir=$WORKDIR"
