---
name: codex-agent
description: Operate OpenAI Codex CLI as a managed coding worker for multi-step implementation tasks. Use when the user wants nanobot to launch and supervise `codex` in `tmux`, handle approvals or follow-up turns, relay progress back to a chat channel, or keep a reusable Codex workflow/knowledge base. Not for simple one-shot edits that nanobot can do directly.
metadata: {"nanobot":{"emoji":"🧠","requires":{"bins":["codex","tmux","python3"]}}}
---

# Codex Agent

Use this skill when a task should be delegated to the external `codex` CLI instead of being solved directly inside nanobot.

## When to use

- The user explicitly asks to use Codex or adapt `codex-agent`.
- The task benefits from a persistent interactive TTY in `tmux`.
- You want Codex to keep working across nanobot turns.
- You need notify hooks so Codex completion wakes nanobot again.

## Architecture in nanobot

- `tmux` hosts the long-running Codex TUI session.
- `hooks/on_complete.py` sends a user notification and injects a follow-up system message back into nanobot.
- `hooks/pane_monitor.sh` watches for approval prompts in the tmux pane and injects a review task back into nanobot.
- `scripts/start_codex.sh` and `scripts/stop_codex.sh` manage the tmux session and monitor lifecycle.
- Use `nanobot hooks notify` for one-shot outbound notifications.
- Use `nanobot hooks inject --system` to wake nanobot with Codex state.

## Working style

- First decide whether nanobot should do the work directly; prefer direct edits for small tasks.
- For Codex runs, pick a short session name and a concrete workdir.
- Prefer `--full-auto` for routine coding tasks; omit it when you want approval intervention.
- Send prompt text and `Enter` as separate `tmux send-keys` calls with a short sleep between them.
- After each injected completion or approval event, assess whether Codex should continue, be corrected, or be reported done.

## Start a Codex session

```bash
SESSION=codex-task
WORKDIR=/path/to/repo
SOCKET="${NANOBOT_TMUX_SOCKET_DIR:-${TMPDIR:-/tmp}/nanobot-tmux-sockets}/nanobot.sock"

{baseDir}/scripts/start_codex.sh "$SESSION" "$WORKDIR" --full-auto
```

To send the actual prompt:

```bash
tmux -S "$SOCKET" send-keys -t "$SESSION":0.0 -l -- "$PROMPT"
sleep 1
tmux -S "$SOCKET" send-keys -t "$SESSION":0.0 Enter
```

## Notify hook setup

Point Codex notify to the bundled hook. In `~/.codex/config.toml`:

```toml
notify = ["python3", "{baseDir}/hooks/on_complete.py"]
```

Required environment variables for the hook scripts:

- `CODEX_AGENT_CHANNEL` — target channel such as `feishu`, `telegram`, `discord`, `email`, `mochat`
- `CODEX_AGENT_CHAT_ID` — destination chat/user/room id
- `CODEX_AGENT_SESSION` — nanobot session key like `telegram:123456789`
- `CODEX_AGENT_NOTIFY` — set to `1` to send user-visible progress messages

## Approval handling

- `hooks/pane_monitor.sh` scans the pane for approval prompts.
- When approval is needed, nanobot receives a system message describing the command and session.
- If the task is safe and aligned, approve with `tmux send-keys -t target '1' Enter`.
- If not, reject and provide a corrective follow-up prompt.

## Reporting rules

- Only tell the user `done` after you have reviewed Codex output.
- Summarize changed files, validation status, and any remaining risk.
- If Codex gets stuck or drifts, keep the same tmux session and send a corrective instruction.

## Files to read when editing this skill

- `references/workflow.md` for the nanobot-specific orchestration contract
- `hooks/on_complete.py` for completion behavior
- `hooks/pane_monitor.sh` for approval detection

## Cleanup

```bash
{baseDir}/scripts/stop_codex.sh "$SESSION"
```
