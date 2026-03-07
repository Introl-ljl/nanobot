# Codex Agent Workflow

## Goal

Run Codex as a supervised coding worker that nanobot can resume across turns.

## Session contract

- One Codex task maps to one tmux session.
- The user-facing conversation stays in the original nanobot chat session.
- Hook scripts call `nanobot hooks inject --system` so follow-up messages append to the same nanobot session context.

## Hook message types

- Completion hook message: asks nanobot to review Codex output and decide whether to continue or report back.
- Approval hook message: includes the candidate shell command and tmux session so nanobot can approve, reject, or ask the user.

## Recommended execution loop

1. Clarify the user request.
2. Decide Codex vs direct nanobot execution.
3. Start tmux-hosted Codex session if Codex is chosen.
4. Send an optimized prompt.
5. Let hooks wake nanobot on state changes.
6. Iterate in the same session until result quality is acceptable.
7. Clean up the tmux session and monitor.

## Failure handling

- If the notify hook is not configured, fallback to manual `tmux capture-pane` polling.
- If `codex` is missing, stop immediately and tell the user to install Codex CLI.
- If the target channel cannot be notified, still inject back into nanobot so the orchestration loop continues.
