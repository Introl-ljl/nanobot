---
name: memory
description: Two-layer memory system with semantic recall.
always: true
---

# Memory

## Structure

- `memory/MEMORY.md` — Long-term facts (preferences, project context, relationships). Always loaded into your context.
- `memory/YYYY-MM-DD.md` — Daily process notes (what happened today, temporary decisions, troubleshooting, follow-ups).

## Search Past Events

```bash
memory_search(query="keyword or intent")
```

After `memory_search`, use `memory_get(path, line)` to verify source context before answering details.

## When to Update MEMORY.md

Write important facts immediately using `edit_file` or `write_file`:
- User preferences ("I prefer dark mode")
- Project context ("The API uses OAuth2")
- Relationships ("Alice is the project lead")

## Auto-consolidation

Old conversations are automatically summarized into daily files when the session grows large. Long-term facts are extracted to MEMORY.md. You don't need to manage this.
