---
name: write-note
description: Write a project note to notes/{date}_{topic}.md — setup records, procedures, experiment findings. Use when the user asks to "write a note", record how something was set up, or document a procedure or finding.
---

# Note writing

Notes live in `notes/` (repo root) and are named `{YYYY-MM-DD}_{topic}.md`:

- Date: today, from the session date reminder or `date +%F`. Never guess.
- Topic: short lowercase slug with underscores, e.g. `a100_dev_setup`, `flex_bf16_fallback`.
- Create `notes/` if missing. If a note with the same name exists, ask before overwriting.

## Structure

Follow this order; drop sections that don't apply rather than leaving them empty:

```markdown
# <Title>

Date: YYYY-MM-DD
Hardware/Environment: < GPUs, torch/CUDA versions, anything the procedure depends on >

## Why this is needed
< The problem or motivation, 2-6 sentences. Root causes, not symptoms. >

## Setup / Procedure
< Exact commands in fenced code blocks, in execution order, including how to undo
  destructive side effects first (e.g. stash local changes before checkout). >

## Verification
< How it was smoke-tested, with the observed output (timings, loss values, exit codes).
  Only results actually observed — never predicted ones. >

## Running
< The normal day-to-day command(s). >

## Caveats
< Numerics/performance comparability, unchanged-but-relevant settings, known limits. >

## Rollback
< How to return to the prior state. >
```

## Rules

- Verify before writing: every command in the note must have been run (or be trivially
  derivable from one that was). Cite code locations as `path/to/file:line`.
- Keep it under ~60 lines. Plain Markdown, light structure; no emoji.
- Notes are untracked by default — do not commit unless the user asks.
- After writing, reply with the path and a 2-3 line summary, not the full content.
