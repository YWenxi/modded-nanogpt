---
name: commit-by-parts
description: Commit pending changes split into separate logical commits, using Conventional Commits message style with bullet-list bodies. Use when the user asks to "commit by parts", commit in separate logical pieces, or split changes into multiple commits.
---

# Commit by parts

Split the working tree's pending changes into multiple commits, one per logical change,
instead of a single mixed commit.

## Procedure

1. Run `git status --short` and `git diff` (plus `git diff` on untracked files' contents by
   reading them) to inventory everything pending.
2. Group the changes into logical parts: one concern per commit (e.g. a note amendment, a new
   note, a code fix, a config change). A file may need to be split with `git add -p` if it
   contains two unrelated changes — ask the user before splitting hunks within one file.
3. Stage and commit each part separately, in dependency order (earlier commits should not
   depend on later ones).
4. After all commits, show `git log --oneline` for the new commits and confirm
   `git status --short` is clean (or list what was deliberately left out).

## Commit message format

Follow https://www.conventionalcommits.org/en/v1.0.0/ :

```
<type>(<optional scope>): <short imperative summary>

- <bullet describing one aspect of the change>
- <another bullet>
- <as many as needed, one idea each>
```

- Types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `style`.
  Use `docs` for notes, `feat` for new functionality, `fix` for bug fixes.
- Summary: imperative, lowercase after the colon, no trailing period, ~72 chars max.
- Body: bullet list (`- `), one idea per bullet; include the "why" and key file/record
  references rather than paraphrasing the diff. Separate body from summary with a blank line.
- Breaking changes: add `!` after the type/scope and a `BREAKING CHANGE:` footer.
- Do not squash unrelated changes into one commit just to reduce commit count; but also do
  not split one coherent change across commits.

## Rules

- Never commit files the user has not asked to commit (secrets, scratch outputs); list
  anything deliberately left uncommitted in the final reply.
- Match the repo's existing history if it clearly conflicts with these rules, and say so.
- Run repo-mandated pre-commit checks (tests, linters) on the final state if the project
  defines them.
