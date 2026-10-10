---
name: log-record-score
description: Log a training run's score in the README.md record tracker tables. Use when the user asks to "log the score", "log the record", or add a run/result to the README tracker.
---

# Log record score

The root `README.md` has a "Record tracker" with one table per track:

- Track 1: GPT-2 small (target 3.28) — runs of `train_gpt.py` variants
- Track 2: GPT-2 Medium (target 2.92) — runs of `train_gpt_medium.py` variants

Each row links to a dated record dir under `a100_records/`. Every record dir should
have a tracker row; if you notice an unlisted record dir while logging, add it too
(marked as ablation if it set no record — see the AdamW / PolarExpress rows).

## Procedure

1. Identify the run's record dir (usually `a100_records/{date}_{Name}/`) and its README.
2. Get the numbers from the run log, not from memory:
   - final val loss: last `step:N/N val_loss:X` line (grep the console log)
   - record time: `train_time` ms on that line, converted to minutes (2 decimals)
   - baseline for comparison: the previous best row's val loss / the track target
3. Append a row to the matching track's table, keeping the column order:

   `| # | Record time | Date | Title | Description | Record |`

   - `#` — next integer in that track
   - `Record time` — minutes, e.g. `7.05 minutes`. For runs that set no record:
     `— (ablation)` or `— (target not reached within N steps)`
   - `Date` — the run date (YYYY-MM-DD), matching the record dir name
   - `Title` — short name of the change, e.g. `NorMuon optimizer`
   - `Description` — what changed + val loss vs baseline/target at equal steps,
     plus one notable statistic (e.g. "ahead at all 47 val evals", "diverges
     without safety factor"). For ablations, end with "Not a record — ablation only"
   - `Record` — relative link: `[a100_records/2026-10-09_NorMuon](a100_records/2026-10-09_NorMuon)`

4. Verify: `git diff README.md` shows only the new row(s), numbers match the logs,
   and the link target directory exists.

## Rules

- One row per run/record dir, appended in date order; never reorder or edit
  existing rows except to fix factual errors.
- Numbers must come from the run logs in the record dir — quote them as printed
  (4 decimals for val loss when the log has that precision).
- Comparisons must be at equal step counts and hyperparameters; say so in the
  Description ("at the same 1695 steps").
- Do not commit unless the user asks (commit-by-parts handles commits separately).
