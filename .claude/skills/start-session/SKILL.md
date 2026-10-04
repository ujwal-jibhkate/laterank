---
name: start-session
description: Start a laterank work session. Reads the progress log, checks the repo's health, and proposes today's plan. Run it at the start of each day or after /clear.
argument-hint: "[today's goal, optional]"
disable-model-invocation: true
---

# Start a work session

A new session remembers nothing about earlier sessions. `docs/dev/progress.md` is the project's
memory, so read it before anything else.

1. Read `CLAUDE.md` and `docs/dev/progress.md`. Run `python3 scripts/specs.py active` and read
   that spec in full: its unticked acceptance tests and its **Tasks** table are today's work. If
   no spec is approved, today's first task is drafting the next one with `/spec <id>`, because
   no code can change until it's approved.
2. Look at the repo:
   - `git status --short`
   - `git log --oneline -10` (a brand-new repo has no commits yet; that's fine)
   - `git worktree list` (leftover worktrees from parallel agents)
3. Check health, without fixing anything yet:
   - `uv sync --quiet`
   - `uv run pytest -q`
   - `uv run ruff check .`
4. Report to the owner in 15 lines or fewer:
   - where we are: milestone, the active spec (id and status), how many of its acceptance tests
     are ticked, and the last log entry's date
   - anything that changed since that entry: commits it doesn't mention, uncommitted files,
     leftover worktrees
   - test and lint status: pass or fail counts, plus the first failure's name if any
   - the log's **Next** items
5. Propose today's plan: 1–3 tasks, each with a one-line "done means". Mark which tasks are
   independent enough to run in parallel with `/split-work`. Two tasks count as independent only
   when they touch different files and neither needs the other's code.
6. If the owner gave a goal ($ARGUMENTS), plan around it. If it conflicts with the log's
   **Next**, say so.
7. Stop and wait for the owner's OK before writing any code.
