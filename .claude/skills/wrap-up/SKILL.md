---
name: wrap-up
description: End a laterank work session. Runs the checks, updates docs/dev/progress.md so the next session can pick up, and proposes a commit. Run it before closing the terminal or before /clear.
disable-model-invocation: true
---

# Wrap up a work session

The next session starts from `docs/dev/progress.md` and nothing else. Write it for someone who
wasn't here today.

1. Run the checks and keep the real results, even if they fail:
   - `uv run ruff check .`
   - `uv run ruff format --check .`
   - `uv run pytest -q`
2. Update the active spec (`python3 scripts/specs.py active`):
   - Tick each acceptance test whose check now exists and passes. Leave the rest unticked.
   - If today's work changed the plan, add a dated line under **Changes**. If the code no longer
     matches **Interfaces**, fix one or the other now, and tell the owner which.
   - Run `python3 scripts/specs.py check`.
   - If every acceptance test is ticked, suggest `/spec <id> done` to the owner. Don't mark it
     done yourself.
3. Update `docs/dev/progress.md`:
   - Rewrite **Current state** (at most 6 lines): milestone, checkpoint, what works, what's broken.
   - Add a new entry at the top of **Log**, headed with today's absolute date (`### 2026-10-05`):
     - **Done**: what was built or changed, naming the files.
     - **Decisions**: each with a one-line *why*. Copy lasting decisions to CLAUDE.md as well.
     - **Surprises**: bugs, wrong assumptions, numbers that didn't match.
     - **Next**: the concrete first step for the next session, specific enough to start without
       asking ("write the naive-loop test for `kernels/maxsim.py::maxsim_scores`", not "continue
       kernels").
   - Update the **Ported from smve-lab** table if code moved over.
   - Keep each entry under about 20 lines. The log is a map, not a diary.
4. Show the owner a short summary of the changes and a proposed commit message: a subject line of
   72 characters or fewer in the imperative, then a body explaining *why*, and a final line
   `Spec: <id>` when `src/` or `tests/` changed (the git hook rejects the commit without it).
   Never add Co-Authored-By, "Generated with", or any other Claude attribution.
5. Ask whether to commit. Commit only on a yes. Never push.
6. If `git worktree list` shows agent worktrees, list them with their branches and ask whether to
   merge, keep, or remove each one.
