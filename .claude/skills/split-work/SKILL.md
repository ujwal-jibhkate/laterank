---
name: split-work
description: Split laterank work into independent tasks and run them in parallel with implementer agents, each in its own git worktree. A reviewer agent checks each result before anything is merged.
argument-hint: "[what to parallelize, optional]"
disable-model-invocation: true
---

# Run independent tasks in parallel

## 1. Preconditions (stop if any fails)

- The repo has at least one commit, and `git status --short` is clean. Agent worktrees are created
  from the last commit, so uncommitted work is invisible to them. If the tree is dirty, suggest
  `/wrap-up` first.
- A spec is approved (`python3 scripts/specs.py active`) and committed. The tasks come from its
  **Tasks** table, and the agents build against its **Interfaces**.
- Every task's inputs already exist on main. If two tasks share an interface, it must already
  be in the spec's **Interfaces** section, written down exactly. Otherwise, update the spec
  first.

## 2. Plan, and get approval

From the spec's **Tasks** table (rows marked parallel), the owner's request ($ARGUMENTS) and the
code, build a table:

| task | files it owns (creates or edits) | must not touch | done means | test command |
|---|---|---|---|---|

Rules:
- No file appears in two tasks' "owns" column.
- 2 to 4 tasks. More is harder to review than it saves.
- Not everything fits. Work that needs design judgment stays in the main session.

Show the table and wait for the owner's OK.

## 3. Spawn

Start one `implementer` agent per task, all in the **same message** so they run in parallel. Each
prompt must stand alone, because the agent hasn't seen this conversation. Include:
- the spec's path and id, and which of its acceptance tests this task must make pass
- the goal, the files it owns, the files it must not touch, and what "done" means
- the exact test command
- which project skills apply (`port-module`, `golden-values`)
- the relevant smve-lab source paths, when porting

## 4. Review and merge

For each finished agent:
1. Run the `reviewer` agent on its branch (independent branches can be reviewed in parallel).
2. Report to the owner: branch, files, test results, the reviewer's verdict and its top findings.
3. Merge only with the owner's OK, one branch at a time (`git merge --no-ff <branch>`), and run
   `uv run pytest -q` after each merge.
4. After merging, ask before removing the worktree and branch
   (`git worktree remove <path>` and `git branch -d <branch>`).

Add a line to the progress log for each merged branch.
