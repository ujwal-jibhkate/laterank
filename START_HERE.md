# Start here

Read this whenever you've forgotten how to work on this repo. It takes 5 minutes.

**laterank** is a local late-interaction retrieval library: a cheap first stage finds about 100
candidate documents, then exact MaxSim reranks them. It grows out of the research repo
`~/Projects/smve-lab`. The why behind every design choice is in
`~/Projects/smve-lab/docs/library_design_explained.md`.

---

## 1. The daily loop

```bash
cd ~/Projects/laterank
claude --add-dir ~/Projects/smve-lab      # Claude can read the research code and results
```

```text
/start-session          where are we? health check, today's plan → you say OK
   ... work: one task at a time, /clear between unrelated tasks ...
/wrap-up                checks, ticks the spec, updates the log, proposes a commit → you say yes
```

That's it. A new session remembers nothing from earlier ones. It learns where the project stands
from three files, which `/start-session` reads for you:

| file | answers |
|---|---|
| `docs/dev/progress.md` | where are we, what happened, what's **Next** |
| `docs/specs/<id>-*.md` (the approved one) | what are we building right now, and how do we know it's done |
| `CLAUDE.md` | the rules that never change |

---

## 2. Specs: the one rule that matters

Every checkpoint (M0-1, M0-2, M0-3, ...) gets a one-page **spec** *before* any code.

```text
/spec M0-3                Claude drafts docs/specs/M0-3-<name>.md and explains it to you
/spec M0-3 approve        YOU decide this. Code in src/ and tests/ can now change
   ... build, review, /wrap-up ...
/spec M0-3 change "..."   the plan changed: the spec is updated first, with a dated reason
/spec M0-3 done           YOU decide this, once every acceptance test is ticked
```

- Only **one** spec is approved at a time, which means exactly one checkpoint is in progress.
- Every commit that touches code ends with `Spec: M0-3`.
- Full rules: `docs/specs/README.md`.

Which spec is active right now:

```bash
python3 scripts/specs.py active
```

---

## 3. Every command you'll need

### Inside Claude Code

| type this | what it does |
|---|---|
| `/start-session [goal]` | start the day: reads the log and the spec, checks tests, proposes a plan |
| `/wrap-up` | end the day: checks, updates the spec and log, proposes a commit |
| `/spec <id> [approve\|done\|change "..."]` | write or move a spec through its life |
| `/split-work` | run 2–4 independent tasks in parallel with agents |
| `/explain <topic>` | teaches you something from the code, with a small example it runs |
| `/ponytail [lite\|full\|ultra\|off]` | how hard Claude pushes for less code (default `full`); no argument shows the level |
| `/ponytail-review` | lists over-built code in the current diff as a delete-list |
| `/ponytail-audit` | the same, for the whole repo |
| `/ponytail-debt` | collects every `# ponytail:` shortcut into one list |
| `@agent-reviewer review my changes` | a read-only review: `merge` or `fix first` |
| `@agent-researcher how does bm25s store postings?` | research with sources |
| `/clear` | start fresh in the same terminal (do this between unrelated tasks) |
| Shift+Tab | cycle permission modes; **plan mode** proposes a plan before touching anything |
| `claude -c` | reopen your last conversation (fine for short breaks; for a new day, start fresh) |

### In the terminal

```bash
uv sync                                   # install everything
uv run pytest -q                          # all tests (golden tests skip unless the env var is set)
LATERANK_GOLDEN_DIR=~/Projects/smve-lab/artifacts/embeddings uv run pytest -m slow   # golden tests
uv run ruff check . && uv run ruff format --check .                                 # lint
python3 scripts/specs.py check            # are all specs valid?
git log --oneline --grep "Spec: M0-2"     # every commit that belongs to one checkpoint
```

---

## 4. Running agents in parallel

Use this only for tasks that touch **different files** and **don't depend on each other**.
Designing an interface is never parallel work: it goes in the spec first.

The easy way:

```text
/split-work
```

It checks that everything is committed, builds a task table from the spec, waits for your OK,
starts one `implementer` agent per task (each in its own git worktree), has the `reviewer` check
each one, and merges only with your OK.

Or ask in plain words:

```text
Use separate implementer agents in parallel for spec M0-3 tasks 2 and 3.
```

Agents get a copy of the **last commit**, so commit before you split. 2–3 agents at a time is
plenty, since each one costs about as much as a full session.

---

## 5. Ponytail: less code

The `ponytail` plugin is on for this repo. It makes Claude (and every agent) climb a ladder
before writing code: *is it needed at all → does laterank already have it → numpy/stdlib →
one line → only then the minimum that works.* It never cuts safety: no-pickle loads, atomic
writes, input validation, golden tests (full list in `CLAUDE.md`).

How it fits the spec workflow:
- `/spec` asks "do we need this?" while drafting. Skipped things are listed in the spec's
  **Out of scope** as `Skipped: X, add when Y`.
- Implementers build only what the spec names. The reviewer adds a **Delete-list** of over-built
  code to every review.
- A deliberate shortcut gets a `# ponytail: <limit>, <upgrade path>` comment, and `/wrap-up`
  logs it.
- Explanations to you stay full length. Ponytail shrinks code, not teaching.

It installs itself on a fresh clone: Claude Code reads `.claude/settings.json` and asks you to
trust the `ponytail` marketplace the first time.

---

## 6. Where everything lives

```text
START_HERE.md            ← you are here
CLAUDE.md                rules for Claude (and for you)
docs/dev/progress.md     project log: current state, checklist, what was ported, dated entries
docs/specs/              one spec per checkpoint + README (rules) + _template.md
src/laterank/            the library, in layers: api → orchestration → components → storage → kernels
tests/                   mirrors src/; tests/golden/ re-checks smve-lab's numbers
scripts/specs.py         the program that enforces the spec rules
scripts/git-hooks/       git hook: Spec trailer required, no Claude attribution
.claude/skills/          /start-session, /wrap-up, /spec, /split-work, /explain, port-module, golden-values
.claude/agents/          implementer, reviewer, researcher
.claude/settings.json    the spec hook (blocks code edits without an approved spec) + the ponytail plugin
```

---

## 7. When something blocks you

| you see | why | do this |
|---|---|---|
| "laterank is spec-driven: src/ and tests/ can change only while a spec is approved" | no spec is approved | `/spec <next id>`, read it, then `/spec <id> approve` |
| a prompt asking to confirm a spec set to `approved` / `done` | only you may do that | allow it only if you asked for it |
| `commit-msg: ... needs a trailer naming its spec` | the commit touches code | add a last line `Spec: M0-3` (or `Spec: none - <reason>` for a tiny fix) |
| `commit-msg: spec M0-3 is still a draft` | the plan isn't approved | approve the spec first |
| `commit-msg: Claude attribution is not allowed` | a Co-Authored-By line slipped in | delete that line |
| `spec error: ...` in CI | a spec is missing a section or has a bad status | `python3 scripts/specs.py check` locally and fix |
| golden tests "skipped" | `LATERANK_GOLDEN_DIR` isn't set | normal in CI; set it locally to run them |
| Claude doesn't know where we are | the log is stale | check `docs/dev/progress.md` **Next**, then `/wrap-up` properly next time |

---

## 8. Setting up a fresh clone or a new machine

```bash
git clone https://github.com/ujwal-jibhkate/laterank.git && cd laterank
git config core.hooksPath scripts/git-hooks      # turns on the commit-msg hook (once per clone)
uv sync
uv run pytest -q
```

For golden tests you also need smve-lab's embeddings (~14 GB, never committed) at
`~/Projects/smve-lab/artifacts/embeddings/`.

---

## 9. House rules (short version)

- Claude never adds Co-Authored-By or "Generated with" lines. The git hook enforces this.
- Claude commits only when you say so, and never pushes. **You** push.
- `kernels/` and `storage/` never import torch. Imports point down the layers only.
- Reference numbers come from the `golden-values` skill, never from memory.
- When you don't understand something: `/explain <it>`.
- Ponytail is on: the least code that works, never at the cost of safety. `/ponytail off` turns it off for a session.
