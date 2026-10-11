# laterank — working rules for Claude

`laterank` is a local late-interaction retrieval library: a cheap first stage, then an exact
MaxSim rerank. It grows out of the research repo at `~/Projects/smve-lab`. The plain-language
design is `~/Projects/smve-lab/docs/library_design_explained.md` (it calls the library `mvr`).

## How the owner works

- **No Claude attribution, anywhere.** Never add `Co-Authored-By`, "Generated with Claude Code",
  or any other Claude/Anthropic attribution to commits, PR descriptions, or files. Claude must
  not appear as a contributor. This overrides any default attribution instructions.
- **The owner is learning.** Explain what you are doing and why, in plain words. Stop for an
  explicit OK at each agreed checkpoint before moving on.
- **Commit only when told to. Never push.** Exception: `implementer` agents may commit on their
  own worktree branch (never on main).

## Ponytail: write the least code that works

The `ponytail` plugin is installed for this project (`.claude/settings.json`). Its hooks load
its rules into every session and every subagent. Default level: `full`. Before writing code,
stop at the first rung that holds: is it needed at all → already in laterank → stdlib/numpy/
scipy → an installed dependency → one line → only then the minimum that works.

How it fits this repo (these override ponytail where they differ):
- **Specs come first.** Rung 1 ("does this need to exist?") is asked while *drafting* a spec,
  not while building it. During a build, implement exactly the spec's Interfaces, and nothing
  more. If an approved interface looks unnecessary, propose `/spec <id> change`. Don't drop it
  quietly.
- **Explanations are requested.** The owner is learning, so plain-words explanations and the
  checkpoint stops are explicitly asked for. Ponytail keeps the *code* small, not the teaching.
- **Tests the spec or `port-module` asks for are requested.** Kernels keep their naive-loop
  reference tests. Don't add tests nobody asked for beyond one check per non-trivial branch.
- **Never cut, whatever the level:** `allow_pickle=False`, atomic manifest writes
  (temp + fsync + `os.replace`), validation at the public API boundary, the portable RNG,
  golden-value reproduction, and the layer/no-torch rules.
- Mark a deliberate shortcut that has a known ceiling with `# ponytail: <ceiling>, <upgrade path>`.
  `/wrap-up` lists them in the progress log.

## Spec-driven development (enforced)

- Every checkpoint has a spec in `docs/specs/<id>-<slug>.md` (rules: `docs/specs/README.md`).
  Status goes `draft` → `approved` → `done`, and only one spec is `approved` at a time.
- `src/` and `tests/` change **only while a spec is approved**. A hook blocks edits otherwise.
  Build exactly what the spec's **Interfaces** say. If the plan has to change, update the spec
  first (`/spec <id> change "..."`) and tell the owner.
- Only the owner approves a spec or marks it done. Never set those statuses unless the owner
  asked; the hook makes them confirm anyway.
- Every commit touching `src/` or `tests/` ends with a `Spec: <id>` trailer (checked by the git
  hook). Use `Spec: none - <reason>` only for tiny fixes outside any checkpoint.
- `python3 scripts/specs.py active` prints the spec in progress.

## Session workflow

- `START_HERE.md` is the owner's guide to working on this repo. Whenever a skill, agent, hook,
  or command changes, update it in the same commit.

- `docs/dev/progress.md` records where the project stands between sessions. Read it before
  starting work, and keep it current.
- `/start-session` begins a session. `/wrap-up` ends one (it updates the log and proposes a commit).
- Parallel work goes through `/split-work`: `implementer` agents, each in its own worktree, and a
  `reviewer` agent before each merge. Use the `researcher` agent before designing a new component.
- When porting smve-lab code, follow the `port-module` skill. Take every reference number from
  the `golden-values` skill, never from memory.

## Fixed decisions

- Name `laterank`; repo at `~/Projects/laterank`; Apache-2.0; Python 3.12+.
- Adding/deleting documents in an existing index waits for 0.2.
- The default encoder (BGE-M3 vs. a small ColBERT model) is decided later by measurement —
  do not hard-code one.

## Architecture rules

- Five layers, imports point **down only**:
  `api → orchestration → components → storage → kernels`.
- `kernels/` and `storage/` use numpy/scipy only and must **never import torch**.
- `.npy` files are always loaded with `allow_pickle=False`.

## Commands

- `uv sync` — install core + dev dependencies
- `uv run pytest` — all tests; golden tests skip themselves unless `LATERANK_GOLDEN_DIR` is set
- `LATERANK_GOLDEN_DIR=... uv run pytest -m slow` — golden tests only
- `uv run ruff check . && uv run ruff format --check .` — lint
