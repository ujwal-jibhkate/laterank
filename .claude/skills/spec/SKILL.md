---
name: spec
description: Draft, revise, approve, or close a laterank checkpoint spec in docs/specs/. Code in src/ and tests/ can change only while a spec is approved.
argument-hint: "<id> [approve | done | change \"what and why\"]   e.g. M0-3"
disable-model-invocation: true
---

# /spec $ARGUMENTS

Read `docs/specs/README.md` first. Then do one of the following, depending on the arguments.

## `/spec <id>`: draft a new spec, or revise a draft

1. Gather the inputs:
   - the checkpoint's line in `docs/dev/progress.md`
   - the relevant part of the design (`~/Projects/smve-lab/docs/library_design_explained.md`)
   - the code it builds on
   - the smve-lab sources it ports (`port-module` skill)
   - reference numbers (`golden-values` skill)
   For anything the design leaves open, use the `researcher` agent or ask the owner. Don't guess.
2. Copy `docs/specs/_template.md` to `docs/specs/<id>-<slug>.md` and fill in every section:
   - **Interfaces**: exact signatures with shapes and dtypes, file layouts, and error types. Be
     specific enough that two agents could build against them without talking to each other.
   - **Decisions**: every choice with its reason. Mark anything that differs from the design doc.
   - **Acceptance tests**: concrete and checkable, each a `- [ ]` line. Include the
     reviewer's `merge` verdict as the last one.
   - **Tasks**: ordered, each owning distinct files. Mark which ones can run in parallel
     (`/split-work` reads this table).
   - **Ponytail pass**: before writing an interface or task, ask whether this checkpoint needs
     it at all, and whether numpy/scipy/stdlib or existing laterank code already does it. Anything
     you leave out goes under **Out of scope** as `Skipped: <X>, add when <Y>`.
   - Keep it to about one page. If it's longer, the checkpoint is too big: propose splitting it.
3. Run `python3 scripts/specs.py check`.
4. Explain the spec to the owner in plain words: the goal, the 2–3 decisions that matter most,
   and anything you're unsure about. Then **stop**. Don't approve it yourself.

## `/spec <id> approve`: only when the owner typed it

Set `status: approved` and `approved: <today's date>`. The hook will ask the owner to confirm.
If another spec is still approved, stop: it must be marked done first.

## `/spec <id> change "<what and why>"`

Edit the affected sections, then add `- <today>: <what> - <why>` under **Changes**. If the
change adds or removes an interface or an acceptance test, get the owner's OK before writing
code against it.

## `/spec <id> done`: only when the owner typed it

1. Run every acceptance test's check and the full `uv run pytest -q`.
2. Tick each passing item (`- [x]`). If any fail, stop and report. Don't mark it done.
3. Set `status: done` (the hook asks the owner to confirm), then run
   `python3 scripts/specs.py check`.
4. Update the checklist in `docs/dev/progress.md`.
