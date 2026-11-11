---
id: M5-3
title: laterank add / delete, README, 0.2.0
status: done
approved: 2026-11-10
---

# M5-3: laterank add / delete, README, 0.2.0

## Goal

The command line can add documents to an index and delete them (`Index.add`/`Index.delete`, M5-2),
the README shows both from Python and from the command line, and `0.2.0` puts them on PyPI.

## Out of scope

- Any change to how `Index.add`/`Index.delete` work (M5-2 is done). Only a wrong number in the
  `add` docstring is fixed (see Interfaces).
- Skipped: `add --dataset`, add when someone needs to add a BEIR corpus to an existing index;
  `--docs folder/corpus.jsonl` already does it.
- Skipped: `delete --ids FILE`, add when a user hits the shell's argument-length limit;
  `laterank delete IDX $(cat ids.txt)` covers long lists until then.
- Skipped: checking `add`'s documents before the model loads. An empty file, an id already in the
  index, an id repeated in the file or an empty `_id` fails after the load (about 7 s with mxbai
  once cached), and the empty-file message says `{'id': ...}` where the file has `_id`. Add the
  checks when that wait matters to someone.
- Skipped: a message that fits `add` when it meets an index built with `Index.create`
  (no encoder). `load_encoder`'s message advises passing `Encoded` queries to `Index.search`, which
  doesn't fit `add`; it is still one line, exit 1. Reword it when a user hits it.
- Skipped: Trusted Publishing. M4-2 planned it "at 0.1.0", and this release skips 0.1.0. Add it
  when a second person releases, or when uploads move into CI.

## Interfaces

`src/laterank/api/cli.py`, two new subcommands. Errors go through `main`'s existing handler: one
line on stderr, exit code 1, no traceback.

```text
laterank add INDEX --docs FILE      # FILE: JSONL rows `_id`, `text`, optional `title` (as `build`)
laterank delete INDEX ID [ID ...]   # an id starting with "-" goes after "--"
```

```python
def _add(args) -> int:
    index = Index.open(args.index)  # a bad folder fails first
    docs = load_corpus(args.docs)  # a missing file or bad JSON fails before the model loads
    index.add(docs, load_encoder(index.manifest.encoder))
    print(
        f"added {len(docs)} documents to {args.index}: "
        f"{index.manifest.segments[0].documents} in total"
    )
    return 0


def _delete(args) -> int:  # never loads the model
    index = Index.open(args.index)
    index.delete(args.ids)
    print(
        f"deleted {len(set(args.ids))} documents from {args.index}: "
        f"{index.manifest.segments[0].documents} in total"
    )
    return 0
```

Help text says each command rewrites the whole index. The module docstring lists six commands.

`src/laterank/orchestration/index.py`, docstring only: `Index.add` says "without SMVE about
10 s"; the M5-2 measurement (354 s = SMVE 318 + MUVERA 26 + write 9) gives about 36 s without
SMVE and about 10 s without SMVE and MUVERA. Fix the sentence to say that.

`README.md`:
- the "Command line" block gains `laterank add ./idx --docs new.jsonl` and
  `laterank delete ./idx d1 d2`;
- a new section "Adding and deleting documents" (after "Choosing a pipeline"):
  `index.add(new_docs, load_encoder(index.manifest.encoder))`, `index.delete(["a"])`; each call
  rewrites the whole index, so the result equals a fresh build; the measured cost (SciFact,
  BGE-M3: 354 s with SMVE, about 10 s without SMVE and MUVERA); any other open `Index` on the
  folder must be reopened;
- the banner "may change before 0.1.0" becomes "may still change before 1.0".

Release: the same steps as M4-2/M5-1, with tag `v0.2.0` and `dist/laterank-0.2.0*`. The owner
tags, uploads with the project-scoped token, and pushes the tag after the upload.

## Decisions

| decision | why | differs from design doc? |
|---|---|---|
| `add` takes only `--docs FILE`, the same JSONL rows as `build` | one way in, already parsed by `load_corpus`; a flag (not a positional) leaves room for `--dataset` later | no |
| `add` has no `--encoder`/`--hf-id`/`--revision`; it uses `load_encoder(index.manifest.encoder)` | `Index.add` refuses any other model anyway; the manifest pins the exact commit | no |
| `delete` takes ids as arguments, not a file | deletes are usually a few ids, and the shell turns a file into arguments | no |
| Print one line with the new total, read from `manifest.segments[0].documents` | public data, so `cli.py` gains no third private access (M3-2's rule) | no |
| No progress notice for the rewrite | `build` has none either; help text and README state the cost | no |
| Version `0.2.0`, not `0.1.0` or a beta (owner, 2026-11-10) | the M5 checklist names `0.2.0`, and CLAUDE.md puts add/delete in 0.2; the Beta classifier stays, so 0.x still signals an unstable API | **yes**: 0.1.0 final never ships |

## Acceptance tests

Tests in `tests/api/test_cli.py`, with the existing fake encoder (no model, no network).

- [x] AT1: `build` 12 docs, then `add --docs` 2 new ones: exit 0, the stdout line above with
  `14 in total`, the new ids are last in `Index.open(...)`, and `search` finds one of them.
- [x] AT2: `delete IDX d3 d5`: exit 0, `10 in total`, both ids gone; the model is never loaded.
- [x] AT3: each refusal is one stderr line, exit 1, index folder byte-unchanged: `add` with an id
  already in the index; `add` with a missing `--docs` file (the model never loads); `delete` with
  an unknown id; `delete` of every id.
- [x] AT4: `laterank --help` lists the six commands.
- [x] AT5 (manual, numbers recorded in the log): on a real SciFact index (mxbai, `bm25 > maxsim@100`),
  `add` 2 documents, `search` finds one, `delete` them, then `evaluate` prints nDCG@10 0.7459 again
  (M5-2 proved delete-after-add equals the original build byte for byte).
- [x] AT6: `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .` pass; CI is
  green on every job for the commit to be tagged.
- [x] AT7: from the tag, `dist/` holds exactly the `0.2.0` sdist and wheel and `twine check`
  passes. After the upload, in a new venv outside the repo, `pip install "laterank[colbert]"`
  installs `0.2.0`, the quick-start ranks doc `a` first, and `laterank add --help` runs.
- [x] AT8: the `reviewer` agent returns `merge` (before the tag).

## Tasks

| # | task | files it owns | depends on | parallel? |
|---|---|---|---|---|
| 1 | `add`/`delete` subcommands + AT1-AT4 tests | `src/laterank/api/cli.py`, `tests/api/test_cli.py` | - | yes (with 2) |
| 2 | README section, command block, banner; `Index.add` docstring number | `README.md`, `src/laterank/orchestration/index.py` | - | yes (with 1) |
| 3 | AT5 on real SciFact | none | 1 | no |
| 4 | reviewer (AT8), commit on the owner's OK; owner pushes, CI (AT6) | none | 1, 2, 3 | no |
| 5 | owner tags, builds, uploads, pushes the tag; verify from PyPI (AT7) | none | 4 | no |

## Changes

- 2026-11-10: the README banner's label also changes, "Pre-release" to "Beta" - `0.2.0` is not a
  pre-release version, so the old label would be false; "Beta" matches the classifier.
- 2026-11-10 (reviewer round 1, "fix first", 8 findings): README's snippet continues the Quick
  start (its `encoder`, `delete(["b"])`, so `a` stays the top hit) instead of the Interfaces'
  `load_encoder(...)` and `["a"]`; `load_encoder` is named in the text. README's reopen rule now
  covers other `Index` objects in the same program, and its advice says one `add` plus one
  `delete` (an update is both). AT2 passes a repeated id. Docstring states the 36 s too. Two
  reviewer findings (docs checked after the model load; the precomputed-index message) became
  Skipped lines, no code (Claude's default, owner may override). `_add`'s comment narrowed to
  what it checks. AT5's numbers written to the progress log.
- 2026-11-10 (reviewer round 2): "merge"; 3 wording nits in the log and this spec, fixed.
