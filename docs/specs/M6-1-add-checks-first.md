---
id: M6-1
title: add checks documents before the model loads
status: approved
approved: 2026-11-12
---

# M6-1: add checks documents before the model loads

## Goal

`laterank add` refuses bad documents (empty file, an id already in the index, an id repeated in
the file, an empty `_id`) before the model loads, so a mistake costs under a second instead of
about 7 s (mxbai, cached) or longer (first download). Python users get the same: `index.add(docs)`
loads the index's own model only after the checks pass.

## Out of scope

- Any change to what `add` checks or to the rewrite itself (M5-2). The checks already exist in
  `Index.add` and already run before encoding; only the order relative to the model load moves.
- Skipped: a message that fits `add` on an index built with `Index.create` (no encoder), add when
  a user hits it (M5-3's Skipped line stands). With `encoder=None` it now fails after the checks,
  still one line, exit 1.
- Skipped: `delete --ids FILE`, add when a user hits the shell's argument-length limit.

## Interfaces

`src/laterank/orchestration/index.py`: `encoder` becomes optional.

```python
def add(self, docs: Sequence[Mapping], encoder: Encoder | None = None) -> None:
    """... encoder: None (default) loads the index's own model with
    load_encoder(self.manifest.encoder), after every check below has passed."""
    # existing checks, unchanged, then:
    if encoder is None:
        encoder = load_encoder(self.manifest.encoder)
    # existing encoder.info check, encode, rewrite: unchanged
```

`load_encoder` is imported from `laterank.components.encoder` (orchestration → components: a
downward import, allowed). Its `ValueError`s (unknown encoder name, model mismatch) pass through.

`src/laterank/api/cli.py`, `_add`:

```python
def _add(args) -> int:
    index = Index.open(args.index)
    docs = load_corpus(args.docs)
    if not docs:
        raise ValueError(f"{args.docs} has no documents")
    index.add(docs)  # checks first, then loads the model
    print(...)  # unchanged
    return 0
```

`README.md`, "Adding and deleting documents": the snippet keeps passing `encoder` (it continues
the Quick start, where `encoder` is already loaded); one sentence adds that `index.add(new_docs)`
loads the index's own model.

## Decisions

| decision | why | differs from design doc? |
|---|---|---|
| Make `Index.add`'s `encoder` optional (None loads the manifest's model after the checks), instead of a public `check_docs` method or a private call from `cli.py` | no new method; `cli.py` already reaches into `_pipeline` twice and M3-2's rule allows no third private access; a Python user also gets the fast refusal | no |
| Keep the `encoder` argument | the Quick start and every M5-2 test pass one; removing it would break 0.2.0 callers | no |
| The empty-file check lives in `_add`, not `Index.add` | `Index.add`'s message talks about `{'id': ...}` mappings; a file user needs the file's name. One line | no |

## Acceptance tests

Tests in `tests/api/test_cli.py` (fake encoder, model load replaced by `pytest.fail`) and
`tests/orchestration/test_add_delete.py`.

- [ ] AT1: `laterank add` refuses each of these with one stderr line, exit 1, the index folder
  byte-unchanged, and the model never loaded: an empty `--docs` file (message names the file); an
  id already in the index; an id repeated in the file; an empty `_id`. (The existing `add-clash`
  case flips from `loads=True` to `False`.)
- [ ] AT2: `index.add(docs)` with no encoder adds the documents and equals `index.add(docs,
  encoder)` (same doc ids and same top hit for one query); `index.add(bad_docs)` raises
  `ValueError` without loading the model.
- [ ] AT3: the existing M5-2/M5-3 tests pass unchanged (`laterank add` happy path still loads the
  model once).
- [ ] AT4: `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .` pass; CI green.
- [ ] AT5: the `reviewer` agent returns `merge`.

## Tasks

| # | task | files it owns | depends on | parallel? |
|---|---|---|---|---|
| 1 | `Index.add(encoder=None)` + AT2 test | `src/laterank/orchestration/index.py`, `tests/orchestration/test_add_delete.py` | - | no |
| 2 | `_add` empty check, `index.add(docs)` + AT1 cases | `src/laterank/api/cli.py`, `tests/api/test_cli.py` | 1 | no |
| 3 | README sentence | `README.md` | 1 | no |
| 4 | reviewer (AT5), commit on the owner's OK; owner pushes, CI (AT4) | none | 1-3 | no |

## Changes
