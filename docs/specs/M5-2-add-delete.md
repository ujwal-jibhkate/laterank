---
id: M5-2
title: Index.add and Index.delete
status: approved
approved: 2026-11-10
---

# M5-2: Index.add and Index.delete

## Goal

An existing index can gain and lose documents, and afterwards it is identical to an index freshly
built from the remaining documents (same arrays, same search results). Each change rewrites the
whole index into one new segment (owner's option A, 2026-11-10), so `open`, the manifest format
and the crash-safety rules stay as they are.

## Out of scope

- CLI `laterank add` / `delete`, README section, `0.2.0` release: M5-3.
- Multi-segment indexes, deleted-rows lists, merges (research option B). Option A's rewrite is
  B's `compact()`, so nothing here is thrown away if B comes later.
- Skipped: `add` for indexes built from precomputed vectors (`Index.create`, no encoder), add
  when a user asks; they can rebuild with `create`.
- Skipped: update/upsert (replace a document by id), add when asked; `delete` then `add` does it
  (two rewrites).
- Skipped: reusing unchanged per-document arrays (MUVERA, 26 s on SciFact) instead of
  recomputing, add when an index without SMVE needs faster adds (owner chose not to, 2026-11-10).
- Skipped: a lock against two writers at once, add when concurrent writers are a real use.
- Skipped: streaming the rewrite. The token matrix is held in RAM, as `build` already does
  (3.6 GiB for SciFact, about 20 times more at 100k docs). Marked `# ponytail:` in `_rewrite`.
  Add a chunked `.npy` memmap when a corpus doesn't fit in RAM.

## Interfaces

`orchestration/index.py`, two public methods. Both rewrite the index, then update `self` in place
(`self` now reads the new segment), so search works straight away without reopening.

```python
class Index:
    def add(self, docs: Sequence[Mapping], encoder: Encoder) -> None:
        """Encode `docs` with `encoder` and add them after the existing documents.

        Rewrites the whole index: every stage is recomputed over old + new documents, so the
        result equals Index.build on (old docs in their order) + (new docs in the given order).
        Measured on SciFact with BGE-M3 (5,183 docs, all stages): 354 s, of which SMVE is 318 s;
        without SMVE about 10 s. Old documents are not re-encoded.

        Raises (all before encoding; the index is unchanged):
            ValueError: `docs` is empty or not all mappings; an id is already in the index or
                repeats within `docs`; `encoder.info != self.manifest.encoder` (another model,
                revision or settings would put the new vectors in a different space).
            ValueError: as in `create`, e.g. bm25 finds no word (raised before anything is written).
        """

    def delete(self, ids: Iterable[str]) -> None:
        """Remove the documents with these ids. The rest keep their order.

        Rewrites the whole index, as `add` does; the result equals Index.build on the remaining
        documents.

        Raises (the index is unchanged):
            ValueError: `ids` is a bare str, is empty, names an id not in the index (lists up to
                5), or would remove every document.
        """
```

Private helpers in the same file (not an interface others build on, listed so the tasks agree):
- `_stage_arrays(docs, flat, offsets, encoder, dense, lexical, *, bm25, smve, muvera) ->
  (arrays, stages)`: the body of `create` from the token checks to MUVERA, moved out unchanged.
  `create` calls it; so does the rewrite.
- `_rewrite(self, docs, flat, offsets, dense, lexical) -> None`: builds the arrays with the
  stage flags read from `self.manifest.stages`, then:
  1. `write_segment(path, n, ...)`, where `n` = 1 + the highest number of any `seg-NNNNNN*` entry
     in the folder, listed or not, so a stray folder from a crash never blocks the next number;
  2. `write_manifest` with the same encoder, default pipeline and `created_at`, the new stages
     and `segments=(new,)`: the atomic commit point;
  3. re-initialise `self` from `Index.open(path)`;
  4. remove every `seg-*` folder the new manifest doesn't list (`ignore_errors=True`: on Windows,
     open memmaps can block this; the next rewrite removes it).

How the old data is read back, per stored array:
- docs: `DocStore[row]` records, written again as they are.
- tokens: `tokens/flat.npy` + `offsets.npy`; new tokens are cast to the stored dtype.
- dense: `dense/vectors.npy` (float32).
- lexical: the postings as `sp.csc_matrix((weights, rows, indptr), shape=(n_docs, vocab))`, then
  rows selected (delete) or stacked with the new matrix (add).
- bm25, smve, muvera: recomputed from docs / tokens (bm25's vocab and SMVE's center depend on the
  whole corpus; MUVERA is recomputed for simplicity).

## Decisions

| decision | why | differs from design doc? |
|---|---|---|
| Rewrite into one new segment on every change (option A) | exact match with a fresh build, including BM25 stats and SMVE's corpus-wide center; reuses the atomic manifest swap; `open` stays single-segment | **yes**: §6 says "writes a new segment folder instead of rebuilding everything" |
| Accept the rewrite cost and write it in the docstring (owner, 2026-11-10) | SMVE is 90% (318 s of 354 s on SciFact) and can't be done in part: its center changes with any doc | no |
| One spec for add + delete; M5 checklist becomes 2 = add + delete, 3 = CLI, README, `0.2.0` | under option A both are "rewrite with another doc list"; the shared rewrite is most of the code | **yes**: changes the M5 checklist (owner's OK needed) |
| `add` takes an `Encoder` and checks `encoder.info == manifest.encoder` before encoding | same model, revision and settings as the old vectors, or the stages compare vectors from two spaces | no |
| New documents go after the old ones; delete keeps the rest's order | `top_k` breaks ties by lower row, so row order is part of "same results" | no |
| Methods return `None` and update `self` | the old segment is deleted, so the old object would point at removed files | no |
| Keep `created_at` from the first build; `created_with` becomes the rewriting laterank | the index was created then; but the rewriting version wrote every file, and `read_manifest` errors quote it as the writer | no |
| Single writer assumed, no lock. Other open `Index` objects on the folder must reopen after `add`/`delete` (said in both docstrings) | one process changes an index at a time; the old segment is deleted, so an old object's document lookups fail with `FileNotFoundError` | no |

## Acceptance tests

Tests use `FakeEncoder` (`tests/conftest.py`), with all stages on (dense, lexical, hybrid, bm25,
smve, muvera). "Identical" means: every `.npy` and `docs.jsonl` in the segment byte-equal, and the
manifests equal apart from the segment id.

- [x] AT1: `build(A).add(B)` is identical to `build(A + B)`.
- [x] AT2: `build(A + B).delete(ids of B)` is identical to `build(A)`, with B's rows taken from the
  middle as well as the end.
- [x] AT3: after `add`, `search` on the same object finds a new document (no reopen), and so does
  `Index.open(path)`. After either method only the new segment folder is left.
- [x] AT4: every refusal in Interfaces raises `ValueError` and leaves the index folder
  byte-unchanged; `add`'s refusals happen before `encode_docs` is called.
- [x] AT5: a failure inside `write_manifest` (monkeypatched) leaves the old index opening with the
  old documents, and the next `add` succeeds despite the stray segment folder.
- [x] AT6: `create` behaves as before after `_stage_arrays` is extracted (existing tests pass).
- [ ] AT7: `uv run pytest -q` and `uv run ruff check . && uv run ruff format --check .` pass; CI green.
- [x] AT8: reviewer verdict `merge`.

## Tasks

| # | task | files it owns | depends on | parallel? |
|---|---|---|---|---|
| 1 | Extract `_stage_arrays` from `create` (AT6) | `src/laterank/orchestration/index.py` | - | no |
| 2 | `_rewrite`, `add`, `delete` | `src/laterank/orchestration/index.py` | 1 | no |
| 3 | Tests AT1-AT5 | `tests/orchestration/test_add_delete.py` (new) | 2 | no |
| 4 | M5 checklist renumbered in `docs/dev/progress.md` | `docs/dev/progress.md` | - | yes |

## Changes

- 2026-11-10: "identical" manifests also ignore `created_at` - `add` keeps the first build's time
  (a Decision), so it can't equal a fresh build's timestamp.
- 2026-11-10: private `_old_heads` reads stored dense/lexical back for both methods; `add` also
  refuses (after encoding) an encoder whose dense/lexical heads differ from the index's stages -
  `FakeEncoder(dense=False)` has the same `EncoderInfo`, so the info check alone misses it.
- 2026-11-10 (reviewer round 1, owner chose 1a and 2a): the Decision that other-process readers
  keep working was false (`DocStore` reopens `docs.jsonl`); now other open `Index` objects must
  reopen, documented, no code. `created_with` records the rewriting version. Also: `add` runs
  storage's document checks before encoding; `delete` refuses bytes; `add`/`delete` drop their
  old-segment memmaps before the rewrite so Windows can delete the old folder; heads refusal tested.
