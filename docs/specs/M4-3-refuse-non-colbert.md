---
id: M4-3
title: Refuse non-ColBERT models
status: done
approved: 2026-11-08
---

# M4-3: Refuse non-ColBERT models

## Goal

`ColBERTEncoder` stops with a clear error when the checkpoint is not a ColBERT model. Today
sentence-transformers accepts such a checkpoint and adds a **randomly initialised** projection,
so the build succeeds and the search results mean nothing. Measured on BGE-M3 through
`MultiVectorEncoder`: two loads of the same commit give token vectors with mean cosine -0.09
(mxbai, a real ColBERT model: 1.0000). The index and its queries would use different random
projections, and `load_encoder`'s settings comparison can't see it. The beta must not ship this.

## Out of scope

- Skipped: checking before the weights download (a config-only `snapshot_download` first), add when
  a wasted download bothers someone. A wrong model still downloads once (BGE-M3: 4.3 GB cached).
- Skipped: an override to allow non-ColBERT models, add when someone trains a projection here.
- Skipped: native retrieval models (`architectures` ending in `ForRetrieval`, e.g. ColPali), add
  when laterank handles images. They are refused: untested, and laterank is text-only.
- Not promised: that every accepted model searches well. Only mxbai and answerai have been measured.
- M4-2 (PyPI) and the M4-1 reviewer's low findings.

## Interfaces

```python
# components/colbert.py (json is stdlib; nothing else new)
def _is_colbert_checkpoint(folder: Path) -> bool:
    """Does this snapshot folder hold trained per-token vectors? Reads files, loads nothing."""


# ColBERTEncoder.__init__, right after snapshot_download and before MultiVectorEncoder(...):
# not a ColBERT checkpoint -> ValueError, whose message names `hf_id`, says the token vectors
# would be meaningless, gives a ColBERT model as an example and points BGE-M3 to its own encoder.
```

The rule, compared on each module's last dotted `type` segment (exact match, so `TokenPooling`
variants are not `Pooling`):

| `modules.json` | other condition | result | real example (checked 2026-11-08) |
|---|---|---|---|
| absent | `config.json` `architectures` has `HF_ColBERT` | accept | answerai, colbert-ir/colbertv2.0, JaColBERT |
| absent | anything else, or no `config.json` | refuse | bert-base-uncased |
| present | has `Dense`, no `Pooling`, `model_type` allowed | accept | mxbai, lightonai/GTE-ModernColBERT-v1 |
| present | has `Pooling`, or no `Dense`, or any other `model_type` | refuse | BGE-M3, all-MiniLM-L6-v2, distiluse |

`model_type` comes from `config_sentence_transformers.json`. Allowed: absent or null,
`SentenceTransformer`, `ColBERT`, `MultiVectorEncoder`. For any other value the library discards
the declared modules and builds a random projection.

## Decisions

| decision | why | differs from design doc? |
|---|---|---|
| Check the checkpoint's files, not the loaded model | loading erases the evidence: `Pooling` is dropped, and a random `Dense` looks like a trained one | no |
| Decide by the declared modules; `model_type` can only refuse | LightOn's PyLate models have no `model_type`, so requiring one refuses real ColBERT models; a foreign one makes the library discard the modules | no |
| Refuse `Pooling` even with a `Dense` head (distiluse) | that head was trained on whole-sentence vectors; the library would reuse it on tokens | no |
| The check lives in `ColBERTEncoder.__init__`, not the CLI | the CLI, `Index.build` users and `load_encoder` all pass through it | no |
| `ValueError` | same as the bad-revision error; `cli.main` already prints it and exits 1 | no |
| An index already built from a non-ColBERT model is refused at `load_encoder` | its random projection is lost, so correct queries can't be made | no |

## Acceptance tests

- [x] AT1: `_is_colbert_checkpoint` gives the table's result for each row, using real-shape files
  in `tmp_path`, including a `HierarchicalTokenPooling` module (accepted), null `architectures`
  (refused), each allowed `model_type` (accepted) and `CrossEncoder` (refused).
- [x] AT2: with the stubs, a non-ColBERT folder makes `ColBERTEncoder` raise `ValueError` naming the
  `hf_id` before `MultiVectorEncoder` is constructed; `load_encoder` on such an entry raises too.
- [x] AT3 (golden, `-m slow`, needs `--extra colbert` and the cached BGE-M3): `laterank build` with
  `--encoder colbert --hf-id BAAI/bge-m3 --revision <pinned sha>` returns 1, prints the error, and
  creates no index folder.
- [x] AT4 (golden, `-m slow`): `tests/golden/test_colbert.py` still passes for mxbai and answerai.
- [x] AT5: `uv run pytest -q` passes and `uv run ruff check . && uv run ruff format --check .` is clean.
- [x] AT6: the `reviewer` agent returns `merge`.

## Tasks

| # | task | files it owns | depends on | parallel? |
|---|---|---|---|---|
| 1 | `_is_colbert_checkpoint`, the refusal, the stub fixture writes a ColBERT `modules.json`, AT1, AT2 | `components/colbert.py`, `tests/components/test_colbert.py` | none | no |
| 2 | the real-model refusal through the CLI (AT3), run AT4 | `tests/golden/test_colbert.py` | 1 | no |

## Changes

- 2026-11-08: the `modules.json` rows also check `model_type`; AT1 gains those cases (owner OK) -
  reviewer finding 1: for a foreign `model_type` such as `CrossEncoder` the library discards the
  declared modules and adds a random projection, which the first rule accepted.
- 2026-11-08: AT6's `merge` verdict predates the `model_type` fix above. The fix is the one the
  reviewer proposed; there was no second review pass (owner closed the spec knowing this).
