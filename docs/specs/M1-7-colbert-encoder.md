---
id: M1-7
title: ColBERT encoder and the two-model measurement
status: approved
approved: 2026-11-04
---

# M1-7: ColBERT encoder and the two-model measurement

## Goal

After this checkpoint a user can build an index with a small ColBERT model
(`Index.build(path, docs, ColBERTEncoder("answerdotai/answerai-colbert-small-v1"))`). The model
is pinned to one commit, and the manifest records every setting that changes its vectors
(prefixes, query expansion, skiplist, lengths). We then measure two candidates on SciFact
(`answerai-colbert-small-v1`, 96-d, and `mxbai-edge-colbert-v0-32m`, 64-d) so the owner can pick
the default ColBERT model by number. The code does not pick one. A small extra: the missing
`# ponytail:` comment on MUVERA's in-RAM build (carried over from M1-5).

## Out of scope

- Choosing the default model: the owner decides from the measured table, and the pick goes in the
  progress log. No `DEFAULT_ENCODER` constant. Skipped: a default, add when the owner picks one.
- PyLate: it pins `sentence-transformers==5.3.0`, and the lock has 6.2.0. Skipped: PyLate.
- GTE-ModernColBERT (149M params, 596 MB) and the others in the researcher's table.
  Skipped: more candidates, add when the two don't settle it.
- Skipped: overriding `document_length`/`query_length` (it changes the vectors; add as a keyword
  and record it when SciFact's long abstracts show truncation hurting).
- Skipped: recording the sentence-transformers version (no field pins its loading logic).
- Skipped: encoders declaring their heads, `Index.search_text` (as in M1-6).

## Interfaces

```python
# components/colbert.py (new; sentence_transformers and huggingface_hub imported inside __init__)
class ColBERTEncoder:
    def __init__(
        self,
        hf_id: str,                  # required: no default model
        revision: str | None = None, # full 40-hex sha, or None = today's main (sha recorded)
        *,
        device: str = "auto",
        batch_size: int = 32,
    ): ...
    info: EncoderInfo                # name "colbert", hf_id, revision=<sha>, dim, settings

    def encode_docs(self, texts) -> EncodedDocs   # tokens float16, dense=None, lexical=None
    def encode_query(self, text) -> Encoded       # tokens float32 (expansion rows included), text set
```

- Loads `snapshot_download(hf_id, revision, ignore_patterns=["*.onnx", "onnx/*"])` (about 134 MB
  instead of 993 MB), then `MultiVectorEncoder(str(folder), device=...)`. The sha is checked
  *before* downloading, as in `BGEM3Encoder`. Vectors are L2-normalised by the library.
- `info.settings`, read off the loaded model (not the repo defaults):
  `{"document_length", "query_length", "query_expansion" (null or the library's dict: strategy, attend, token, length),
  "query_prefix", "document_prefix", "skiplist" (sorted list of words), "skiplist_tasks" (sorted, e.g. ["document"]),
  "normalize": true, "vector_dtype": "float16"}`. The exact attribute names are found in task 1.
- `ImportError` without the extra says `pip install "laterank[colbert]"`.

```python
# components/encoder.py: ENCODERS gains "colbert"; `_colbert(info)` rebuilds from hf_id +
# revision only. load_encoder's existing `encoder.info != info` check catches any settings drift.
```

pyproject: `colbert = ["sentence-transformers>=6.0.0"]` (already in `uv.lock` through FlagEmbedding).

## Decisions

| decision | why | differs from design doc? |
|---|---|---|
| `MultiVectorEncoder` (sentence-transformers 6), not PyLate | it reads the checkpoint's own prefixes, expansion and skiplist; PyLate can't install next to ST 6 | no |
| `hf_id` has no default | CLAUDE.md: the default model is decided by measurement | no |
| Pin by `snapshot_download(revision=sha)` + local folder, ONNX files skipped | `MultiVectorEncoder` accepts a folder and skips the Hub; same trick as BGE-M3 | no |
| Settings come from the loaded model | the manifest must determine the vectors, whatever the repo's defaults are | yes: the doc lists model + revision only |
| Query keeps its expansion rows (answerai: 32 vectors) | that is how the checkpoint was trained; MaxSim sums over them | no |
| Candidates: answerai-colbert-small-v1 (SciFact 74.8 on its card, 96-d, Apache-2.0) and mxbai-edge-colbert-v0-32m (about 74.0, 64-d, Apache-2.0) | the two small, permissively licensed ones; mxbai matches the doc's 64-d storage numbers | no |
| Measured pipeline: `bm25 > maxsim@100` | the doc's default for ColBERT models; needs no dense head | no |
| Storage numbers change: 96-d is 57.6 KB/doc (about 12.8x smaller than BGE-M3), not 38 KB (19x), which is 64-d | researcher's arithmetic, 300 tokens x dim x 2 B | yes (the doc's 19x assumes 64-d) |

## Acceptance tests

- [x] AT1: with `sentence_transformers` and `huggingface_hub` stubbed (`tests/components/test_colbert.py`):
  a revision that isn't a 40-hex sha raises `ValueError` before any download; the download passes
  `ignore_patterns` for ONNX; `info` carries the sha, dim and every settings key above;
  `encode_docs` gives float16 tokens, `offsets` of length n+1 and `dense=lexical=None`;
  `encode_query` gives float32 tokens and `text`.
- [x] AT2: `load_encoder` rebuilds a `colbert` entry; a `colbert` entry without `hf_id`, and one
  whose rebuilt settings differ, raise `ValueError` (stubs, no network).
- [x] AT3: `import laterank` and every module outside `components/{bgem3,colbert}.py` import with
  no torch (`tests/test_layers.py`); `ColBERTEncoder("x")` without the extra raises the
  `ImportError` above.
- [x] AT4 (golden, `-m slow`, real models, `uv sync --extra colbert`): for each of the two models,
  5 SciFact texts give token vectors with `dim == info.dim` and unit norm (within 0.01, fp16),
  and a query returns the expansion row count its settings say.
- [x] AT5 (golden, `-m slow`): for each model, a full SciFact index with `bm25=True`; `bm25 >
  maxsim@100` has nDCG@10 >= 0.60 (a floor against broken vectors, my guess: both cards say about
  0.74 for exhaustive search; tell me if you want it different). The test prints nDCG@10,
  Recall@100, tokens per doc, KB per doc, and build seconds for both; I put the table in the
  `golden-values` skill and the progress log.
- [x] AT6: `MuveraStage.arrays` carries a `# ponytail:` comment naming the in-RAM `fdes` ceiling
  and its upgrade (a `.npy` memmap written in chunks).
- [x] AT7: `uv run pytest -q` passes and `uv run ruff check . && uv run ruff format --check .` is
  clean.
- [x] AT8: the `reviewer` agent returns `merge`.

## Tasks

| # | task | files it owns | depends on | parallel? |
|---|---|---|---|---|
| 1 | add the `colbert` extra, `uv sync --extra colbert`, then a throwaway smoke check of both models (shapes, norms, where prefixes/expansion/skiplist live on the loaded object); findings go under Changes | `pyproject.toml`, `uv.lock` | none | no |
| 2 | `ColBERTEncoder`, the `colbert` entry in `ENCODERS`, stubbed tests (AT1-AT3) | `components/colbert.py`, `components/encoder.py`, `tests/components/test_colbert.py`, `tests/components/test_encoder.py`, `tests/test_layers.py` | 1 | yes, with 3 |
| 3 | the MUVERA `# ponytail:` comment (AT6) | `components/muvera.py` | none | yes, with 2 |
| 4 | golden test: smoke check + two-model measurement (AT4, AT5) | `tests/golden/test_colbert.py` | 2 | no |

## Changes

- 2026-11-04: task 1 done (smoke check on the real models, sentence-transformers 6.2.0, CPU). Both
  pinned shas download (answerai 10 files, mxbai 17), dim 96 and 64, token vectors unit-norm to
  1e-7. Where the settings live on the loaded model `m` (a `MultiVectorEncoder`):
  `m[0]` is the `Transformer`: `.document_length`, `.query_length` (answerai: `None`, mxbai: 48),
  `.query_expansion` (answerai `{'strategy': 'fixed', 'attend': False, 'token': None, 'length':
  32}`, mxbai `None`); `m.prompts` (`{'query': ..., 'document': ...}`: answerai `'[unused0] '` /
  `'[unused1] '`, mxbai `'[Q] '` / `'[D] '`); the `MultiVectorMask` child: `.skiplist_words`
  (32 punctuation marks in both) and `.skiplist_tasks` (`['document']`). `query_expansion` is
  stored as the library holds it, including its `token` key. A short mxbai query gave 8 vectors
  (no expansion), an answerai one gave 32. mxbai's `Transformer` lowercases (`do_lower_case`);
  that's the model's own business and isn't a laterank setting.
- 2026-11-04: Interfaces now list the `token` key of `query_expansion` (the code stored it from the
  start; task 1's note had it, the Interfaces line was stale) - reviewer finding 2.
- 2026-11-04: `settings` gains `skiplist_tasks` (owner OK, reviewer finding 3) - a checkpoint that
  skipped punctuation in queries too would otherwise be recorded as "documents only" with
  different vectors. Both pinned models give `['document']`.
