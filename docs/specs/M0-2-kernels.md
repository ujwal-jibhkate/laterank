---
id: M0-2
title: Kernels
status: approved
approved: 2026-10-03
---

# M0-2: Kernels

Written after the fact on 2026-10-03, when the spec workflow was introduced. The owner's M0
scaffold prompt (checkpoint 2) served as the approved plan. The code was already built, so the
owner should review this spec before marking it done.

## Goal

The numeric core, ported from smve-lab to numpy/scipy only: MaxSim, top-k, inverted-index
scoring, SMVE encoding, a portable random generator for anchors, and ranking metrics. Everything
above the kernels builds on these.

## Out of scope

- MUVERA FDE and BM25 (M1).
- P, MRR and MAP metrics (M2).
- Caching anchor matrices on disk (M1, see Decisions).

## Interfaces

```python
# kernels/maxsim.py
maxsim_scores(...)              # ragged (flat, offsets) queries x docs -> scores
subset_ragged(...)              # pick a subset of ragged items
# kernels/topk.py
top_k(scores, k) -> np.ndarray  # indices; 1-D or 2-D; ties go to the lower index
# kernels/inverted.py
ragged_ranges(starts, lengths); to_postings(doc_matrix) -> (indptr, rows, weights)
postings_read(indptr, terms) -> int; score_postings(...)
# kernels/smve.py
topk_blocks(P, k, reps=1); smve_single(...); smve_encode(...) -> CSR; token_mean(flat, sample, seed)
# kernels/rng.py
splitmix64(seed, start, stop); standard_normal(seed, start, stop)
make_anchors(d, w, seed=0, chunk_columns=4096) -> (d, w) float32; anchor_sha256(B) -> str
sample_indices(n, size, seed=0)
# kernels/metrics.py
dcg(...); ndcg_at_k(ranked, qrel, k); recall_at_k(ranked, qrel, k); mean_metric(...)
```

## Decisions

| decision | why | differs from design doc? |
|---|---|---|
| RNG computes log/sin/cos with its own polynomials (only + − × ÷ √) | IEEE-754 makes those bit-identical everywhere; `np.log`/`np.cos` can differ in the last bit by platform | no (refines it) |
| anchors generated column by column | a wider matrix starts with the narrower one | no |
| `anchor_sha256` hashes the whole float32 matrix | simpler and stricter | **yes**: the design said "first anchor block" |
| `top_k` breaks ties by lower index | `argpartition` alone is arbitrary at the k-th place | no |
| metrics live in `kernels/metrics.py` | no `eval/` layer exists yet | no |
| `token_mean` samples with `sample_indices` (portable RNG) and averages in float64 | `np.random` streams may change between numpy versions; float64 is more accurate. The centre μ differs slightly from smve-lab's | no |
| each RNG purpose gets its own stream: anchors use the seed directly, `sample_indices` uses a seed derived from (seed, "sample") | the token sample and the anchors must not read the same random bits | no |
| `score_postings` returns float32 (accumulates in float64) | one score dtype across scorers; smve-lab returned float64. Rare extra ties are broken by lower index; the M1 BM25 golden test confirms this stays inside ±0.002 | no |
| SMVE encodes a text with no tokens to all zeros | matches smve-lab | no |
| `top_k` never negates or converts scores | negating unsigned ints wraps around, and converting int64 to float64 loses precision above 2**53; every numeric dtype is now ranked exactly | no |
| `maxsim_scores` checks `offsets[-1] == len(flat)` | a mismatch otherwise scores silently from the wrong tokens | no |

## Acceptance tests

- [x] AT1: every kernel matches a slow, obvious reference loop on small random inputs.
- [x] AT2: hypothesis property tests: MaxSim unchanged under doc-token shuffle, and never lower
  when a doc token is added.
- [x] AT3: a frozen RNG test vector, so any change in the generator's output fails.
- [x] AT4: `grep -rn "import torch" src/laterank/kernels` finds nothing.
- [x] AT5: `uv run pytest -q` and `uv run ruff check .` pass.
- [x] AT6: the reviewer agent's verdict is `merge`.

## Tasks

| # | task | files it owns | depends on | parallel? |
|---|---|---|---|---|
| 1 | port kernels + tests | `src/laterank/kernels/`, `tests/kernels/` | M0-1 | done before specs existed |

## Changes

- 2026-10-03: reviewer verdict "fix first". Fixed: SMVE crashed on an empty document alone in
  a chunk (int64 `bincount`); `top_k` was wrong for unsigned/bool scores; `maxsim_scores`
  accepted offsets that don't cover the token array; the kernels docstring matched AT4's grep.
  Added separate RNG streams per purpose (changes the frozen `sample_indices` value, never
  released), a frozen SHA-256 of 10**6 float64 normals, and an odd-d anchor layout test.
  Recorded the `token_mean`, stream, and float32-scores decisions above. - review findings
- 2026-10-03: re-review verdict "merge", with one low finding: the first `top_k` fix (convert to
  float64) lost int64 precision above 2**53. `top_k` now never negates or converts scores. -
  re-review finding
- 2026-10-03: spec written after the fact. Building the largest anchor matrix (1024 × 131,072)
  takes about 4.5 s, so M1 should avoid regenerating it on every `open`.
