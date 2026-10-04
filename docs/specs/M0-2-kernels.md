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

## Acceptance tests

- [ ] AT1: every kernel matches a slow, obvious reference loop on small random inputs.
- [ ] AT2: hypothesis property tests: MaxSim unchanged under doc-token shuffle, and never lower
  when a doc token is added.
- [ ] AT3: a frozen RNG test vector, so any change in the generator's output fails.
- [ ] AT4: `grep -rn "import torch" src/laterank/kernels` finds nothing.
- [ ] AT5: `uv run pytest -q` and `uv run ruff check .` pass.
- [ ] AT6: the reviewer agent's verdict is `merge`.

## Tasks

| # | task | files it owns | depends on | parallel? |
|---|---|---|---|---|
| 1 | port kernels + tests | `src/laterank/kernels/`, `tests/kernels/` | M0-1 | done before specs existed |

## Changes

- 2026-10-03: spec written after the fact. Building the largest anchor matrix (1024 × 131,072)
  takes about 4.5 s, so M1 should avoid regenerating it on every `open`.
