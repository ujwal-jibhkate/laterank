---
id: M1-2
title: Lexical and hybrid first stages
status: done
approved: 2026-10-11
---

# M1-2: Lexical and hybrid first stages

## Goal

Two more first stages for BGE-M3 indexes. `lexical` scores BGE-M3's learned sparse token
weights over an inverted index. `hybrid` scores `dense + 0.3 · lexical`, which is the design's
default first stage for BGE-M3. After this checkpoint, `hybrid > maxsim@100` runs end to end
and reproduces smve-lab (SciFact 0.6977). It is also the first stage that is *composed* from
other stages instead of storing its own arrays, which sets the pattern for later ones.

## Out of scope

- Turning text into lexical weights (BGE-M3's sparse head): M1-6, encoders. Here the weights
  arrive precomputed.
- Skipped: choosing `alpha` per index or per search (`hybrid(alpha=0.5)`). Add it with stage
  parameters in pipeline strings when `tune` (M2) needs it. The manifest records `alpha` so a
  later change stays readable.
- Skipped: picking the default pipeline from the stages that were built (`hybrid > maxsim@100`
  when both heads exist). Add it with presets in M1-6. `Index.create` keeps its M1-1 default.
- Skipped: WAND/MaxScore pruning. Exhaustive term-at-a-time scoring is fast at this scale.
  Add it when a latency measurement asks for it.
- NFCorpus and ArguAna golden tests: a later M1 checkpoint before gate G1. They need qrels
  fixtures and ArguAna's `ignore_identical_ids`.
- Multi-segment indexes: still 0.2. Lexical weights have no corpus-wide statistics, so
  segments need no special handling here (unlike BM25 in M1-3).

## Interfaces

```python
# components/base.py: Encoded gains one optional field
@dataclass(frozen=True)
class Encoded:
    tokens: np.ndarray
    dense: np.ndarray | None = None
    lexical: tuple[np.ndarray, np.ndarray] | None = None  # (terms (m,) int, weights (m,) float)
    # stored as (int64, float32) arrays; ValueError for unequal lengths, non-1-D, wrong kinds.
    # m = 0 is allowed (every document then scores 0).


# components/dense.py: two changes
class DenseStage:
    def scores(self, q: Encoded) -> np.ndarray: ...  # (n_docs,) float32; search = top_k of it
    @classmethod
    def from_segment(cls, seg: Segment, cfg: dict, n_docs: int) -> "DenseStage": ...

    # was from_segment(seg). ValueError when the stored shape isn't (n_docs, cfg["dim"]).


# components/lexical.py
class LexicalStage:  # BGE-M3 lexical weights over an inverted index
    def __init__(self, indptr: np.ndarray, rows: np.ndarray, weights: np.ndarray, n_docs: int):
        ...
        # the arrays of kernels.inverted.to_postings; vocab = len(indptr) - 1
        # ValueError for inconsistent arrays or a row >= n_docs

    @staticmethod
    def arrays(doc_matrix: sp.spmatrix | sp.sparray) -> dict[str, np.ndarray]:
        ...
        # (n_docs, vocab) -> {"lexical/indptr.npy", "lexical/rows.npy", "lexical/weights.npy"}

    @classmethod
    def from_segment(cls, seg: Segment, cfg: dict, n_docs: int) -> "LexicalStage":
        ...
        # ValueError when vocab != cfg["vocab"]

    def scores(self, q: Encoded) -> np.ndarray:
        ...  # (n_docs,) float32
        # ValueError if q.lexical is None or a term id is outside [0, vocab)

    def search(self, q: Encoded, k: int) -> Candidates: ...


# components/hybrid.py
HYBRID_ALPHA = 0.3  # the BGE-M3 paper's value; smve-lab's golden runs use it


class HybridStage:  # stores nothing; reuses the dense and lexical stages
    def __init__(self, dense: DenseStage, lexical: LexicalStage, alpha: float): ...
    def scores(
        self, q: Encoded
    ) -> np.ndarray: ...  # dense.scores(q) + float32(alpha) * lexical.scores(q)
    def search(self, q: Encoded, k: int) -> Candidates: ...


# orchestration/index.py
class Index:
    @classmethod
    def create(
        cls,
        path,
        docs,
        tokens,
        *,
        encoder,
        dense=None,
        lexical: sp.spmatrix | sp.sparray | None = None,  # (n_docs, vocab) float
        pipeline: str = "dense > maxsim@100",
    ) -> "Index": ...

    # dense -> stage "dense"; lexical -> stage "lexical"; both -> also stage "hybrid"
```

```text
manifest.json stages (all three when both heads are given):
  "dense":   {"type": "dense", "dim": 1024}
  "lexical": {"type": "lexical", "vocab": 250002}
  "hybrid":  {"type": "hybrid", "alpha": 0.3}
seg-000001/lexical/indptr.npy    (vocab + 1,) int64
seg-000001/lexical/rows.npy      (nnz,) int32, ascending within a term
seg-000001/lexical/weights.npy   (nnz,) float32
```

## Decisions

| decision | why | differs from design doc? |
|---|---|---|
| Lexical weights are stored as postings (CSC, from `kernels.inverted.to_postings`), not CSR | search asks "which documents contain term t?"; the kernel already exists and is tested | no |
| `hybrid` stores nothing: it is built at open from the `dense` and `lexical` stages | no duplicated data; the manifest entry holds only `alpha` | no |
| `hybrid` is built automatically whenever both `dense` and `lexical` are given | it costs nothing to store, and it is the design's BGE-M3 default | no |
| `alpha = 0.3`, fixed, recorded in the manifest | the BGE-M3 paper's value, and the one smve-lab's golden numbers used (not tuned on test data) | no |
| Stages get `scores(q)` (every document) beside `search(q, k)` | hybrid adds two full score vectors; `search` is then `top_k(scores)` | adds a method |
| `from_segment(seg, cfg, n_docs)`: each stored stage checks its own arrays against the manifest and the document count | `Index` stops knowing each stage's array shapes; it turns the `ValueError` into `IndexFormatError` (as in M1-1) | changes the M1-1 signature |
| A query term outside the vocabulary is a `ValueError`, not silently dropped | it means the query came from a different model than the index | no |
| Query weights may repeat a term; repeats add up | equals the sparse product smve-lab computed after `sum_duplicates` | no |
| The golden fixture reads smve-lab's stored sparse arrays (`*_sparse_{indptr,ids,weights}.npy`, vocab 250,002) | the same inputs smve-lab scored; no re-encoding | no |

## Acceptance tests

- [x] AT1: `LexicalStage.scores` equals `Q @ D.T` from scipy (rtol 1e-5) on random sparse data,
  including a query with repeated terms and one with no terms. `search` breaks ties by lower
  row and returns every document when k is larger than the corpus. A term id ≥ vocab or a
  missing `q.lexical` raises `ValueError`.
- [x] AT2: `HybridStage.scores` equals `dense.scores + float32(0.3) * lexical.scores` exactly, and
  its `search` matches a naive sort by (score descending, row).
- [x] AT3: `Encoded(lexical=...)` accepts lists and stores int64/float32 arrays. It raises
  `ValueError` for unequal lengths, 2-D input and float term ids.
- [x] AT4: `Index.create` with dense and lexical writes the three manifest stages above.
  `search` with `lexical`, `hybrid` and `hybrid > maxsim@20` equals `Pipeline.run` on the
  in-memory stages. With only `lexical`, no `hybrid` stage exists, and asking for it lists
  `['lexical']`.
- [x] AT5: `Index.open` raises `IndexFormatError` when a lexical row is ≥ the document count,
  when the stored vocab differs from the manifest, and when a `hybrid` stage lacks a `dense`
  or `lexical` stage.
- [x] AT6: golden SciFact, ±0.002 (`results/cross_dataset/summary.md`), nDCG@10 / Recall@100:
  `lexical` 0.6376 / 0.8980, `hybrid` 0.6842 / 0.9237, `lexical > maxsim@100` 0.7048 / 0.8980,
  `hybrid > maxsim@100` 0.6977 / 0.9237.
- [x] AT7: `uv run pytest -q` and `uv run ruff check .` pass on all three CI operating systems.
- [x] AT8: the reviewer agent's verdict is `merge`.

## Tasks

| # | task | files it owns | depends on | parallel? |
|---|---|---|---|---|
| 1 | `Encoded.lexical`, `DenseStage.scores` + new `from_segment`, `LexicalStage` | `components/{base,dense,lexical}.py`, `tests/components/test_{dense,lexical}.py` | M1-1 | no (others need it) |
| 2 | `HybridStage` | `components/hybrid.py`, `tests/components/test_hybrid.py` | 1 | no (small) |
| 3 | `Index` wiring | `orchestration/index.py`, `tests/orchestration/test_index.py` | 1, 2 | yes, with 4 |
| 4 | golden tests | `tests/golden/{conftest,test_scifact}.py` | 1, 2 | yes, with 3 |

## Changes

- 2026-10-11: `Encoded.dense` refuses NaN/inf, like `Encoded.lexical` (a NaN query made every
  dense and hybrid score NaN, and search returned no hits). - reviewer follow-up, owner OK
- 2026-10-11: review round 2: the lexical finiteness check now runs on the converted float32
  weights, so every scipy sparse format works (round 1's check broke `lil` and `dok`). -
  reviewer verdict "fix first"
- 2026-10-11: review fixes, no interface added or removed. `Index.open` requires the hybrid
  `alpha` to be a finite int/float (not bool, null or a string). `LexicalStage` checks its
  arrays' dtypes (int indptr/rows, float weights). `Index.create` refuses NaN/inf/complex
  lexical weights and NaN/inf dense vectors. `Encoded.lexical` gives one clear `ValueError` for
  non-pairs, complex and non-finite weights. Dropped the duplicate sparse-matrix check in
  `LexicalStage.arrays` (`Index.create` checks it). Recorded: `hybrid` finds its partners by the
  stage names `dense` and `lexical`. Added the `lexical` Recall@100 row to the `golden-values`
  skill. - reviewer verdict "fix first"
- 2026-10-11: approved as drafted.
- 2026-10-11: drafted.
