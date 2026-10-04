"""Exact ColBERT-style late-interaction (MaxSim) scoring.

For a query q with token vectors q_1..q_n and a document d with token vectors d_1..d_m:

    MaxSim(q, d) = sum_{i=1..n}  max_{j=1..m}  q_i . d_j

Every query token finds its best-matching document token; those best matches are summed. With
L2-normalized token vectors (BGE-M3, ColBERT) each dot product is a cosine similarity.

Ragged layout. Texts have different numbers of tokens, so token vectors are stored "flat":
all tokens of all texts stacked into one (T, dim) array, plus an (n + 1,) `offsets` array where
text i owns rows offsets[i]:offsets[i+1].

The naive way loops over every (query, doc) pair: millions of tiny matmuls. Instead:

  1. Take a batch of queries' tokens at once:     Q  (Tq, dim)
  2. Take a chunk of documents' tokens at once:   D  (Td, dim)
  3. One big matmul:                              S = Q @ D.T  (Tq, Td)
     S[i, j] = similarity of query token i with document token j.
  4. Max over each document's column segment  ->  (Tq, n_docs_in_chunk)
  5. Sum over each query's row segment        ->  (n_queries, n_docs_in_chunk)

Steps 4-5 use `np.maximum.reduceat` / `np.add.reduceat`, which reduce consecutive segments given
their start offsets: exactly the shape of the ragged layout. The result equals the naive loop
(tests/kernels/test_maxsim.py checks this) but runs as a few large BLAS calls.

Ported from smve-lab's maxsim.py; candidate gathering for reranking lives in a higher layer.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm


def _check_ragged(flat: np.ndarray, offsets: np.ndarray, name: str) -> None:
    offsets = np.asarray(offsets)
    if offsets.ndim != 1 or len(offsets) == 0 or offsets[0] != 0:
        raise ValueError(f"{name} must be a 1-D array starting at 0")
    if offsets[-1] != len(flat):
        raise ValueError(f"{name} ends at {offsets[-1]}, but the token array has {len(flat)} rows")
    # reduceat silently returns wrong values for empty segments, so rule them out up front.
    if np.any(np.diff(offsets) <= 0):
        raise ValueError(f"every item in {name} needs at least one token")


def _batches(offsets: np.ndarray, max_tokens: int) -> list[tuple[int, int]]:
    """Split items into consecutive [start, end) runs of about `max_tokens` tokens each
    (always at least one item per run, so one long item still gets its own run)."""
    n = len(offsets) - 1
    out, start = [], 0
    while start < n:
        end = int(np.searchsorted(offsets, offsets[start] + max_tokens, side="right")) - 1
        end = min(max(end, start + 1), n)
        out.append((start, end))
        start = end
    return out


def maxsim_scores(
    q_flat: np.ndarray,
    q_offsets: np.ndarray,
    d_flat: np.ndarray,
    d_offsets: np.ndarray,
    chunk_tokens: int = 32_768,
    q_chunk_tokens: int = 16_384,
    show_progress: bool = False,
) -> np.ndarray:
    """Score every query against every document with MaxSim.

    Args:
        q_flat: (Tq, dim) query token vectors, any float dtype.
        q_offsets: (n_queries + 1,) int64 offsets into q_flat.
        d_flat: (Td, dim) document token vectors; may be a float16 memory map.
        d_offsets: (n_docs + 1,) int64 offsets into d_flat.
        chunk_tokens: approximate number of document tokens per matmul.
        q_chunk_tokens: approximate number of query tokens per pass over the documents. The
            similarity block is (q_chunk_tokens x chunk_tokens) float32, ~2 GB at the defaults.
        show_progress: draw a tqdm bar on stderr.

    Returns:
        (n_queries, n_docs) float32 matrix of MaxSim scores.
    """
    _check_ragged(q_flat, q_offsets, "q_offsets")
    _check_ragged(d_flat, d_offsets, "d_offsets")
    n_q, n_d = len(q_offsets) - 1, len(d_offsets) - 1
    scores = np.empty((n_q, n_d), dtype=np.float32)
    q_batches = _batches(q_offsets, q_chunk_tokens)
    d_batches = _batches(d_offsets, chunk_tokens)

    pbar = tqdm(total=n_d * len(q_batches), desc="MaxSim", unit="doc", disable=not show_progress)
    for qs, qe in q_batches:
        Q = np.asarray(q_flat[q_offsets[qs] : q_offsets[qe]], dtype=np.float32)
        q_starts = q_offsets[qs:qe] - q_offsets[qs]
        for ds, de in d_batches:
            t0, t1 = d_offsets[ds], d_offsets[de]
            D = np.asarray(d_flat[t0:t1], dtype=np.float32)  # (Td, dim)
            S = Q @ D.T  # (Tq, Td): every query token vs every doc token
            best_per_doc = np.maximum.reduceat(S, d_offsets[ds:de] - t0, axis=1)  # (Tq, docs)
            scores[qs:qe, ds:de] = np.add.reduceat(best_per_doc, q_starts, axis=0)
            pbar.update(de - ds)
    pbar.close()
    return scores


def subset_ragged(
    flat: np.ndarray, offsets: np.ndarray, indices: np.ndarray | list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Pick items `indices` (in that order) out of a ragged (flat, offsets) array.

    Returns a new (flat, offsets) pair holding only those items; `flat` keeps its dtype.
    """
    indices = np.asarray(indices, dtype=np.int64)
    lens = offsets[indices + 1] - offsets[indices]
    new_offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    if len(indices) == 0:
        return flat[:0], new_offsets
    return np.concatenate([flat[offsets[i] : offsets[i + 1]] for i in indices]), new_offsets
