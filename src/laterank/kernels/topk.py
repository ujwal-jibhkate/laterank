"""Top-k selection with deterministic tie-breaking.

`np.argpartition` finds the k best in O(n) instead of sorting everything, and we then sort only
those k. One subtlety: when several items tie at the k-th score, argpartition may return *any*
of them, and the choice can change between numpy versions or platforms. laterank's rule is
"higher score first, then lower row index first", so the same index, query and library version
always give the same ranking. Scores are never negated or converted, so every numeric dtype
(including unsigned ints, bools and int64 above 2**53) is ranked exactly. Scores must not
contain NaN.
"""

from __future__ import annotations

import numpy as np


def _top_k_1d(scores: np.ndarray, k: int) -> np.ndarray:
    n = len(scores)
    if k == 0:
        return np.zeros(0, dtype=np.int64)
    if k >= n:
        candidates = np.arange(n)
    else:
        part = np.argpartition(scores, n - k)[n - k :]  # the k largest, in no order
        kth = scores[part].min()
        above = np.flatnonzero(scores > kth)  # always in the top k
        tied = np.flatnonzero(scores == kth)[: k - len(above)]  # lowest rows win ties
        candidates = np.concatenate([above, tied])
    # lexsort sorts ascending by its last key first: (score, -row). Reversed, that is score
    # descending, then row ascending.
    return candidates[np.lexsort((-candidates, scores[candidates]))[::-1]]


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores, best first; ties broken by lower index.

    Args:
        scores: (n,) scores for one query, or (n_queries, n) for many.
        k: how many to keep; capped at n.

    Returns:
        int64 indices: (min(k, n),) for 1-D input, (n_queries, min(k, n)) for 2-D input.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    scores = np.asarray(scores)
    if scores.ndim == 1:
        return _top_k_1d(scores, k).astype(np.int64)
    if scores.ndim == 2:
        k = min(k, scores.shape[1])
        out = np.empty((scores.shape[0], k), dtype=np.int64)
        for row in range(scores.shape[0]):
            out[row] = _top_k_1d(scores[row], k)
        return out
    raise ValueError(f"scores must be 1-D or 2-D, got shape {scores.shape}")
