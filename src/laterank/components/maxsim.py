"""The `maxsim` reranker: exact MaxSim, computed only for the candidates it is handed.

It reads the candidates' token vectors out of the index's token store (memory-mapped, so only
those documents are paged in) and scores them with the MaxSim kernel.

Ties go to the lower global row, whatever order the candidates arrived in: the rows are sorted
first, and `top_k` breaks ties by lower position. (smve-lab kept the first stage's order.)
"""

from __future__ import annotations

import numpy as np

from laterank.components.base import Candidates, Encoded
from laterank.kernels.maxsim import _check_ragged, maxsim_scores, subset_ragged
from laterank.kernels.topk import top_k
from laterank.storage.arrays import load_ragged
from laterank.storage.segment import Segment

TOKENS = "tokens"  # tokens/flat.npy + tokens/offsets.npy inside a segment


class MaxSimReranker:
    def __init__(self, flat: np.ndarray, offsets: np.ndarray):
        """The index's token store: flat (T, dim) float, offsets (n_docs + 1,) int64."""
        _check_ragged(flat, offsets, "token offsets")
        self.flat, self.offsets = flat, offsets

    @classmethod
    def from_segment(cls, seg: Segment) -> MaxSimReranker:
        return cls(*load_ragged(seg.folder / TOKENS))

    def rerank(self, q: Encoded, cands: Candidates) -> Candidates:
        dim = self.flat.shape[1]
        if q.tokens.shape[1] != dim:
            raise ValueError(f"query tokens have dim {q.tokens.shape[1]}, the index has {dim}")
        rows = np.sort(cands.rows)
        d_flat, d_offsets = subset_ragged(self.flat, self.offsets, rows)
        q_offsets = np.array([0, len(q.tokens)], dtype=np.int64)
        scores = maxsim_scores(q.tokens, q_offsets, d_flat, d_offsets)[0]
        order = top_k(scores, len(rows))
        return Candidates(rows[order], scores[order])
