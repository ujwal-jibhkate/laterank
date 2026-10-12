"""The `hybrid` first stage: dense + alpha * lexical, BGE-M3's default single-pass combination.

It stores nothing of its own. The index builds it from its `dense` and `lexical` stages, and
its manifest entry records only `alpha`. alpha = 0.3 is the BGE-M3 paper's value, and the one
behind smve-lab's golden hybrid numbers (kept fixed there, not tuned on test queries).
"""

from __future__ import annotations

import numpy as np

from laterank.components.base import Candidates, Encoded
from laterank.components.dense import DenseStage
from laterank.components.lexical import LexicalStage
from laterank.kernels.topk import top_k

HYBRID_ALPHA = 0.3


class HybridStage:
    def __init__(self, dense: DenseStage, lexical: LexicalStage, alpha: float):
        self.dense, self.lexical, self.alpha = dense, lexical, np.float32(alpha)

    def scores(self, q: Encoded) -> np.ndarray:
        """(n_docs,) float32: dense.scores(q) + alpha * lexical.scores(q)."""
        return self.dense.scores(q) + self.alpha * self.lexical.scores(q)

    def search(self, q: Encoded, k: int) -> Candidates:
        scores = self.scores(q)
        rows = top_k(scores, k)
        return Candidates(rows, scores[rows])
