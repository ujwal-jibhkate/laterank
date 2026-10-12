"""The `dense` first stage: one vector per document, scored by inner product against all of them.

Brute force is exact and, at tens of thousands of documents, fast: SciFact is a single
(5183 x 1024) @ (1024,) product. Vectors are stored as float32 in `dense/vectors.npy`.
"""

from __future__ import annotations

import numpy as np

from laterank.components.base import Candidates, Encoded
from laterank.kernels.topk import top_k
from laterank.storage.segment import Segment

VECTORS = "dense/vectors.npy"


class DenseStage:
    def __init__(self, vectors: np.ndarray):
        """vectors: (n_docs, d) float, one row per document (may be a memory map)."""
        if vectors.ndim != 2 or not np.issubdtype(vectors.dtype, np.floating):
            raise ValueError(
                f"dense vectors must be an (n_docs, d) float array, got {vectors.shape} "
                f"{vectors.dtype}"
            )
        self.vectors = vectors

    @staticmethod
    def arrays(vectors: np.ndarray) -> dict[str, np.ndarray]:
        """The segment arrays this stage stores, keyed by their path inside the segment."""
        return {VECTORS: DenseStage(np.asarray(vectors)).vectors.astype(np.float32)}

    @classmethod
    def from_segment(cls, seg: Segment, cfg: dict, n_docs: int) -> DenseStage:
        """Open the stored vectors; ValueError unless they are (n_docs, cfg["dim"])."""
        stage = cls(seg.array(VECTORS))
        if stage.vectors.shape != (n_docs, cfg["dim"]):
            raise ValueError(
                f"dense vectors are {stage.vectors.shape}, expected ({n_docs}, {cfg['dim']})"
            )
        return stage

    def scores(self, q: Encoded) -> np.ndarray:
        """(n_docs,) float32 inner products with the query's dense vector."""
        d = self.vectors.shape[1]
        if q.dense is None:
            raise ValueError("the dense stage needs a query with a dense vector (Encoded.dense)")
        if q.dense.shape != (d,):
            raise ValueError(f"the query's dense vector has shape {q.dense.shape}, expected ({d},)")
        return np.asarray(self.vectors @ q.dense.astype(np.float32), dtype=np.float32)

    def search(self, q: Encoded, k: int) -> Candidates:
        scores = self.scores(q)
        rows = top_k(scores, k)
        return Candidates(rows, scores[rows])
