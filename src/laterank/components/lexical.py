"""The `lexical` first stage: BGE-M3's learned sparse token weights, over an inverted index.

BGE-M3 gives each text a sparse vector over its ~250k-token vocabulary: a learned weight for
each token the text contains. A document's score is the sum, over tokens it shares with the
query, of query weight x document weight (FlagEmbedding's `compute_lexical_matching_score`).
The weights are stored as postings (`kernels/inverted.py`), so only documents sharing a token
with the query are touched.

Ported from smve-lab's evaluate_bgem3_single_vector.py, which computed the same scores as a
scipy sparse product.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from laterank.components.base import Candidates, Encoded
from laterank.kernels.inverted import score_postings, to_postings
from laterank.kernels.topk import top_k
from laterank.storage.segment import Segment

INDPTR, ROWS, WEIGHTS = "lexical/indptr.npy", "lexical/rows.npy", "lexical/weights.npy"


class LexicalStage:
    def __init__(self, indptr: np.ndarray, rows: np.ndarray, weights: np.ndarray, n_docs: int):
        """The arrays of `kernels.inverted.to_postings`; the vocabulary size is len(indptr) - 1."""
        if (
            indptr.ndim != 1
            or rows.ndim != 1
            or indptr.dtype.kind not in "iu"
            or rows.dtype.kind not in "iu"
            or not np.issubdtype(weights.dtype, np.floating)
            or len(indptr) == 0
            or indptr[0] != 0
            or indptr[-1] != len(rows)
            or rows.shape != weights.shape
            or np.any(np.diff(indptr) < 0)
        ):
            raise ValueError(
                f"inconsistent postings: indptr {indptr.shape} {indptr.dtype}, rows {rows.shape} "
                f"{rows.dtype}, weights {weights.shape} {weights.dtype}"
            )
        if len(rows) and (rows.min() < 0 or rows.max() >= n_docs):
            raise ValueError(f"postings point at rows outside the {n_docs} documents")
        self.indptr, self.rows, self.weights, self.n_docs = indptr, rows, weights, n_docs
        self.vocab = len(indptr) - 1

    @staticmethod
    def arrays(doc_matrix: sp.spmatrix | sp.sparray) -> dict[str, np.ndarray]:
        """(n_docs, vocab) sparse weights -> the segment arrays this stage stores."""
        return dict(zip((INDPTR, ROWS, WEIGHTS), to_postings(doc_matrix), strict=True))

    @classmethod
    def from_segment(cls, seg: Segment, cfg: dict, n_docs: int) -> LexicalStage:
        """Open the stored postings; ValueError unless the vocabulary is cfg["vocab"]."""
        stage = cls(seg.array(INDPTR), seg.array(ROWS), seg.array(WEIGHTS), n_docs)
        if stage.vocab != cfg["vocab"]:
            raise ValueError(f"lexical postings cover {stage.vocab} terms, expected {cfg['vocab']}")
        return stage

    def scores(self, q: Encoded) -> np.ndarray:
        """(n_docs,) float32: sum over shared terms of query weight x document weight."""
        if q.lexical is None:
            raise ValueError(
                "the lexical stage needs a query with lexical weights (Encoded.lexical)"
            )
        terms, weights = q.lexical
        if len(terms) and (terms.min() < 0 or terms.max() >= self.vocab):
            raise ValueError(
                f"query term ids must lie in [0, {self.vocab}); was the query encoded by the "
                "same model as the index?"
            )
        return score_postings(self.indptr, self.rows, self.weights, self.n_docs, terms, weights)

    def search(self, q: Encoded, k: int) -> Candidates:
        scores = self.scores(q)
        rows = top_k(scores, k)
        return Candidates(rows, scores[rows])
