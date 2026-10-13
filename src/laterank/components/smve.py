"""The `smve` first stage: TopK's SMVE, one sparse vector per document over random anchors.

Each document's token vectors are projected onto w random unit "anchors", each token keeps its
k strongest anchors, and the kept values are averaged per anchor (`kernels/smve.py` explains
why this approximates MaxSim). A query is encoded the same way at search time (summed, not
averaged), and the score is one sparse dot product over postings, as in `lexical`.

The anchors are part of the index: a query must use the same ones as the documents. They are
256 MB at w = 65,536, so they aren't stored; `open` regenerates them from (seed, d, reps * w)
with laterank's portable generator and checks their SHA-256 against the manifest (design §7).

Ported from smve-lab's evaluate_smve.py (setting w65536_k32_center).
"""

from __future__ import annotations

import numpy as np

from laterank.components.base import Candidates, Encoded
from laterank.components.lexical import LexicalStage
from laterank.components.maxsim import TOKENS
from laterank.kernels.inverted import score_postings, to_postings
from laterank.kernels.rng import anchor_sha256, make_anchors
from laterank.kernels.smve import smve_encode, token_mean
from laterank.kernels.topk import top_k
from laterank.storage.segment import Segment

SMVE_BALANCED = {"w": 65536, "k": 32, "reps": 1, "seed": 0, "center": True}

INDPTR, ROWS, WEIGHTS = "smve/indptr.npy", "smve/rows.npy", "smve/weights.npy"
CENTER = "smve/center.npy"


def _anchors(d: int, cfg: dict) -> np.ndarray:
    return make_anchors(d, cfg["reps"] * cfg["w"], cfg["seed"])


class SMVEStage:
    def __init__(
        self,
        indptr: np.ndarray,
        rows: np.ndarray,
        weights: np.ndarray,
        n_docs: int,
        anchors: np.ndarray,
        k: int,
        reps: int,
        center: np.ndarray | None,
    ):
        """Postings of the document vectors (checked as `lexical`'s), the (d, reps * w) float32
        anchors, and the (d,) float32 center (None when the tokens aren't centered)."""
        self._postings = LexicalStage(indptr, rows, weights, n_docs)
        width = self._postings.vocab
        if anchors.ndim != 2 or anchors.dtype != np.float32 or anchors.shape[1] != width:
            raise ValueError(
                f"anchors must be (d, {width}) float32, got {anchors.shape} {anchors.dtype}"
            )
        if type(reps) is not int or type(k) is not int or reps < 1 or width % reps:
            raise ValueError(f"reps must be a positive int dividing {width}, got {reps!r}")
        if not 1 <= k <= width // reps:
            raise ValueError(f"need 1 <= k <= {width // reps} (anchors per block), got {k!r}")
        if center is not None and (
            center.shape != (anchors.shape[0],)
            or not np.issubdtype(center.dtype, np.floating)
            or not np.isfinite(center).all()
        ):
            raise ValueError(
                f"center must be ({anchors.shape[0]},) finite floats, got {center.shape}"
            )
        self.anchors, self.k, self.reps, self.center = anchors, k, reps, center

    @staticmethod
    def arrays(
        flat: np.ndarray, offsets: np.ndarray, cfg: dict
    ) -> tuple[dict[str, np.ndarray], str]:
        """Encode every document -> (the segment arrays this stage stores, anchor_sha256)."""
        B = _anchors(flat.shape[1], cfg)
        center = token_mean(flat) if cfg["center"] else None  # documents only, reused for queries
        docs = smve_encode(
            flat, offsets, B, cfg["k"], is_query=False, reps=cfg["reps"], center=center
        )
        arrays = dict(zip((INDPTR, ROWS, WEIGHTS), to_postings(docs), strict=True))
        if center is not None:
            arrays[CENTER] = center
        return arrays, anchor_sha256(B)

    @classmethod
    def from_segment(cls, seg: Segment, cfg: dict, n_docs: int) -> SMVEStage:
        """Regenerate the anchors; ValueError if they differ from the manifest's hash."""
        ints = [cfg[key] for key in ("w", "k", "reps", "seed")]
        if any(type(x) is not int or x < 0 for x in ints) or type(cfg["center"]) is not bool:
            raise ValueError(f"smve settings must be non-negative ints and a bool center: {cfg}")
        indptr = seg.array(INDPTR)
        if len(indptr) != cfg["reps"] * cfg["w"] + 1:  # before building a huge anchor matrix
            raise ValueError(f"smve postings cover {len(indptr) - 1} anchors, expected reps * w")
        # ponytail: anchors rebuilt on every open (~2.2 s at w = 65,536); build them on the first
        # query, or cache them, if open time matters
        B = _anchors(seg.array(f"{TOKENS}/flat.npy").shape[1], cfg)  # d = the token dim
        if anchor_sha256(B) != cfg["anchor_sha256"]:
            raise ValueError(
                "the regenerated SMVE anchors don't match the ones the index was built with "
                "(different laterank generator?); rebuild the index"
            )
        center = seg.array(CENTER, mmap=False) if cfg["center"] else None
        rows, weights = seg.array(ROWS), seg.array(WEIGHTS)
        return cls(indptr, rows, weights, n_docs, B, cfg["k"], cfg["reps"], center)

    def scores(self, q: Encoded) -> np.ndarray:
        """(n_docs,) float32: SMVE(query tokens, summed) . SMVE(document tokens, averaged)."""
        d = self.anchors.shape[0]
        if q.tokens.shape[1] != d:
            raise ValueError(f"query tokens have dim {q.tokens.shape[1]}, the index has {d}")
        offsets = np.array([0, len(q.tokens)], dtype=np.int64)
        v = smve_encode(q.tokens, offsets, self.anchors, self.k, True, self.reps, self.center)
        p = self._postings
        return score_postings(p.indptr, p.rows, p.weights, p.n_docs, v.indices, v.data)

    def search(self, q: Encoded, k: int) -> Candidates:
        scores = self.scores(q)
        rows = top_k(scores, k)
        return Candidates(rows, scores[rows])
