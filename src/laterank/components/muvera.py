"""The `muvera` first stage: one dense vector per document, scored by inner product.

MUVERA (Dhulipala et al., 2024) buckets each document's tokens with random hyperplanes and
stacks the per-bucket averages into one fixed-size vector (`kernels/muvera.py` explains why a
dot product of two such vectors approximates MaxSim). A query is encoded the same way at search
time (bucket sums, not averages), then scored against every document like `dense` does.

The hyperplanes and projection are part of the index: queries must use the same ones as the
documents. They are small (about 6 MB at d = 1024), so unlike SMVE's anchors they are stored.

Ported from smve-lab's `muvera.py` (setting r40_k4_p32, uncentered).
"""

from __future__ import annotations

import numpy as np

from laterank.components.base import Candidates, Encoded
from laterank.kernels.muvera import _check_k_sim, fde_encode, make_planes
from laterank.kernels.topk import top_k
from laterank.storage.segment import Segment

MUVERA_DEFAULT = {"reps": 40, "k_sim": 4, "d_proj": 32, "seed": 0}

FDES = "muvera/fdes.npy"
HYPERPLANES = "muvera/hyperplanes.npy"
PROJECTION = "muvera/projection.npy"


class MuveraStage:
    def __init__(
        self, fdes: np.ndarray, hyperplanes: np.ndarray, projection: np.ndarray, k_sim: int
    ):
        """fdes (n_docs, reps * 2**k_sim * d_proj), hyperplanes (d, reps * k_sim) and
        projection (d, reps * d_proj): float32 (fdes may be a memory map). The planes are checked
        for NaN/inf here; fdes is checked once, when `arrays` builds it, not on every open."""
        _check_k_sim(k_sim)
        for name, a in (("fdes", fdes), ("hyperplanes", hyperplanes), ("projection", projection)):
            if a.ndim != 2 or a.dtype != np.float32:
                raise ValueError(f"{name} must be a 2-D float32 array, got {a.shape} {a.dtype}")
        d, width = hyperplanes.shape
        if width < k_sim or width % k_sim or projection.shape[0] != d:
            raise ValueError(
                f"hyperplanes must be (d, reps * {k_sim}) and projection (d, reps * d_proj); "
                f"got {hyperplanes.shape} and {projection.shape}"
            )
        reps = width // k_sim
        if projection.shape[1] == 0 or projection.shape[1] % reps:
            raise ValueError(f"projection {projection.shape} isn't (d, {reps} * d_proj)")
        d_proj = projection.shape[1] // reps
        expected = reps * 2**k_sim * d_proj
        if fdes.shape[1] != expected:
            raise ValueError(f"fdes has {fdes.shape[1]} columns, expected {expected}")
        if not (np.isfinite(hyperplanes).all() and np.isfinite(projection).all()):
            raise ValueError("MUVERA planes must be finite (no NaN or inf)")
        self.fdes, self.k_sim, self.reps, self.d_proj = fdes, k_sim, reps, d_proj
        self.hyperplanes, self.projection = hyperplanes, projection

    @staticmethod
    def arrays(flat: np.ndarray, offsets: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
        """Encode every document -> the segment arrays this stage stores, keyed by path."""
        G, P = make_planes(flat.shape[1], cfg["reps"], cfg["k_sim"], cfg["d_proj"], cfg["seed"])
        fdes = fde_encode(flat, offsets, G, P, cfg["k_sim"], is_query=False)
        # min/max carry NaN and show +-inf without a copy of fdes; huge tokens can overflow
        if not (np.isfinite(fdes.min(initial=0)) and np.isfinite(fdes.max(initial=0))):
            raise ValueError("muvera: the encodings overflowed (token values too large)")
        return {FDES: fdes, HYPERPLANES: G, PROJECTION: P}

    @classmethod
    def from_segment(cls, seg: Segment, cfg: dict, n_docs: int) -> MuveraStage:
        """Open the stored arrays; ValueError unless they agree with cfg and n_docs."""
        if type(cfg["seed"]) is not int or cfg["seed"] < 0:  # unused on open, but recorded
            raise ValueError(f"muvera seed must be a non-negative int: {cfg}")
        stage = cls(seg.array(FDES), seg.array(HYPERPLANES), seg.array(PROJECTION), cfg["k_sim"])
        if (stage.reps, stage.d_proj) != (cfg["reps"], cfg["d_proj"]):
            raise ValueError(
                f"muvera arrays have reps={stage.reps}, d_proj={stage.d_proj}; "
                f"the manifest says reps={cfg['reps']}, d_proj={cfg['d_proj']}"
            )
        if len(stage.fdes) != n_docs:
            raise ValueError(f"muvera fdes has {len(stage.fdes)} rows, expected {n_docs}")
        return stage

    def scores(self, q: Encoded) -> np.ndarray:
        """(n_docs,) float32: FDE(query tokens, summed) . FDE(document tokens, averaged)."""
        d = self.hyperplanes.shape[0]
        if q.tokens.shape[1] != d:
            raise ValueError(f"query tokens have dim {q.tokens.shape[1]}, the index has {d}")
        offsets = np.array([0, len(q.tokens)], dtype=np.int64)
        v = fde_encode(q.tokens, offsets, self.hyperplanes, self.projection, self.k_sim, True)[0]
        return np.asarray(self.fdes @ v, dtype=np.float32)

    def search(self, q: Encoded, k: int) -> Candidates:
        scores = self.scores(q)
        rows = top_k(scores, k)
        return Candidates(rows, scores[rows])
