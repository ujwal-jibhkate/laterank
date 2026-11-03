"""The muvera stage must score exactly like the dot product of its encoded query and documents."""

import numpy as np
import pytest

from laterank.components.base import Encoded
from laterank.components.muvera import MuveraStage
from laterank.kernels.muvera import fde_encode, make_planes

D = 8
CFG = {"reps": 2, "k_sim": 3, "d_proj": 4, "seed": 5}


def _corpus(seed=0, n=30):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(0, 6, size=n)  # some documents have no tokens
    flat = rng.standard_normal((lengths.sum(), D)).astype(np.float16)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return flat, offsets, rng.standard_normal((4, D)).astype(np.float32)


def _stage(flat, offsets, cfg=CFG):
    a = MuveraStage.arrays(flat, offsets, cfg)
    return MuveraStage(
        a["muvera/fdes.npy"], a["muvera/hyperplanes.npy"], a["muvera/projection.npy"], cfg["k_sim"]
    )


def test_scores_equal_the_encoded_dot_product():
    flat, offsets, q = _corpus()
    stage = _stage(flat, offsets)
    G, P = make_planes(D, 2, 3, 4, seed=5)
    docs = fde_encode(flat, offsets, G, P, 3, False)
    query = fde_encode(q, np.array([0, len(q)]), G, P, 3, True)[0]
    np.testing.assert_allclose(stage.scores(Encoded(q)), docs @ query, rtol=1e-5, atol=1e-6)
    assert (stage.reps, stage.d_proj, stage.fdes.shape) == (2, 4, (30, 2 * 8 * 4))


def test_ties_in_row_order_and_query_dim_checked():
    flat = np.ones((4, D), dtype=np.float32)
    offsets = np.array([0, 1, 2, 3, 4])  # four identical documents: all scores tie
    stage = _stage(flat, offsets)
    assert stage.search(Encoded(np.ones((2, D))), 99).rows.tolist() == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="dim"):
        stage.scores(Encoded(np.ones((2, D + 1))))


def test_rejects_bad_arrays():
    flat, offsets, _ = _corpus()
    a = MuveraStage.arrays(flat, offsets, CFG)
    F, G, P = (a[f"muvera/{x}.npy"] for x in ("fdes", "hyperplanes", "projection"))
    inf = G.copy()
    inf[0, 0] = np.inf
    for args in (
        (F[:, :-1], G, P, 3),  # fdes width doesn't match reps * 2**k_sim * d_proj
        (F, G[:, :-1], P, 3),  # hyperplane width isn't a multiple of k_sim
        (F, G, P[:, :-1], 3),  # projection width isn't a multiple of reps
        (F, G[:-1], P, 3),  # hyperplanes and projection disagree on d
        (F.astype(np.float64), G, P, 3),
        (F, G, P, 9),
        (F, inf, P, 3),
    ):
        with pytest.raises(ValueError):
            MuveraStage(*args)


def test_arrays_refuses_encodings_that_overflow():
    flat = np.full((4, D), 3e38, dtype=np.float32)  # sums of projected tokens leave float32
    with pytest.raises(ValueError, match="overflowed"):
        MuveraStage.arrays(flat, np.array([0, 2, 4]), CFG)
