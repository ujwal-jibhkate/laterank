"""The smve stage must score exactly like the sparse product of its encoded query and documents."""

import numpy as np
import pytest

from laterank.components.base import Encoded
from laterank.components.smve import SMVEStage
from laterank.kernels.rng import make_anchors
from laterank.kernels.smve import smve_encode, token_mean

D = 8


def _corpus(seed=0, n=30):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(0, 6, size=n)  # some documents have no tokens
    flat = rng.standard_normal((lengths.sum(), D)).astype(np.float16)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return flat, offsets, rng.standard_normal((4, D)).astype(np.float32)


def _stage(flat, offsets, cfg):
    a, _ = SMVEStage.arrays(flat, offsets, cfg)
    center = a.get("smve/center.npy")
    B = make_anchors(D, cfg["reps"] * cfg["w"], cfg["seed"])
    p = [a[f"smve/{x}.npy"] for x in ("indptr", "rows", "weights")]
    return SMVEStage(*p, len(offsets) - 1, B, cfg["k"], cfg["reps"], center), B, center


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("reps", [1, 2])
def test_scores_equal_the_encoded_sparse_product(center, reps):
    flat, offsets, q = _corpus()
    cfg = {"w": 32, "k": 4, "reps": reps, "seed": 3, "center": center}
    stage, B, mu = _stage(flat, offsets, cfg)
    if center:
        np.testing.assert_array_equal(mu, token_mean(flat))
    docs = smve_encode(flat, offsets, B, 4, False, reps, mu)
    query = smve_encode(q, np.array([0, len(q)]), B, 4, True, reps, mu)
    expected = (query @ docs.T).toarray()[0]
    np.testing.assert_allclose(stage.scores(Encoded(q)), expected, rtol=1e-5, atol=1e-7)


def test_ties_in_row_order_and_query_dim_checked():
    flat = np.ones((4, D), dtype=np.float32)
    offsets = np.array([0, 1, 2, 3, 4])  # four identical documents: all scores tie
    stage, _, _ = _stage(flat, offsets, {"w": 16, "k": 2, "reps": 1, "seed": 0, "center": False})
    assert stage.search(Encoded(np.ones((2, D))), 99).rows.tolist() == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="dim"):
        stage.scores(Encoded(np.ones((2, D + 1))))


def test_rejects_bad_settings():
    flat, offsets, _ = _corpus()
    cfg = {"w": 16, "k": 2, "reps": 1, "seed": 0, "center": True}
    stage, B, mu = _stage(flat, offsets, cfg)
    p = (stage._postings.indptr, stage._postings.rows, stage._postings.weights, 30)
    for args in (
        (B[:, :8], 2, 1, mu),  # anchors don't cover the postings' width
        (B.astype(np.float64), 2, 1, mu),
        (B, 17, 1, mu),  # k > anchors per block
        (B, 2, 3, mu),  # reps doesn't divide the width
        (B, True, 1, mu),
        (B, 2, 1, mu[:4]),  # center of the wrong length
        (B, 2, 1, np.full(D, np.nan, dtype=np.float32)),
    ):
        with pytest.raises(ValueError):
            SMVEStage(*p, *args)
