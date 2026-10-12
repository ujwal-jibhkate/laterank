"""The MaxSim reranker must match a per-document MaxSim loop, and break ties by global row."""

import numpy as np
import pytest

from laterank.components.base import Candidates, Encoded
from laterank.components.maxsim import MaxSimReranker


def _cands(rows):
    rows = np.asarray(rows, dtype=np.int64)
    return Candidates(rows, np.zeros(len(rows), dtype=np.float32))


def _naive(q, flat, offsets, row):
    doc = flat[offsets[row] : offsets[row + 1]].astype(np.float64)
    return sum(max(float(qt @ dt) for dt in doc) for qt in q.astype(np.float64))


def test_matches_naive_loop():
    rng = np.random.default_rng(0)
    lengths = rng.integers(1, 8, size=30)
    flat = rng.standard_normal((lengths.sum(), 8)).astype(np.float16)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    q = rng.standard_normal((5, 8)).astype(np.float32)
    rows = [3, 17, 0, 29, 11, 8]
    expected = sorted(rows, key=lambda r: -_naive(q, flat, offsets, r))
    got = MaxSimReranker(flat, offsets).rerank(Encoded(q), _cands(rows))
    assert got.rows.tolist() == expected
    np.testing.assert_allclose(
        got.scores, [_naive(q, flat, offsets, r) for r in expected], rtol=1e-5
    )


def test_ties_go_to_the_lower_row_whatever_the_input_order():
    flat = np.ones((5, 2), dtype=np.float32)  # every document scores the same
    offsets = np.arange(6, dtype=np.int64)
    reranker = MaxSimReranker(flat, offsets)
    q = Encoded(np.ones((1, 2), dtype=np.float32))
    for rows in ([4, 1, 3], [1, 3, 4], [3, 4, 1]):
        assert reranker.rerank(q, _cands(rows)).rows.tolist() == [1, 3, 4]


def test_bad_inputs():
    with pytest.raises(ValueError, match="ends at"):
        MaxSimReranker(np.ones((3, 2), dtype=np.float32), np.array([0, 2], dtype=np.int64))
    reranker = MaxSimReranker(np.ones((2, 2), dtype=np.float32), np.array([0, 1, 2]))
    with pytest.raises(ValueError, match="dim"):
        reranker.rerank(Encoded(np.ones((1, 3), dtype=np.float32)), _cands([0]))
