"""The dense stage must rank exactly like scoring every document and sorting by (score, row)."""

import numpy as np
import pytest

from laterank.components.base import Encoded
from laterank.components.dense import VECTORS, DenseStage


def _query(dense):
    return Encoded(tokens=np.ones((1, 4), dtype=np.float32), dense=dense)


def test_matches_naive_loop():
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((50, 8)).astype(np.float32)
    q = rng.standard_normal(8).astype(np.float32)
    scores = [float(np.float32(v @ q)) for v in vectors]
    expected = sorted(range(50), key=lambda i: (-scores[i], i))[:10]
    got = DenseStage(vectors).search(_query(q), 10)
    assert got.rows.tolist() == expected
    np.testing.assert_allclose(got.scores, [scores[i] for i in expected], rtol=1e-6)


def test_ties_in_row_order_and_k_past_the_corpus():
    vectors = np.array([[1.0], [2.0], [1.0], [2.0]], dtype=np.float32)
    got = DenseStage(vectors).search(_query(np.array([1.0], dtype=np.float32)), 99)
    assert got.rows.tolist() == [1, 3, 0, 2]


def test_arrays_stores_float32():
    out = DenseStage.arrays(np.ones((3, 2), dtype=np.float64))
    assert list(out) == [VECTORS] and out[VECTORS].dtype == np.float32


def test_bad_inputs():
    with pytest.raises(ValueError, match="float array"):
        DenseStage(np.ones((3, 2), dtype=np.int64))
    stage = DenseStage(np.ones((3, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="dense vector"):
        stage.search(_query(None), 1)
    with pytest.raises(ValueError, match="expected"):
        stage.search(_query(np.ones(5, dtype=np.float32)), 1)
