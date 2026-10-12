"""Hybrid = dense + 0.3 * lexical, in float32, ranked like a naive sort."""

import numpy as np
import scipy.sparse as sp

from laterank.components.base import Encoded
from laterank.components.dense import DenseStage
from laterank.components.hybrid import HYBRID_ALPHA, HybridStage
from laterank.components.lexical import LexicalStage


def test_scores_and_ranking():
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((30, 4)).astype(np.float32)
    docs = sp.random(30, 20, density=0.3, format="csr", dtype=np.float32, random_state=rng)
    dense = DenseStage(vectors)
    lexical = LexicalStage(*LexicalStage.arrays(docs).values(), n_docs=30)
    q = Encoded(np.ones((1, 2), np.float32), rng.standard_normal(4), ([1, 5, 9], [1.0, 0.5, 2.0]))
    hybrid = HybridStage(dense, lexical, HYBRID_ALPHA)

    expected = dense.scores(q) + np.float32(0.3) * lexical.scores(q)
    got = hybrid.scores(q)
    assert got.dtype == np.float32 and np.array_equal(got, expected)
    order = sorted(range(30), key=lambda i: (-expected[i], i))[:10]
    assert hybrid.search(q, 10).rows.tolist() == order
