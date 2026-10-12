"""The lexical stage must score exactly like the sparse product Q @ D.T that smve-lab used."""

import numpy as np
import pytest
import scipy.sparse as sp

from laterank.components.base import Encoded
from laterank.components.lexical import LexicalStage

TOKENS = np.ones((1, 2), dtype=np.float32)


def _stage(doc_matrix):
    a = LexicalStage.arrays(doc_matrix)
    return LexicalStage(*a.values(), n_docs=doc_matrix.shape[0])


def _query(terms, weights):
    return Encoded(TOKENS, lexical=(terms, weights))


def test_matches_the_sparse_product():
    rng = np.random.default_rng(0)
    docs = sp.random(40, 30, density=0.2, format="csr", dtype=np.float32, random_state=rng)
    stage = _stage(docs)
    for terms, weights in (
        ([3, 7, 3, 12], [0.5, 1.0, 0.25, 2.0]),  # term 3 repeats: its weights add up
        ([], []),  # no terms: every document scores 0
    ):
        q = sp.csr_matrix((weights, (np.zeros(len(terms), int), terms)), shape=(1, 30))
        expected = (q @ docs.T).toarray()[0]
        np.testing.assert_allclose(stage.scores(_query(terms, weights)), expected, rtol=1e-5)


def test_ties_in_row_order_and_k_past_the_corpus():
    docs = sp.csr_matrix(np.array([[1.0, 0], [2.0, 0], [1.0, 0], [0, 1.0]], dtype=np.float32))
    got = _stage(docs).search(_query([0], [1.0]), 99)
    assert got.rows.tolist() == [1, 0, 2, 3]


def test_bad_queries():
    stage = _stage(sp.csr_matrix(np.eye(3, dtype=np.float32)))
    with pytest.raises(ValueError, match="lexical weights"):
        stage.scores(Encoded(TOKENS))
    with pytest.raises(ValueError, match=r"\[0, 3\)"):
        stage.scores(_query([3], [1.0]))


def test_encoded_lexical_takes_lists_and_checks_shapes():
    q = _query([1, 2], [0.5, 1])
    assert q.lexical[0].dtype == np.int64 and q.lexical[1].dtype == np.float32
    for bad in (
        ([1, 2], [0.5]),
        ([[1]], [[0.5]]),
        ([1.5], [0.5]),
        ([1], [1 + 2j]),
        ([1], [np.nan]),
    ):
        with pytest.raises(ValueError, match="lexical must be"):
            _query(*bad)
    for not_a_pair in (5, ([1], [0.5], [2])):
        with pytest.raises(ValueError, match="lexical must be"):
            Encoded(TOKENS, lexical=not_a_pair)


def test_bad_postings():
    a = list(LexicalStage.arrays(sp.csr_matrix(np.eye(3, dtype=np.float32))).values())
    with pytest.raises(ValueError, match="outside the 2 documents"):
        LexicalStage(*a, n_docs=2)
    with pytest.raises(ValueError, match="inconsistent"):
        LexicalStage(a[0], a[1][:-1], a[2][:-1], n_docs=3)
    with pytest.raises(ValueError, match="inconsistent"):
        LexicalStage(a[0], a[1].astype(np.float64), a[2], n_docs=3)
