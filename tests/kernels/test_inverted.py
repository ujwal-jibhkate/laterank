"""Inverted-index scoring must equal the plain sparse matrix product."""

import numpy as np
import scipy.sparse as sp
from hypothesis import given, settings
from hypothesis import strategies as st

from laterank.kernels.inverted import postings_read, ragged_ranges, score_postings, to_postings


def test_ragged_ranges_matches_loop():
    starts, lengths = np.array([10, 50, 7, 3]), np.array([3, 2, 0, 1])
    expected = [i for s, n in zip(starts, lengths, strict=True) for i in range(s, s + n)]
    assert ragged_ranges(starts, lengths).tolist() == expected == [10, 11, 12, 50, 51, 3]
    assert ragged_ranges(np.array([], dtype=int), np.array([], dtype=int)).tolist() == []


def test_postings_layout():
    docs = sp.csr_matrix(np.array([[0, 2, 0], [1, 0, 3], [0, 4, 0]], dtype=np.float32))
    indptr, rows, weights = to_postings(docs)
    assert indptr.tolist() == [0, 1, 3, 4]  # term 0: doc 1; term 1: docs 0, 2; term 2: doc 1
    assert rows.tolist() == [1, 0, 2, 1]
    assert weights.tolist() == [1, 2, 4, 3]
    assert indptr.dtype == np.int64 and rows.dtype == np.int32 and weights.dtype == np.float32
    assert postings_read(indptr, np.array([1, 2])) == 3


def _naive(docs_dense, terms, q_weights):
    """Loop over every document and every query term."""
    out = np.zeros(docs_dense.shape[0])
    for d in range(docs_dense.shape[0]):
        for t, qw in zip(terms, q_weights, strict=True):
            out[d] += qw * docs_dense[d, t]
    return out


@settings(max_examples=100, deadline=None)
@given(
    n_docs=st.integers(1, 30),
    n_terms=st.integers(1, 40),
    density=st.floats(0, 0.5),
    seed=st.integers(0, 2**32 - 1),
    n_query_terms=st.integers(0, 10),
)
def test_matches_naive_loop(n_docs, n_terms, density, seed, n_query_terms):
    rng = np.random.default_rng(seed)
    docs = sp.random(n_docs, n_terms, density=density, format="csr", rng=rng, dtype=np.float32)
    terms = rng.integers(0, n_terms, size=n_query_terms)  # duplicates allowed
    q_weights = rng.random(n_query_terms).astype(np.float32)
    got = score_postings(*to_postings(docs), n_docs, terms, q_weights)
    assert got.dtype == np.float32 and got.shape == (n_docs,)
    np.testing.assert_allclose(got, _naive(docs.toarray(), terms, q_weights), rtol=1e-5, atol=1e-6)


def test_query_with_no_terms_scores_zero():
    docs = sp.csr_matrix(np.eye(3, dtype=np.float32))
    got = score_postings(*to_postings(docs), 3, np.array([], dtype=int), np.array([]))
    assert got.tolist() == [0, 0, 0]
