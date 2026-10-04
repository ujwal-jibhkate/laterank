"""top_k against a full sort, including heavy ties."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from laterank.kernels.topk import top_k


def _naive(scores, k):
    """Sort every index by (score descending, index ascending) and keep k."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]


def test_sorted_best_first():
    scores = np.array([0.1, 0.9, 0.5, 0.3], dtype=np.float32)
    assert top_k(scores, 3).tolist() == [1, 2, 3]


def test_ties_go_to_the_lower_index():
    scores = np.array([1.0, 2.0, 1.0, 1.0, 2.0, 1.0])
    assert top_k(scores, 3).tolist() == [1, 4, 0]
    assert top_k(scores, 4).tolist() == [1, 4, 0, 2]


def test_k_edge_cases():
    scores = np.array([3.0, 1.0, 2.0])
    assert top_k(scores, 0).tolist() == []
    assert top_k(scores, 10).tolist() == [0, 2, 1]
    assert top_k(np.array([]), 5).tolist() == []
    with pytest.raises(ValueError):
        top_k(scores, -1)


def test_integer_and_bool_scores():
    # Negating unsigned ints wraps around (-5 as uint8 is 251); scores must be compared as values.
    assert top_k(np.array([0, 5, 3], dtype=np.uint8), 1).tolist() == [1]
    assert top_k(np.array([0, 5, 3], dtype=np.uint64), 2).tolist() == [1, 2]
    assert top_k(np.array([-2, 7, 7], dtype=np.int32), 2).tolist() == [1, 2]
    assert top_k(np.array([False, True, True]), 2).tolist() == [1, 2]
    # Above 2**53, neighbouring int64 values collapse to one float64: compare them exactly.
    assert top_k(np.array([2**60, 2**60 + 1], dtype=np.uint64), 1).tolist() == [1]
    assert top_k(np.array([2**62, 2**62 + 1, -(2**63)], dtype=np.int64), 2).tolist() == [1, 0]


def test_two_dimensional_is_row_by_row():
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 4, size=(5, 20)).astype(np.float32)
    got = top_k(scores, 7)
    assert got.shape == (5, 7) and got.dtype == np.int64
    for row in range(5):
        assert got[row].tolist() == _naive(scores[row], 7)


@settings(max_examples=300, deadline=None)
@given(
    # Few distinct values -> lots of ties at the k-th place.
    scores=arrays(np.float32, st.integers(0, 40), elements=st.sampled_from([-1.0, 0.0, 0.5, 2.0])),
    k=st.integers(0, 45),
)
def test_matches_full_sort_with_ties(scores, k):
    assert top_k(scores, k).tolist() == _naive(scores.tolist(), k)


@settings(max_examples=200, deadline=None)
@given(
    scores=arrays(np.float64, st.integers(1, 60), elements=st.floats(-1e6, 1e6)),
    k=st.integers(1, 60),
)
def test_matches_full_sort_on_arbitrary_floats(scores, k):
    assert top_k(scores, k).tolist() == _naive(scores.tolist(), k)
