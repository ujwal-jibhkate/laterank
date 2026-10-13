"""SMVE: the batched encoder vs. a dense scatter reference."""

import numpy as np
import pytest
import scipy.sparse as sp
from hypothesis import given, settings
from hypothesis import strategies as st

from laterank.kernels.inverted import score_postings, to_postings
from laterank.kernels.rng import make_anchors
from laterank.kernels.smve import smve_encode, token_mean, topk_blocks


def _dense_reference(x, B, k, is_query, center=None, reps=1):
    """smve-lab's notebook version in plain numpy: scatter each token's top-k (per block of
    w/reps anchors) into a dense (n_tokens, w) matrix, then sum (query) or average over the
    tokens that hit each anchor."""
    x = x.astype(np.float64)
    if center is not None:
        x = x - center
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    P = x @ B.astype(np.float64)
    sparse, mask = np.zeros_like(P), np.zeros_like(P)
    w = P.shape[1] // reps
    for t in range(len(P)):
        for r in range(reps):
            top = r * w + np.argsort(-P[t, r * w : (r + 1) * w])[:k]
            sparse[t, top] = P[t, top]
            mask[t, top] = 1
    pooled = sparse.sum(axis=0)
    return pooled if is_query else pooled / np.maximum(mask.sum(axis=0), 1)


def _ragged(rng, lengths, dim):
    flat = rng.standard_normal((sum(lengths), dim)).astype(np.float16)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return flat, offsets


def test_batched_matches_reference_across_chunks():
    rng = np.random.default_rng(1)
    B = make_anchors(16, 64, seed=0)
    flat, offsets = _ragged(rng, [3, 1, 7, 4, 10, 2], dim=16)
    mu = token_mean(flat)
    for is_query in (True, False):
        for center in (None, mu):
            expected = np.stack(
                [
                    _dense_reference(flat[offsets[i] : offsets[i + 1]], B, 5, is_query, center)
                    for i in range(len(offsets) - 1)
                ]
            )
            # max_elems=64 -> 1 token per chunk: forces one item per chunk.
            for max_elems in (64, 64 * 8, 10**6):
                got = smve_encode(flat, offsets, B, 5, is_query, center=center, max_elems=max_elems)
                assert isinstance(got, sp.csr_matrix) and got.dtype == np.float32
                np.testing.assert_allclose(got.toarray(), expected, rtol=1e-5, atol=1e-6)


def test_reps_equal_concatenated_independent_blocks():
    """R repetitions == R separate SMVE encodings, one per anchor block, side by side."""
    rng = np.random.default_rng(3)
    R, w, k = 3, 32, 4
    B = make_anchors(16, R * w, seed=0)
    flat, offsets = _ragged(rng, [3, 1, 7, 4, 10, 2], dim=16)
    for is_query in (True, False):
        blocks = [
            smve_encode(flat, offsets, B[:, r * w : (r + 1) * w], k, is_query) for r in range(R)
        ]
        expected = sp.hstack(blocks).toarray()
        for max_elems in (R * w, 10**6):
            got = smve_encode(flat, offsets, B, k, is_query, reps=R, max_elems=max_elems)
            np.testing.assert_allclose(got.toarray(), expected, rtol=1e-5, atol=1e-6)
        single = np.stack(
            [
                _dense_reference(flat[offsets[i] : offsets[i + 1]], B, k, is_query, reps=R)
                for i in range(len(offsets) - 1)
            ]
        )
        np.testing.assert_allclose(single, expected, rtol=1e-5, atol=1e-6)


def test_empty_documents_encode_to_zeros_at_any_chunk_size():
    # An empty item alone in a chunk once crashed (int64 bincount); smve-lab returned zeros.
    rng = np.random.default_rng(6)
    B = make_anchors(8, 32, seed=0)
    lengths = [0, 5, 0, 0, 3, 0]
    flat = rng.standard_normal((sum(lengths), 8)).astype(np.float32)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    for is_query in (True, False):
        expected = np.stack(
            [
                _dense_reference(flat[offsets[i] : offsets[i + 1]], B, 3, is_query)
                for i in range(len(lengths))
            ]
        )
        assert not expected[[0, 2, 3, 5]].any()
        for max_elems in (32, 2 * 32, 10**6):
            got = smve_encode(flat, offsets, B, 3, is_query, max_elems=max_elems)
            np.testing.assert_allclose(got.toarray(), expected, rtol=1e-5, atol=1e-6)
        nothing = smve_encode(flat[:0], np.zeros(3, dtype=np.int64), B, 3, is_query)
        assert nothing.shape == (2, 32) and nothing.nnz == 0


def test_inverted_index_scores_equal_sparse_product():
    rng = np.random.default_rng(2)
    B = make_anchors(16, 64, seed=0)
    q = smve_encode(*_ragged(rng, [3, 5], dim=16), B, 4, is_query=True)
    d = smve_encode(*_ragged(rng, [4, 6, 2], dim=16), B, 4, is_query=False)
    postings = to_postings(d)
    expected = (q @ d.T).toarray()
    for i in range(q.shape[0]):
        row = q[i]
        got = score_postings(*postings, d.shape[0], row.indices, row.data)
        np.testing.assert_allclose(got, expected[i], rtol=1e-5, atol=1e-6)


@settings(max_examples=100, deadline=None)
@given(
    lengths=st.lists(st.integers(1, 8), min_size=1, max_size=6),
    k=st.integers(1, 8),
    reps=st.integers(1, 3),
    seed=st.integers(0, 2**32 - 1),
)
def test_each_token_adds_at_most_k_anchors_per_block(lengths, k, reps, seed):
    rng = np.random.default_rng(seed)
    w = 8
    B = make_anchors(6, reps * w, seed=1)
    flat, offsets = _ragged(rng, lengths, dim=6)
    enc = smve_encode(flat, offsets, B, k, is_query=False, reps=reps)
    for i, n in enumerate(lengths):
        cols = enc[i].indices
        assert len(cols) <= n * k * reps
        for r in range(reps):  # and at most n*k inside each block
            assert np.sum((cols >= r * w) & (cols < (r + 1) * w)) <= min(n * k, w)


def test_topk_blocks_rejects_bad_shapes():
    P = np.zeros((2, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="divisible"):
        topk_blocks(P, 2, reps=3)
    with pytest.raises(ValueError, match="k"):
        topk_blocks(P, 6, reps=2)
    values, idx = topk_blocks(P, 5, reps=2)  # k == block width is allowed
    assert values.shape == idx.shape == (2, 10)


def test_topk_blocks_across_threads_matches_argsort():
    rng = np.random.default_rng(3)
    P = rng.standard_normal((2001, 64)).astype(np.float32)  # several slices, a short last one
    for reps, k in ((1, 5), (4, 3)):
        values, idx = topk_blocks(P, k, reps)
        w = 64 // reps
        local = np.argsort(-P.reshape(2001, reps, w), axis=-1)[..., :k]
        want = local + np.arange(reps)[:, None] * w  # local -> global column
        assert np.array_equal(np.sort(idx, axis=1), np.sort(want.reshape(2001, -1), axis=1))
        assert np.array_equal(values, np.take_along_axis(P, idx, axis=1))


def test_token_mean_is_seeded_and_exact_when_sample_covers_everything():
    rng = np.random.default_rng(5)
    flat = rng.standard_normal((50, 4)).astype(np.float16)
    np.testing.assert_allclose(token_mean(flat, sample=1000), flat.astype(np.float64).mean(axis=0))
    assert np.array_equal(token_mean(flat, sample=10, seed=3), token_mean(flat, sample=10, seed=3))
    assert token_mean(flat).dtype == np.float32
