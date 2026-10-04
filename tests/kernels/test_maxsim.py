"""The vectorized MaxSim must match the obvious per-pair loop, and obey MaxSim's invariants."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from laterank.kernels.maxsim import maxsim_scores, subset_ragged


def _ragged(rng, lengths, dim, dtype=np.float16):
    flat = rng.standard_normal((sum(lengths), dim)).astype(dtype)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return flat, offsets


def _naive(q_flat, q_off, d_flat, d_off):
    out = np.zeros((len(q_off) - 1, len(d_off) - 1), dtype=np.float32)
    for i in range(len(q_off) - 1):
        for j in range(len(d_off) - 1):
            total = 0.0
            for qt in q_flat[q_off[i] : q_off[i + 1]].astype(np.float64):
                total += max(float(qt @ dt) for dt in d_flat[d_off[j] : d_off[j + 1]])
            out[i, j] = total
    return out


def test_matches_naive_loop_across_chunks():
    rng = np.random.default_rng(0)
    q_flat, q_off = _ragged(rng, [3, 1, 7, 4], dim=16)
    d_flat, d_off = _ragged(rng, [5, 2, 9, 1, 6, 3, 8], dim=16)
    expected = _naive(q_flat, q_off, d_flat, d_off)
    # Tiny chunks force many chunks, including single-doc and single-query ones.
    for chunk in (1, 7, 1000):
        for q_chunk in (1, 5, 1000):
            got = maxsim_scores(
                q_flat, q_off, d_flat, d_off, chunk_tokens=chunk, q_chunk_tokens=q_chunk
            )
            assert got.dtype == np.float32
            np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_reads_float16_memory_map(tmp_path):
    rng = np.random.default_rng(1)
    q_flat, q_off = _ragged(rng, [2, 3], dim=8)
    d_flat, d_off = _ragged(rng, [4, 1, 5], dim=8)
    np.save(tmp_path / "flat.npy", d_flat)
    mm = np.load(tmp_path / "flat.npy", mmap_mode="r", allow_pickle=False)
    np.testing.assert_allclose(
        maxsim_scores(q_flat, q_off, mm, d_off), _naive(q_flat, q_off, d_flat, d_off), rtol=1e-5
    )


def test_empty_sets_and_single_items():
    rng = np.random.default_rng(2)
    q_flat, q_off = _ragged(rng, [1], dim=4)
    d_flat, d_off = _ragged(rng, [1], dim=4)
    np.testing.assert_allclose(
        maxsim_scores(q_flat, q_off, d_flat, d_off), _naive(q_flat, q_off, d_flat, d_off)
    )
    no_docs = np.array([0], dtype=np.int64)
    assert maxsim_scores(q_flat, q_off, d_flat[:0], no_docs).shape == (1, 0)
    assert maxsim_scores(q_flat[:0], no_docs, d_flat, d_off).shape == (0, 1)


def test_rejects_items_without_tokens():
    rng = np.random.default_rng(3)
    q_flat, q_off = _ragged(rng, [2], dim=4)
    d_flat, _ = _ragged(rng, [2, 3], dim=4)
    with pytest.raises(ValueError, match="at least one token"):
        maxsim_scores(q_flat, q_off, d_flat, np.array([0, 2, 2, 5]))


def test_rejects_offsets_that_do_not_cover_the_tokens():
    rng = np.random.default_rng(5)
    q_flat, q_off = _ragged(rng, [2], dim=4)
    d_flat, _ = _ragged(rng, [2, 3], dim=4)
    with pytest.raises(ValueError, match="token array has 5 rows"):
        maxsim_scores(q_flat, q_off, d_flat, np.array([0, 3, 8]))  # claims 8 tokens
    with pytest.raises(ValueError, match="token array has 5 rows"):
        maxsim_scores(q_flat, q_off, d_flat, np.array([0, 2, 4]))  # leaves one token out


def test_subset_ragged():
    rng = np.random.default_rng(4)
    flat, off = _ragged(rng, [2, 3, 4], dim=4)
    sub_flat, sub_off = subset_ragged(flat, off, [2, 0])
    assert sub_off.tolist() == [0, 4, 6]
    assert np.array_equal(sub_flat[:4], flat[5:9])
    assert np.array_equal(sub_flat[4:], flat[0:2])
    empty_flat, empty_off = subset_ragged(flat, off, [])
    assert empty_flat.shape == (0, 4) and empty_off.tolist() == [0]


# --- Properties --------------------------------------------------------------------------------

DIM = 8
unit_floats = st.floats(-1, 1, width=32)
vectors = st.integers(1, 12).flatmap(lambda n: arrays(np.float32, (n, DIM), elements=unit_floats))


def _score(q, d):
    return maxsim_scores(q, np.array([0, len(q)]), d, np.array([0, len(d)]))[0, 0]


@settings(max_examples=200, deadline=None)
@given(q=vectors, d=vectors, data=st.data())
def test_doc_token_order_does_not_matter(q, d, data):
    perm = data.draw(st.permutations(range(len(d))))
    assert _score(q, d[list(perm)]) == pytest.approx(_score(q, d), rel=1e-5, abs=1e-5)


@settings(max_examples=200, deadline=None)
@given(q=vectors, d=vectors, extra=arrays(np.float32, (1, DIM), elements=unit_floats))
def test_adding_a_doc_token_never_lowers_the_score(q, d, extra):
    # Each query token's max can only stay or grow when the doc gains a token.
    assert _score(q, np.vstack([d, extra])) >= _score(q, d) - 1e-5
