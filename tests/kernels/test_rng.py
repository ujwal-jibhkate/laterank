"""The portable generator: published SplitMix64 values, frozen outputs, and sanity statistics.

The FROZEN values below were computed once (2026-10-03) and must never change: indexes store a
hash of their anchors, so any change to the generator's output would make existing indexes
unreadable. If one of these tests fails, the generator changed. Revert, don't update the value.
"""

import hashlib
import math

import numpy as np
import pytest

from laterank.kernels.rng import (
    anchor_sha256,
    make_anchors,
    sample_indices,
    splitmix64,
    standard_normal,
)

MASK = 2**64 - 1


def _splitmix64_loop(seed, n):
    """Vigna's reference SplitMix64, one value at a time in plain Python integers."""
    out, s = [], seed
    for _ in range(n):
        s = (s + 0x9E3779B97F4A7C15) & MASK
        z = s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        out.append(z ^ (z >> 31))
    return out


def test_splitmix64_published_values():
    # First outputs for seed 0 and seed 1234567 from the reference implementation.
    assert splitmix64(0, 0, 4).tolist() == [
        0xE220A8397B1DCDAF,
        0x6E789E6AA1B965F4,
        0x06C45D188009454F,
        0xF88BB8A8724C81EC,
    ]
    assert splitmix64(1234567, 0, 1).tolist() == [0x599ED017FB08FC85]


@pytest.mark.parametrize("seed", [0, 1, 42, 2**63 + 5, MASK])
def test_splitmix64_matches_reference_loop(seed):
    assert splitmix64(seed, 0, 50).tolist() == _splitmix64_loop(seed, 50)
    assert splitmix64(seed, 17, 23).tolist() == _splitmix64_loop(seed, 23)[17:]


def test_seed_out_of_range():
    with pytest.raises(ValueError):
        splitmix64(-1, 0, 1)
    with pytest.raises(ValueError):
        splitmix64(2**64, 0, 1)


def test_normals_match_box_muller_with_libm():
    """Our polynomial log/sin/cos agree with the C library to ~1e-15 (but, unlike it, give the
    same bits on every platform)."""
    bits = _splitmix64_loop(7, 2000)
    expected = []
    for b0, b1 in zip(bits[0::2], bits[1::2], strict=True):
        u1 = ((b0 >> 11) + 1) * 2.0**-53
        u2 = (b1 >> 11) * 2.0**-53
        r = math.sqrt(-2 * math.log(u1))
        expected += [r * math.cos(2 * math.pi * u2), r * math.sin(2 * math.pi * u2)]
    np.testing.assert_allclose(standard_normal(7, 0, 2000), expected, rtol=1e-13, atol=1e-14)


def test_normals_frozen():
    # FROZEN: exact float64 bits of the first six normals for seed 0.
    expected = [
        "-0x1.cf9fb99cfab8fp-2",
        "0x1.a9813db388d75p-3",
        "0x1.53470d1ebc1f2p+1",
        "-0x1.f63166b1324a2p-2",
        "-0x1.fa2a51dfe785fp-1",
        "0x1.df42093b5207bp+0",
    ]
    assert [float(v).hex() for v in standard_normal(0, 0, 6)] == expected


def test_million_normals_frozen():
    # FROZEN: SHA-256 of the exact float64 bytes of the first 10**6 normals for seed 0. Unlike
    # the float32 anchor hashes, this catches a last-bit float64 difference on any platform.
    z = np.ascontiguousarray(standard_normal(0, 0, 10**6), dtype="<f8")
    assert hashlib.sha256(memoryview(z).cast("B")).hexdigest() == (
        "7264a9eb97f10755373ef3209808b85c425e0821fdda87c6d2acf902fc4cbee7"
    )


def test_anchor_hash_frozen():
    # FROZEN: SHA-256 of two anchor matrices; the second spans several generation chunks.
    assert anchor_sha256(make_anchors(64, 1000, seed=0)) == (
        "1ae5b3731b5feb6c4221b05257e87e4f5502758b01840cf4ec9eef4ee0134e88"
    )
    assert anchor_sha256(make_anchors(1024, 4096, seed=42)) == (
        "dfc36b00158d1656fafe0d72901b5003b0021aa9c926b7ff5bb52fbf71d48128"
    )


def test_sample_indices_frozen_and_valid():
    # FROZEN: which 5 of 1000 tokens seed 0 picks.
    assert sample_indices(1000, 5, seed=0).tolist() == [36, 307, 395, 498, 766]
    idx = sample_indices(100, 30, seed=9)
    assert len(set(idx.tolist())) == 30 and np.all(np.diff(idx) > 0) and idx.max() < 100
    assert sample_indices(10, 10).tolist() == list(range(10))
    with pytest.raises(ValueError):
        sample_indices(5, 6)


def test_sampling_does_not_reuse_the_anchor_stream():
    # Ranking indices by the raw seed's bits would tie the sample to the anchors.
    raw = np.sort(np.argsort(splitmix64(0, 0, 1000), kind="stable")[:5])
    assert sample_indices(1000, 5, seed=0).tolist() != raw.tolist()


def test_any_slice_equals_the_same_slice_of_one_long_call():
    full = standard_normal(5, 0, 40)
    for start, stop in [(0, 0), (0, 1), (1, 2), (3, 17), (4, 18), (39, 40)]:
        assert np.array_equal(standard_normal(5, start, stop), full[start:stop])


def test_anchors_are_unit_columns_and_depend_only_on_seed_d_w():
    B = make_anchors(16, 40, seed=3)
    assert B.shape == (16, 40) and B.dtype == np.float32 and B.flags.c_contiguous
    np.testing.assert_allclose(np.linalg.norm(B, axis=0), 1, atol=1e-6)
    assert np.array_equal(B, make_anchors(16, 40, seed=3))
    assert not np.array_equal(B, make_anchors(16, 40, seed=4))
    # The chunk size is an implementation detail and must not change the result...
    assert np.array_equal(B, make_anchors(16, 40, seed=3, chunk_columns=7))
    # ...and a wider matrix starts with the narrower one.
    assert np.array_equal(B, make_anchors(16, 100, seed=3)[:, :40])


@pytest.mark.parametrize("d", [1, 3, 15])
def test_anchor_columns_are_normalized_stream_slices(d):
    # Odd d puts chunk boundaries at odd normal indices, splitting Box-Muller pairs.
    w, seed = 9, 11
    B = make_anchors(d, w, seed=seed)
    for chunk in (1, 2, 5, 100):
        assert np.array_equal(B, make_anchors(d, w, seed=seed, chunk_columns=chunk))
    for j in range(w):
        z = standard_normal(seed, j * d, (j + 1) * d)
        np.testing.assert_allclose(B[:, j], z / np.linalg.norm(z), rtol=1e-6, atol=1e-7)


def test_normals_look_standard():
    z = standard_normal(0, 0, 400_000)
    assert abs(z.mean()) < 0.01
    assert abs(z.std() - 1) < 0.01
    assert abs(np.mean(z**4) - 3) < 0.05  # Gaussian kurtosis
    # Anchor directions are spread out: random unit vectors in 256-d are nearly orthogonal.
    B = make_anchors(256, 500, seed=1)
    cos = B.T @ B
    off_diag = cos[~np.eye(500, dtype=bool)]
    assert abs(off_diag.mean()) < 0.01 and off_diag.std() == pytest.approx(1 / 16, rel=0.05)
