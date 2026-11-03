"""MUVERA kernel: `fde_encode` against a plain-Python loop, and what `make_planes` promises."""

import math

import numpy as np
import pytest

from laterank.kernels.muvera import fde_encode, make_planes

K_SIM, REPS, D_PROJ, D = 3, 2, 4, 6


def _reference(flat, offsets, G, P, k_sim, is_query, fill):
    """Items, reps and buckets one at a time; nothing vectorized, nothing chunked."""
    R, B = G.shape[1] // k_sim, 2**k_sim
    dp = P.shape[1] // R
    out = np.zeros((len(offsets) - 1, R, B, dp))
    for i in range(len(offsets) - 1):
        X = flat[offsets[i] : offsets[i + 1]].astype(np.float32)
        for r in range(R):
            bits = X @ G[:, r * k_sim : (r + 1) * k_sim] > 0
            code = [sum(int(b) << j for j, b in enumerate(row)) for row in bits]
            proj = X @ P[:, r * dp : (r + 1) * dp]
            for b in range(B):
                members = [t for t, c in enumerate(code) if c == b]
                if members:
                    total = proj[members].astype(np.float64).sum(axis=0)
                    out[i, r, b] = total if is_query else total / len(members)
                elif fill and len(X):
                    # nearest bucket id in Hamming distance; ties go to the earlier token
                    dist = [bin(c ^ b).count("1") for c in code]
                    out[i, r, b] = proj[int(np.argmin(dist))]
    return out.reshape(len(offsets) - 1, -1)


def _data(lens, seed=0):
    rng = np.random.default_rng(seed)
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    flat = rng.standard_normal((offsets[-1], D)).astype(np.float32)
    G, P = make_planes(D, REPS, K_SIM, D_PROJ, seed=seed)
    return flat, offsets, G, P


LENS = [5, 1, 0, 3, 9, 2]  # short items leave many of the 8 buckets empty; one item has no tokens


@pytest.mark.parametrize(
    ("is_query", "fill"), [(False, None), (True, None), (True, True), (False, False)]
)
def test_fde_matches_the_loop_reference(is_query, fill):
    flat, offsets, G, P = _data(LENS)
    got = fde_encode(flat, offsets, G, P, K_SIM, is_query, fill)
    fills = (not is_query) if fill is None else fill
    want = _reference(flat, offsets, G, P, K_SIM, is_query, fills)
    assert got.shape == (len(LENS), REPS * 2**K_SIM * D_PROJ)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_empty_item_is_all_zeros_even_when_filling():
    flat, offsets, G, P = _data(LENS)
    got = fde_encode(flat, offsets, G, P, K_SIM, is_query=False)
    assert not got[2].any()
    assert got[1].any()


@pytest.mark.parametrize("max_tokens", [1, 7, 10**6])
def test_chunking_never_changes_the_result(max_tokens):
    flat, offsets, G, P = _data(LENS)
    whole = fde_encode(flat, offsets, G, P, K_SIM, False)
    chunked = fde_encode(flat, offsets, G, P, K_SIM, False, max_tokens=max_tokens)
    np.testing.assert_allclose(chunked, whole, rtol=1e-5, atol=1e-6)


def test_no_items_gives_no_rows():
    _, _, G, P = _data(LENS)
    out = fde_encode(np.zeros((0, D), np.float32), np.array([0]), G, P, K_SIM, False)
    assert out.shape == (0, REPS * 2**K_SIM * D_PROJ)


def test_make_planes_shapes_values_and_repeatability():
    G, P = make_planes(D, REPS, K_SIM, D_PROJ, seed=7)
    assert G.shape == (D, REPS * K_SIM) and G.dtype == np.float32
    assert P.shape == (D, REPS * D_PROJ) and P.dtype == np.float32
    assert set(np.abs(P).ravel().tolist()) == {np.float32(1 / math.sqrt(D_PROJ))}
    G2, P2 = make_planes(D, REPS, K_SIM, D_PROJ, seed=7)
    np.testing.assert_array_equal(G, G2)
    np.testing.assert_array_equal(P, P2)
    assert not np.array_equal(G, make_planes(D, REPS, K_SIM, D_PROJ, seed=8)[0])


@pytest.mark.parametrize(
    "bad",
    [
        dict(d=0), dict(reps=0), dict(d_proj=0), dict(k_sim=0), dict(k_sim=9), dict(reps=1.5),
    ],
)  # fmt: skip
def test_make_planes_rejects_bad_settings(bad):
    args = dict(d=D, reps=REPS, k_sim=K_SIM, d_proj=D_PROJ, seed=0) | bad
    with pytest.raises(ValueError):
        make_planes(**args)


def test_fde_encode_rejects_mismatched_inputs():
    flat, offsets, G, P = _data(LENS)
    with pytest.raises(ValueError, match="k_sim"):
        fde_encode(flat, offsets, G, P, 9, False)
    with pytest.raises(ValueError, match="G must be"):
        fde_encode(flat, offsets, G[:-1], P, K_SIM, False)  # wrong d
    with pytest.raises(ValueError, match="G must be"):
        fde_encode(flat, offsets, G[:, :-1], P, K_SIM, False)  # width not a multiple of k_sim
    with pytest.raises(ValueError, match="G must be"):
        fde_encode(flat, offsets, G, P[:, :-1], K_SIM, False)  # width not a multiple of reps
    with pytest.raises(ValueError, match="offsets"):
        fde_encode(flat, offsets[:-1], G, P, K_SIM, False)  # doesn't end at the token count
    with pytest.raises(ValueError, match="offsets"):
        fde_encode(flat, offsets[::-1], G, P, K_SIM, False)
