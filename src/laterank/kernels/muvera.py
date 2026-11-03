"""MUVERA fixed dimensional encodings (Dhulipala et al., 2024, arXiv:2405.19504).

Like SMVE, MUVERA turns a bag of token vectors into ONE vector whose dot product with another
such vector approximates MaxSim. The difference is how tokens are grouped:

  SMVE:   soft. Each token joins its top-k of w random anchors.
  MUVERA: hard. `k_sim` random hyperplanes cut the space into B = 2**k_sim buckets; a token
          lands in exactly one, given by which side of each hyperplane it falls on (SimHash).

Per repetition (`reps` independent ones, concatenated):
  1. bucket(x) = the k_sim sign bits of x @ G, read as an integer.
  2. psi(x)    = x @ P: a random +-1/sqrt(d_proj) projection down to d_proj numbers.
  3. A query's block for bucket b is the SUM of psi over its tokens in b; a document's is the
     MEAN. A document's empty bucket takes the psi of its token whose bucket id is nearest in
     Hamming distance, so a query token landing there still finds a match.
  4. Concatenate the blocks: reps * B * d_proj numbers per item.

Why a dot product works: a query token's nearest document tokens usually share its bucket, and
(query sum) . (document mean) in that bucket is that token's similarity to "its" document tokens.

Ported from smve-lab's `muvera.py` (torch). Differences: planes come from the portable
generator (`make_planes`), and the per-bucket sums use `np.bincount` instead of `index_add_`.
"""

from __future__ import annotations

import math

import numpy as np
from tqdm import tqdm

from laterank.kernels.maxsim import _batches
from laterank.kernels.rng import random_signs, standard_normal

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _check_k_sim(k_sim: int) -> None:
    if type(k_sim) is not int or not 1 <= k_sim <= 8:  # bucket ids are kept in uint8
        raise ValueError(f"k_sim must be an int in 1..8, got {k_sim!r}")


def make_planes(
    d: int, reps: int, k_sim: int, d_proj: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """(G, P): G (d, reps * k_sim) float32 SimHash hyperplanes (standard normals), and
    P (d, reps * d_proj) float32 random +-1 / sqrt(d_proj) projections. Same on every machine."""
    for name, v in (("d", d), ("reps", reps), ("d_proj", d_proj)):
        if type(v) is not int or v < 1:
            raise ValueError(f"{name} must be a positive int, got {v!r}")
    _check_k_sim(k_sim)
    G = standard_normal(seed, 0, d * reps * k_sim).reshape(d, reps * k_sim).astype(np.float32)
    P = random_signs(seed, d * reps * d_proj).reshape(d, reps * d_proj)
    return G, P / np.float32(math.sqrt(d_proj))


def fde_encode(
    flat: np.ndarray,
    offsets: np.ndarray,
    G: np.ndarray,
    P: np.ndarray,
    k_sim: int,
    is_query: bool,
    fill_empty: bool | None = None,
    max_tokens: int = 65_536,
    show_progress: bool = False,
) -> np.ndarray:
    """Encode every item of a ragged (flat, offsets) token array -> (n, reps * 2**k_sim * d_proj).

    flat: (T, d) float token vectors (may be a memory map); offsets: (n + 1,) ints from 0 to T.
    G, P: from `make_planes`; reps = G.shape[1] // k_sim, d_proj = P.shape[1] // reps.
    is_query: sum the projected tokens of a bucket (queries) instead of averaging (documents).
    fill_empty: give an empty bucket its nearest token's projection (ties: the earlier token).
        None means documents fill and queries don't. An item with no tokens is all zeros.
    Items are processed in chunks of about `max_tokens` tokens; chunking never changes the result.
    The output is float32, laid out as (n, reps, 2**k_sim, d_proj) flattened.
    """
    _check_k_sim(k_sim)
    offsets = np.asarray(offsets)
    if (
        offsets.ndim != 1
        or offsets.dtype.kind not in "iu"
        or len(offsets) == 0
        or offsets[0] != 0
        or offsets[-1] != len(flat)
        or np.any(np.diff(offsets.astype(np.int64)) < 0)
    ):
        raise ValueError("offsets must be 1-D ints, non-decreasing, from 0 to the token count")
    if flat.ndim != 2 or not np.issubdtype(flat.dtype, np.floating):
        raise ValueError(f"flat must be a (T, d) float array, got {flat.shape} {flat.dtype}")
    d = flat.shape[1]
    if (
        G.ndim != 2
        or P.ndim != 2
        or G.shape[0] != d
        or P.shape[0] != d
        or G.shape[1] < k_sim
        or G.shape[1] % k_sim
        or P.shape[1] % (G.shape[1] // k_sim)
        or P.shape[1] == 0
    ):
        raise ValueError(
            f"G must be (d={d}, reps * {k_sim}) and P (d, reps * d_proj); got {G.shape}, {P.shape}"
        )
    R = G.shape[1] // k_sim
    dp, B = P.shape[1] // R, 1 << k_sim
    fill_empty = (not is_query) if fill_empty is None else fill_empty

    offsets = offsets.astype(np.int64)
    G, P = np.asarray(G, dtype=np.float32), np.asarray(P, dtype=np.float32)
    n, lens = len(offsets) - 1, np.diff(offsets)
    out = np.zeros((n, R, B, dp), dtype=np.float32)
    powers = (1 << np.arange(k_sim)).astype(np.uint8)  # bit j of the bucket id is worth 2**j
    buckets = np.arange(B, dtype=np.uint8)
    reps_idx = np.arange(R)
    desc = "MUVERA " + ("queries" if is_query else "docs")
    pbar = tqdm(total=n, desc=desc, unit="item", disable=not show_progress)
    for start, end in _batches(offsets, max_tokens):
        m = end - start
        X = np.asarray(flat[offsets[start] : offsets[end]], dtype=np.float32)
        T = len(X)
        codes = ((X @ G > 0).reshape(T, R, k_sim) * powers).sum(-1, dtype=np.uint8)  # (T, R)
        proj = (X @ P).reshape(T, R, dp)

        # One slot per (item, rep, bucket); add each projected token into its slot.
        item = np.repeat(np.arange(m, dtype=np.int64), lens[start:end])
        slot = ((item[:, None] * R + reps_idx) * B + codes).ravel()
        n_slots = m * R * B
        counts = np.bincount(slot, minlength=n_slots)
        flat_proj = proj.reshape(T * R, dp)
        sums = np.empty((n_slots, dp), dtype=np.float64)
        for j in range(dp):
            sums[:, j] = np.bincount(slot, weights=flat_proj[:, j], minlength=n_slots)
        if not is_query:
            sums /= np.maximum(counts, 1)[:, None]
        block = sums.astype(np.float32).reshape(m, R, B, dp)

        if fill_empty:
            empty = (counts == 0).reshape(m, R, B)
            pos = 0
            for i in range(m):
                n_i = int(lens[start + i])
                if n_i and empty[i].any():
                    ham = _POPCOUNT[codes[pos : pos + n_i, :, None] ^ buckets]  # (n_i, R, B)
                    nearest = ham.argmin(axis=0)  # (R, B): the token closest to each bucket
                    r, b = np.nonzero(empty[i])
                    block[i, r, b] = proj[pos + nearest[r, b], r]
                pos += n_i
        out[start:end] = block
        pbar.update(m)
    pbar.close()
    return out.reshape(n, R * B * dp)
