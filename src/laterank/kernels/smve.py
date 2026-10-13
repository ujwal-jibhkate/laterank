"""SMVE: compress a bag of token vectors into ONE sparse vector.

SMVE is TopK's method. MaxSim compares every query token with every document token, so its cost
grows with (query tokens x corpus tokens) and it needs every document token vector. SMVE instead
turns each text into a single sparse vector once, and scores with one sparse dot product:

  1. Projection:     P = X @ B          X: (n_tokens, d) token vectors
                                        B: (d, w) random unit-length "anchors" (kernels/rng.py)
     P[t, a] is how well token t points in the direction of anchor a.
  2. Sparsification: keep each token's top-k anchors, zero the rest.
     A token is now described by the k anchors it is closest to.
  3. Pooling:        add up the kept values per anchor.
       - query: SUM over tokens                 -> q[a]
       - doc:   MEAN over the tokens that hit a -> d[a]
  4. Score:          q . d = sum_a q[a] * d[a]   (kernels/inverted.py)

Repetitions (`reps` = R): split the anchors into R blocks of w/R columns and take the top-k
inside EACH block, so every token keeps R*k anchors. This equals R independent SMVE encodings
side by side, giving a matching query/doc token pair R chances to share an anchor. R=1 is plain
SMVE.

Why this approximates MaxSim: a query token contributes to anchor a only if a is among its
nearest anchors, and d[a] is the average strength of the doc tokens near that same anchor, i.e.
doc tokens close to the query token. "Max over doc tokens" becomes "average over nearby doc
tokens".

Centering (`center`): BGE-M3 token vectors all lean towards a shared direction, so anchors near
it land in almost every token's top-k and carry no information. Subtracting the corpus mean token
(see `token_mean`) and re-normalizing removes that shared part.

`smve_encode` encodes a whole ragged array in token chunks and returns a scipy CSR matrix. The
tests check it against a plain dense version (tests/kernels/test_smve.py).

Ported from smve-lab's smve.py with torch replaced by numpy. Anchors come from laterank's
portable generator, so numbers match smve-lab statistically, not bit for bit.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from laterank.kernels.maxsim import _batches
from laterank.kernels.rng import sample_indices

# argpartition runs on one core but releases the GIL, so row slices can run side by side.
# ponytail: SciFact docs take ~5.2 min with threads (torch: 3.3 min); a group-max prefilter
# (top-k of per-group maxima bounds the k-th value) would cut the partition work if needed
_WORKERS = os.cpu_count() or 1


def _center(X: np.ndarray, center: np.ndarray | None) -> np.ndarray:
    """Subtract `center` and re-normalize each row to unit length (no-op when center is None)."""
    if center is None:
        return X
    X = X - np.asarray(center, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), np.float32(1e-12))


def topk_blocks(P: np.ndarray, k: int, reps: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Top-k of each row within each of `reps` equal column blocks.

    Args:
        P: (T, W) projections, W divisible by reps.

    Returns:
        values (T, reps*k) and GLOBAL column indices (T, reps*k) int64, in no particular order
        within a block (pooling doesn't care about order). Rows are split across threads.
        A NaN counts as the largest value, so callers must pass finite projections.
    """
    T, W = P.shape
    if W % reps:
        raise ValueError(f"anchor count {W} is not divisible by reps={reps}")
    w = W // reps
    if not 1 <= k <= w:
        raise ValueError(f"need 1 <= k <= anchors per block ({w}), got k={k}")
    blocks = P.reshape(T, reps, w)
    values = np.empty((T, reps, k), dtype=P.dtype)
    idx = np.empty((T, reps, k), dtype=np.int64)

    def part(rows: slice) -> None:
        i = np.argpartition(blocks[rows], w - k, axis=-1)[..., w - k :]  # top k, local
        idx[rows] = i
        values[rows] = np.take_along_axis(blocks[rows], i, axis=-1)

    step = max(32, -(-T // _WORKERS))  # one slice per core, never tiny ones
    # A pool per call, not a module-level one: a fork()ed child inherits a shared pool without
    # its threads and would hang on first use.
    with ThreadPoolExecutor(_WORKERS) as pool:
        list(pool.map(part, (slice(a, a + step) for a in range(0, T, step))))  # re-raises
    idx += (np.arange(reps) * w)[None, :, None]  # local -> global column
    return values.reshape(T, reps * k), idx.reshape(T, reps * k)


def smve_encode(
    flat: np.ndarray,
    offsets: np.ndarray,
    B: np.ndarray,
    k: int,
    is_query: bool,
    reps: int = 1,
    center: np.ndarray | None = None,
    max_elems: int = 2**26,
    show_progress: bool = False,
) -> sp.csr_matrix:
    """Encode every item of a ragged (flat, offsets) token array with SMVE.

    Items are processed in chunks of whole items so the (tokens x w) projection block stays
    under `max_elems` floats (2**26 = 256 MB). Within a chunk, each token's top-k
    (item, anchor, value) triplets are merged with `np.unique`, so no dense (items x w) matrix
    is ever built.

    Args:
        flat: (T, d) token vectors (may be a float16 memory map); offsets: (n_items + 1,) int64.
        B: (d, w) float32 anchors; with reps > 1, w = reps * (anchors per block).
        k: anchors kept per token per block.
        is_query: sum-pool (queries) instead of mean-pool (documents).
        center: optional (d,) mean token to subtract before projecting.

    Returns:
        (n_items, w) float32 CSR matrix; items with no tokens get an empty row.
    """
    offsets = np.asarray(offsets, dtype=np.int64)
    n_items, w = len(offsets) - 1, B.shape[1]
    B = np.asarray(B, dtype=np.float32)
    lens = np.diff(offsets)
    tokens_per_chunk = max(1, max_elems // w)

    rows, cols, vals = [], [], []
    desc = "SMVE " + ("queries" if is_query else "docs")
    pbar = tqdm(total=n_items, desc=desc, unit="item", disable=not show_progress)
    for start, end in _batches(offsets, tokens_per_chunk):
        X = _center(np.asarray(flat[offsets[start] : offsets[end]], dtype=np.float32), center)
        values, indices = topk_blocks(X @ B, k, reps)  # (T, reps*k)

        # Which item (0-based within this chunk) each token belongs to.
        item = np.repeat(np.arange(end - start, dtype=np.int64), lens[start:end])
        # One integer key per (item, anchor) pair; equal keys get pooled together.
        keys = (item[:, None] * w + indices).ravel()
        uniq, inverse = np.unique(keys, return_inverse=True)
        # astype: a chunk holding only empty items gives an empty, int64 bincount.
        pooled = np.bincount(inverse, weights=values.ravel(), minlength=len(uniq)).astype(
            np.float64
        )
        if not is_query:
            pooled /= np.bincount(inverse, minlength=len(uniq))

        rows.append(uniq // w + start)
        cols.append(uniq % w)
        vals.append(pooled.astype(np.float32))
        pbar.update(end - start)
    pbar.close()

    if not vals:  # no items at all
        return sp.csr_matrix((n_items, w), dtype=np.float32)
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_items, w),
        dtype=np.float32,
    )


def token_mean(flat: np.ndarray, sample: int = 200_000, seed: int = 0) -> np.ndarray:
    """(d,) float32 mean token vector, estimated from a random sample of `sample` tokens.

    Compute it from documents only and reuse it for queries. The sample is drawn with the
    portable generator (not np.random) so a rebuild with the same seed picks the same tokens;
    the mean is accumulated in float64.
    """
    idx = sample_indices(len(flat), min(sample, len(flat)), seed)
    return np.asarray(flat[idx], dtype=np.float64).mean(axis=0).astype(np.float32)
