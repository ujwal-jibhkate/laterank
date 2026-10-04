"""A portable random generator: the same seed gives the same bits on every machine, forever.

Why not torch or np.random? SMVE anchors (and later MUVERA hyperplanes) are part of an index:
documents were encoded with one anchor matrix B, and every future query must be encoded with
the *same* B, or queries stop matching documents. smve-lab drew B with `torch.Generator`, whose
stream torch does not promise to keep across releases, and numpy's `Generator` makes the same
non-promise (NEP 19). So laterank brings its own generator, small enough to read in one sitting.

How it works:

  1. SplitMix64 (Steele, Lea & Flood 2014; Vigna's reference version) turns a counter into
     64 random bits:  x_i = mix(seed + (i + 1) * 0x9E3779B97F4A7C15).  Output i depends only on
     (seed, i), so any slice of the stream can be computed directly, in vectorized numpy.
  2. Two outputs become two uniforms u1 in (0, 1] and u2 in [0, 1) (top 53 bits each).
  3. Box-Muller turns them into two independent standard normals:
        r = sqrt(-2 ln u1),   z0 = r cos(2 pi u2),   z1 = r sin(2 pi u2).

The portability catch is step 3: `np.log`, `np.cos` and `np.sin` are not bit-identical across
platforms (numpy uses its own SIMD code on some x86 CPUs and the system libm elsewhere), so
they could differ in the last bit between Linux, macOS and Windows. IEEE-754 does guarantee
that + - * / and sqrt are correctly rounded everywhere. So `_log` and `_sincos_2pi` below are
short polynomials built from those operations only, and column norms are summed in a fixed
order. The result: anchors depend only on (seed, d, w), on every machine and numpy version.
(They are accurate to ~1e-16, far more than random directions need.)

Separate streams. Different uses of one seed must not read the same random bits, or their
"random" choices would be tied together. Anchors use the seed's stream directly; every other use
(so far: `sample_indices`) first derives its own seed from (seed, purpose tag).
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_2POW_NEG53 = 2.0**-53

_LN2 = 0.6931471805599453  # literal, not math.log(2): no libm call anywhere
_HALF_PI = math.pi / 2  # math.pi is a literal; halving it is exact
_SQRT_HALF = 0.7071067811865476
# Coefficients as correctly rounded divisions of exact integers: identical on every platform.
_ATANH_COEFS = [1.0 / (2 * j + 1) for j in range(11)]  # 2*atanh(s) = 2s * sum(s^2j / (2j+1))
_SIN_COEFS = [(-1) ** j / math.factorial(2 * j + 1) for j in range(9)]  # x^1 .. x^17
_COS_COEFS = [(-1) ** j / math.factorial(2 * j) for j in range(9)]  # x^0 .. x^16

# Purpose tags: the ASCII bytes of a name, read as a little-endian integer. Never change one.
_SAMPLE_TAG = int.from_bytes(b"sample", "little")


def splitmix64(seed: int, start: int, stop: int) -> np.ndarray:
    """Outputs `start .. stop-1` of the SplitMix64 stream for `seed`, as uint64.

    Output i is mix(seed + (i + 1) * gamma), all arithmetic modulo 2**64.
    """
    if not 0 <= seed < 2**64:
        raise ValueError(f"seed must be in [0, 2**64), got {seed}")
    i = np.arange(start + 1, stop + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):  # uint64 arithmetic wraps modulo 2**64, which we want
        z = np.uint64(seed) + i * _GAMMA
        z = (z ^ (z >> np.uint64(30))) * _MIX1
        z = (z ^ (z >> np.uint64(27))) * _MIX2
    return z ^ (z >> np.uint64(31))


def _derive_seed(seed: int, tag: int) -> int:
    """A seed for one purpose, so that purpose's stream is independent of the raw seed's."""
    return int(splitmix64(seed ^ tag, 0, 1)[0])


def _log(x: np.ndarray) -> np.ndarray:
    """Natural log of x > 0 (float64), using only correctly rounded operations.

    x = m * 2**e with m in [sqrt(1/2), sqrt(2)); ln x = e ln2 + 2 atanh((m-1)/(m+1)).
    |s| <= 0.172, so 11 terms of the atanh series reach float64 precision.
    """
    m, e = np.frexp(x)  # exact: just splits the bits; m in [0.5, 1)
    small = m < _SQRT_HALF
    m = np.where(small, m * 2.0, m)
    e = np.where(small, e - 1, e).astype(np.float64)
    s = (m - 1.0) / (m + 1.0)
    s2 = s * s
    acc = np.full_like(s, _ATANH_COEFS[-1])
    for c in reversed(_ATANH_COEFS[:-1]):  # Horner's rule
        acc = acc * s2 + c
    return e * _LN2 + 2.0 * s * acc


def _sincos_2pi(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(sin 2 pi u, cos 2 pi u) for u in [0, 1), using only correctly rounded operations.

    Split the circle into quarter turns: 4u = q + f with integer q and |f| <= 1/2 (both exact),
    so 2 pi u = q * pi/2 + x with |x| <= pi/4, where Taylor series converge fast.
    """
    t = 4.0 * u
    q = np.floor(t + 0.5)
    x = (t - q) * _HALF_PI
    x2 = x * x
    s = np.full_like(x, _SIN_COEFS[-1])
    for c in reversed(_SIN_COEFS[:-1]):
        s = s * x2 + c
    s = s * x
    c = np.full_like(x, _COS_COEFS[-1])
    for k in reversed(_COS_COEFS[:-1]):
        c = c * x2 + k
    # Rotate by q quarter turns: (sin, cos) -> (cos, -sin) per quarter.
    quarter = q.astype(np.int64) % 4
    sin = np.choose(quarter, [s, c, -s, -c])
    cos = np.choose(quarter, [c, -s, -c, s])
    return sin, cos


def standard_normal(seed: int, start: int, stop: int) -> np.ndarray:
    """Normals `start .. stop-1` of the stream for `seed`, as float64.

    Normal pair j (= normals 2j and 2j+1) uses SplitMix64 outputs 2j and 2j+1, so any slice can
    be generated on its own and always equals the same slice of one long call.
    """
    if not 0 <= start <= stop:
        raise ValueError(f"need 0 <= start <= stop, got {start}, {stop}")
    p0, p1 = start // 2, (stop + 1) // 2  # pairs covering [start, stop)
    bits = splitmix64(seed, 2 * p0, 2 * p1).reshape(-1, 2)
    u1 = ((bits[:, 0] >> np.uint64(11)) + np.uint64(1)).astype(np.float64) * _2POW_NEG53  # (0, 1]
    u2 = (bits[:, 1] >> np.uint64(11)).astype(np.float64) * _2POW_NEG53  # [0, 1)
    r = np.sqrt(-2.0 * _log(u1))
    sin, cos = _sincos_2pi(u2)
    z = np.empty(2 * (p1 - p0), dtype=np.float64)
    z[0::2] = r * cos
    z[1::2] = r * sin
    return z[start - 2 * p0 : stop - 2 * p0]


def make_anchors(d: int, w: int, seed: int = 0, chunk_columns: int = 4096) -> np.ndarray:
    """(d, w) float32 matrix whose columns are random unit vectors ("anchors").

    Gaussian vectors point in uniformly random directions, so normalizing them spreads anchors
    evenly over the sphere. Column j is built from normals j*d .. (j+1)*d - 1 of the stream, so
    the first w columns of a wider matrix equal the narrower one (handy for SMVE repetitions).
    Columns are generated `chunk_columns` at a time to bound memory.
    """
    if d <= 0 or w <= 0:
        raise ValueError(f"d and w must be positive, got d={d}, w={w}")
    B = np.empty((d, w), dtype=np.float32)
    for c0 in range(0, w, chunk_columns):
        c1 = min(c0 + chunk_columns, w)
        Z = standard_normal(seed, c0 * d, c1 * d).reshape(c1 - c0, d).T  # (d, cols)
        sq = np.zeros(c1 - c0, dtype=np.float64)
        for row in Z:  # fixed summation order: row 0, 1, 2, ... on every platform
            sq += row * row
        B[:, c0:c1] = Z / np.sqrt(sq)
    return B


def anchor_sha256(B: np.ndarray) -> str:
    """Hex SHA-256 of an anchor matrix: its float32 little-endian bytes in C (row-major) order.

    Stored in the manifest when an index is built; recomputed when it is opened, so a changed
    generator can never silently encode queries against the wrong anchors.
    """
    data = np.ascontiguousarray(B, dtype="<f4")
    return hashlib.sha256(memoryview(data).cast("B")).hexdigest()


def sample_indices(n: int, size: int, seed: int = 0) -> np.ndarray:
    """`size` distinct indices from range(n), sorted ascending, chosen uniformly at random.

    Each index gets a random 64-bit key; the `size` smallest keys win (a stable sort makes the
    choice deterministic even in the astronomically unlikely case of equal keys). The keys come
    from a stream derived from (seed, "sample"), not from the anchors' stream for `seed`.
    """
    if not 0 <= size <= n:
        raise ValueError(f"need 0 <= size <= n, got size={size}, n={n}")
    if not 0 <= seed < 2**64:
        raise ValueError(f"seed must be in [0, 2**64), got {seed}")
    keys = splitmix64(_derive_seed(seed, _SAMPLE_TAG), 0, n)
    return np.sort(np.argsort(keys, kind="stable")[:size])
