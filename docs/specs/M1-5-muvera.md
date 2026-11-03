---
id: M1-5
title: MUVERA first stage
status: done
approved: 2026-11-03
---

# M1-5: MUVERA first stage

## Goal

A `muvera` first stage: Google's MUVERA ("Fixed Dimensional Encodings") turns a document's token
vectors into ONE dense vector, and a query the same way, so a plain dot product approximates
MaxSim. Tokens are put in hard buckets by random hyperplanes (SimHash), unlike SMVE's soft
anchors. This checkpoint ports smve-lab's `muvera.py` to numpy, stores the encodings in an index,
and checks them on SciFact. Setting (owner's pick, 2026-11-03): smve-lab's `r40_k4_p32`, which
had nDCG@10 0.5530, Recall@100 0.858 and `> maxsim@100` 0.6888 (seed 0, torch RNG).

## Out of scope

- Skipped: choosing reps, k_sim, d_proj or seed (`muvera(preset=...)`). Add with presets in M1-6
  and `tune` in M2. The manifest records every setting, so a later index stays readable.
- Skipped: `center` (smve-lab's option; it lowered nDCG@10 for MUVERA on SciFact: 0.5530 →
  0.5283 at r40_k4_p32, 0.5559 → 0.5440 at r40_k6_p32, 0.4822 → 0.4550 at r20_k5_p16). Add when a dataset shows it helping.
- Skipped: turning `fill_empty` off. The paper always fills empty buckets; add if a use appears.
- Skipped: compressing the encodings (float16, product quantization). The index is 425 MB on
  SciFact, 100× SMVE's. Add when a user's corpus doesn't fit on disk (`# ponytail` below).
- NFCorpus and ArguAna golden tests: a later M1 checkpoint. Encoders, presets: M1-6.

## Interfaces

```python
# kernels/rng.py (one addition)
def random_signs(seed: int, n: int) -> np.ndarray:
    """(n,) float32 of +1.0 / -1.0: bit 63 of SplitMix64 outputs 0..n-1 of the stream derived
    from (seed, tag "signs"), 1 -> +1. Never reads the planes' stream."""


# kernels/muvera.py (new; numpy only, no torch)
def make_planes(
    d: int, reps: int, k_sim: int, d_proj: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """(G, P). G (d, reps*k_sim) float32: standard normals from the portable RNG (SimHash
    hyperplanes). P (d, reps*d_proj) float32: random_signs / sqrt(d_proj). ValueError unless
    d, reps, d_proj >= 1 and 1 <= k_sim <= 8 (bucket ids are uint8)."""


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
    """Ragged tokens (flat (T, d) float, offsets (n+1,) int64) -> (n, reps*2**k_sim*d_proj) float32.
    Per item, rep and bucket (bucket = sign bits of x @ G, read as an integer):
      query:    block = SUM of the projected tokens (x @ P) in the bucket; empty bucket = 0
      document: block = MEAN of them; an empty bucket takes the projection of the item's token
                whose bucket id is nearest in Hamming distance (ties: the earlier token)
    fill_empty=None means "documents fill, queries don't"; an item with no tokens is all zeros.
    Output layout (n, reps, 2**k_sim, d_proj) flattened. Chunks hold <= max_tokens tokens, and
    never change the result."""


# components/muvera.py (new)
MUVERA_DEFAULT = {"reps": 40, "k_sim": 4, "d_proj": 32, "seed": 0}


class MuveraStage:
    def __init__(
        self, fdes: np.ndarray, hyperplanes: np.ndarray, projection: np.ndarray, k_sim: int
    ):
        """fdes (n_docs, reps*2**k_sim*d_proj) float32 (width checked against reps, k_sim, d_proj);
        hyperplanes (d, reps*k_sim); projection (d, reps*d_proj). ValueError if the widths
        don't agree (reps*2**k_sim*d_proj) or the planes aren't finite float32.
        `arrays` raises ValueError if the encodings overflow (so `fdes` is checked once, at
        build, not on every open)."""

    @staticmethod
    def arrays(flat: np.ndarray, offsets: np.ndarray, cfg: dict) -> dict[str, np.ndarray]: ...
    @classmethod
    def from_segment(cls, seg: Segment, cfg: dict, n_docs: int) -> MuveraStage: ...
    def scores(self, q: Encoded) -> np.ndarray:
        """(n_docs,) float32: FDE(q.tokens, query) . FDE(doc)."""

    def search(self, q: Encoded, k: int) -> Candidates: ...


# orchestration/index.py: Index.create(..., muvera: bool = False)
#   True -> stage "muvera" with MUVERA_DEFAULT, from the token vectors (finite check shared
#   with smve=True)
```

```text
manifest.json stage:
  "muvera": {"type": "muvera", "reps": 40, "k_sim": 4, "d_proj": 32, "seed": 0}
seg-000001/muvera/fdes.npy         (n_docs, 20480) float32   one encoding per document
seg-000001/muvera/hyperplanes.npy  (d, 160) float32          G
seg-000001/muvera/projection.npy   (d, 1280) float32         P
```

## Decisions

| decision | why | differs from design doc? |
|---|---|---|
| Setting r40_k4_p32, uncentered, `fill_empty` on | owner's pick; best speed/size trade in smve-lab's 54 runs (425 MB, ~10 ms) | no |
| `G` and `P` are stored, not regenerated and hashed like SMVE's anchors | they are 6 MB at d = 1024 (anchors: 256 MB), so storing is simpler and `open` stays instant | yes, small (design §7) |
| `P` signs come from a new `random_signs` stream; `G` from `standard_normal` | portable bits (`kernels/rng.py` rationale); torch's stream is not promised stable | no |
| Documents stored as a dense matrix; width checked by the constructor, finiteness by `arrays` at build | same layout as `dense`, which also validates at create, not open. Scanning 425 MB on every `open` would defeat the memory map | no |
| Band, not a point; laterank seeds 0, 1, 2 are measured | the RNG differs from torch's, and smve-lab has one seed per setting | no |
| Band for `muvera` alone: [0.500, 0.580]; `> maxsim@100`: 0.6888 ± 0.01 | measured seeds 0/1/2 gave 0.5335 / 0.5378 / 0.5118 alone (spread 0.026) and 0.6895 / 0.6870 / 0.6904 reranked (spread 0.003). The first guess [0.533, 0.573] missed seed 2; the owner chose to widen on 2026-11-03 | no |
| Naive-loop reference test for `fde_encode` | kernels keep one (CLAUDE.md) | no |

`# ponytail:` the build holds the whole `fdes` matrix in RAM (n_docs × 20,480 × 4 bytes; 425 MB
on SciFact, 8 GB at 100k documents). Upgrade: write chunks into an `.npy` memmap.
`# ponytail:` bucket fill and per-slot sums use a sort + `reduceat` or a Hamming lookup
table, not tuned code; upgrade: threads over chunks, as in M1-4, if SciFact build time exceeds 1 min.

## Acceptance tests

- [x] AT1: `fde_encode` equals a plain-Python reference (loops over items, reps, buckets) for
  queries and documents, with empty buckets filled, `max_tokens` small (chunks of 1 and several
  items give the same result as one chunk), an empty item (all zeros), and `fill_empty` on a
  query. `k_sim` 9 and mismatched shapes raise `ValueError`.
- [x] AT2: `random_signs` and `make_planes` return the stated shapes and dtypes, only ±1 (and
  ±1/√d_proj), the same values on a second call, a stable prefix when `n` grows, and a signs
  stream different from the normals' stream for the same seed.
- [x] AT3: `MuveraStage.scores` equals `fde_encode(query) @ fde_encode(docs).T` (rtol 1e-5) on
  small random data; `search` breaks ties by lower row; the constructor rejects widths that
  don't agree and non-finite planes; `arrays` rejects encodings that overflow float32.
- [x] AT4: `Index.create(muvera=True)` writes the manifest stage and three arrays above;
  `search` with `muvera` and with `muvera > maxsim@20` equals `Pipeline.run` on in-memory stages.
  `Index.open` raises `IndexFormatError` for a wrong `reps`, a `hyperplanes.npy` or
  `projection.npy` of the wrong shape, and `fdes.npy` of the wrong row count. NaN tokens are
  refused at `create`.
- [x] AT5: golden SciFact, seed 0: `muvera` nDCG@10 in [0.500, 0.580]; `muvera > maxsim@100`
  nDCG@10 within 0.6888 ± 0.01. (Recall@100 is recorded, not asserted.)
- [x] AT6: seeds 0, 1, 2 measured once (both pipelines, nDCG@10 and Recall@100) and recorded in
  the golden-values skill with the SciFact build time. If a seed lands outside the AT5 bands,
  stop and ask the owner; do not widen the band quietly.
- [x] AT7: `uv run pytest -q` and `uv run ruff check .` pass on all three CI operating systems;
  `tests/test_layers.py` still passes (no torch in `kernels/`).
- [x] AT8: the reviewer agent's verdict is `merge`.

## Tasks

| # | task | files it owns | depends on | parallel? |
|---|---|---|---|---|
| 1 | `random_signs`, `make_planes`, `fde_encode` | `kernels/rng.py`, `kernels/muvera.py`, `tests/kernels/test_rng.py`, `tests/kernels/test_muvera.py` | M1-4 | yes, with 2 (interface above) |
| 2 | `MuveraStage` | `components/muvera.py`, `tests/components/test_muvera.py` | none (kernel signature fixed above) | yes, with 1 |
| 3 | `Index` wiring | `orchestration/index.py`, `tests/orchestration/test_index.py` | 2 | yes, with 4 |
| 4 | golden test, seed measurement, docs | `tests/golden/test_scifact.py`, `.claude/skills/golden-values/SKILL.md`, `CLAUDE.md` (one Commands line if the runtime needs a filter), `START_HERE.md` if a command changes | 1, 2 | yes, with 3 |

## Changes

- 2026-11-03: review follow-up (owner: "go with the best possible solution"): the `fdes` finite check moved from the `MuveraStage` constructor (a full read of 425 MB on every `open`) to `arrays` at build; the constructor checks only the planes. Dropped the `DenseStage` call (width check is one comparison) and trimmed `from_segment`'s duplicate settings checks. AT3 reworded accordingly, no test removed. - reviewer low findings
- 2026-11-03: review: fixed the centering numbers under Out of scope (the first draft paired r40_k4_p32's uncentered 0.5530 with r40_k6_p32's centered 0.5440); text only. - reviewer verdict "merge"
- 2026-11-03: the alone band is [0.500, 0.580], not [0.533, 0.573] (seed 2 scored 0.5118; owner OK). The golden test runs seeds 0, 1, 2 (about 2 min), not seed 0 only, so AT6 stays checked - no interface or acceptance test added or removed.
- 2026-11-03: approved by the owner.
- 2026-11-03: drafted. Setting chosen by the owner (r40_k4_p32 over the paper's r20_k5_p16 and r40_k5_p32).
