---
id: M1-4
title: SMVE first stage
status: approved
approved: 2026-10-12
---

# M1-4: SMVE first stage

## Goal

An `smve` first stage: TopK's SMVE turns each document's token vectors into one sparse vector
over random anchors, and scores a query with one sparse dot product. The kernel exists since
M0-2; this checkpoint stores it in an index, encodes queries at search time, and checks it on
SciFact. It is the first stage that depends on random numbers, so it also fixes how an index
pins its anchors (design §7). Settings are smve-lab's `balanced`: w = 65,536, k = 32, centered.

## Out of scope

- Skipped: choosing w, k, reps, seed or centering (`smve(preset=quality)`). Add with presets in
  M1-6 and `tune` in M2. The manifest records every setting, so a later index stays readable.
- Skipped: storing the anchor matrix (256 MB at w = 65,536, bigger than the index itself).
  Anchors are regenerated from (seed, d, w) and checked against the manifest's hash.
- Skipped: a faster top-k (group-max prefilter instead of `argpartition`). Add when index build
  time matters beyond SciFact size (see `# ponytail` in Decisions).
- MUVERA: M1-5. NFCorpus and ArguAna golden tests: a later M1 checkpoint.

## Interfaces

```python
# kernels/smve.py: topk_blocks partitions row slices in threads (signature unchanged)
def topk_blocks(P: np.ndarray, k: int, reps: int = 1) -> tuple[np.ndarray, np.ndarray]: ...


# components/smve.py (new)
SMVE_BALANCED = {"w": 65536, "k": 32, "reps": 1, "seed": 0, "center": True}


class SMVEStage:
    def __init__(
        self,
        indptr: np.ndarray,
        rows: np.ndarray,
        weights: np.ndarray,
        n_docs: int,
        anchors: np.ndarray,  # (d, reps * w) float32
        k: int,
        reps: int,
        center: np.ndarray | None,  # (d,) float32
    ):
        """Postings as in `lexical` (checked by LexicalStage); ValueError on bad shapes or k."""

    @staticmethod
    def arrays(
        flat: np.ndarray, offsets: np.ndarray, cfg: dict
    ) -> tuple[dict[str, np.ndarray], str]:
        """Encode every document -> (segment arrays below, anchor_sha256 for the manifest)."""

    @classmethod
    def from_segment(cls, seg: Segment, cfg: dict, n_docs: int) -> SMVEStage:
        """Regenerates the anchors; ValueError if their SHA-256 differs from cfg["anchor_sha256"]."""

    def scores(self, q: Encoded) -> np.ndarray:
        """(n_docs,) float32: SMVE(q.tokens, sum-pooled) . SMVE(doc, mean-pooled)."""

    def search(self, q: Encoded, k: int) -> Candidates: ...


# orchestration/index.py
class Index:
    @classmethod
    def create(
        cls,
        path,
        docs,
        tokens,
        *,
        encoder,
        dense=None,
        lexical=None,
        bm25=False,
        smve: bool = False,
        pipeline="dense > maxsim@100",
    ) -> Index: ...

    # smve=True -> stage "smve" with SMVE_BALANCED, from the token vectors
```

```text
manifest.json stage:
  "smve": {"type": "smve", "w": 65536, "k": 32, "reps": 1, "seed": 0, "center": true,
           "anchor_sha256": "<64 hex>"}
seg-000001/smve/indptr.npy    (reps * w + 1,) int64      postings, as lexical/
seg-000001/smve/rows.npy      (nnz,) int32
seg-000001/smve/weights.npy   (nnz,) float32             mean-pooled document values
seg-000001/smve/center.npy    (d,) float32               only when center is true
```

## Decisions

| decision | why | differs from design doc? |
|---|---|---|
| Anchors regenerated at `open` from (seed, d, reps·w), hash checked against the manifest | design §7; storing them would cost 256 MB. `# ponytail:` ~2.2 s per open at w = 65,536; upgrade: generate on first query, or cache | no |
| Settings fixed to `balanced` (w 65,536, k 32, reps 1, seed 0, centered) and all recorded | the golden band is for this setting; presets choose others in M1-6 | no |
| The center is the documents' mean token (`kernels.smve.token_mean`, 200,000 tokens, portable sample) and is stored | queries must be centered with the same vector the documents were | no |
| Document vectors stored as postings, checked by reusing `LexicalStage` | same layout and same scoring kernel as `lexical`; no second validator | no |
| `topk_blocks` partitions row slices in a thread pool (`os.cpu_count()` threads) | `argpartition` was 89% of the encode time and single-threaded: SciFact took ~20 min. Threads (one slice per core, a pool per call) cut SciFact to ~5.2 min with identical results (numpy releases the GIL). `# ponytail:` vs torch's 3.3 min; upgrade: group-max prefilter | adds threads to a kernel |
| The golden check is a band, not a point (golden-values skill) | laterank's anchors come from its own generator, so only the statistics match smve-lab | no |
| Measure laterank seeds 0, 1, 2 once and record them in the golden-values skill | the ±0.01 rerank tolerance is marked provisional until confirmed across seeds. Owner OK | no |
| The SMVE golden tests stay under `-m slow` (~5 min more); skip them with `-k "not smve"`, written in CLAUDE.md's Commands | pytest's built-in name filter needs no new marker; the full golden suite runs at checkpoint ends anyway. Owner asked for Claude's pick | no |

## Acceptance tests

- [x] AT1: `topk_blocks` with threads returns the same set of columns per row as a plain
  `argsort` reference on random input, for reps 1 and 4 (existing tests keep passing).
- [x] AT2: `SMVEStage.scores` equals `smve_encode(query) @ smve_encode(docs).T` (rtol 1e-5) on
  small random data, centered and not; `search` breaks ties by lower row.
- [x] AT3: `Index.create(smve=True)` writes the manifest stage and four arrays above; `search`
  with `smve` and `smve > maxsim@20` equals `Pipeline.run` on the in-memory stages.
  `Index.open` raises `IndexFormatError` for a wrong `anchor_sha256`, a wrong `w`, and a
  `center.npy` of the wrong length.
- [x] AT4: golden SciFact, seed 0: `smve` nDCG@10 in [0.537, 0.572]; `smve > maxsim@100`
  nDCG@10 within 0.6898 ± 0.01.
- [x] AT5: seeds 0, 1, 2 measured once (both pipelines, nDCG@10 and Recall@100) and recorded
  in the golden-values skill; every value lands in the AT4 bands. SciFact encode time recorded.
- [ ] AT6: `uv run pytest -q` and `uv run ruff check .` pass on all three CI operating systems.
- [x] AT7: the reviewer agent's verdict is `merge`.

## Tasks

| # | task | files it owns | depends on | parallel? |
|---|---|---|---|---|
| 1 | threaded `topk_blocks` | `kernels/smve.py`, `tests/kernels/test_smve.py` | M1-3 | yes, with 2 |
| 2 | `SMVEStage` | `components/smve.py`, `tests/components/test_smve.py` | — (kernel API unchanged) | yes, with 1 |
| 3 | `Index` wiring | `orchestration/index.py`, `tests/orchestration/test_index.py` | 2 | yes, with 4 |
| 4 | golden test + seed measurement | `tests/golden/test_scifact.py`, `.claude/skills/golden-values/SKILL.md`, `CLAUDE.md` (one Commands line) | 1, 2 | yes, with 3 |

## Changes

- 2026-10-13: review round 2: the token finiteness check uses min/max (the `isfinite` array was
  a 1.8 GB temporary on SciFact); AT1's thread test uses 2001 rows (a short last slice). -
  reviewer verdict "fix first"
- 2026-10-13: review fixes, no interface or acceptance test added or removed. The thread pool
  is created per call (a module-level pool hung in fork()ed children). Slices are T / cores
  rows (min 32), not min 256 rows (that capped the encoder at 4 threads): SciFact build 8.3 →
  5.2 min. `Index.create(smve=True)` refuses non-finite token vectors with `ValueError` (the
  `argpartition(P, w - k)` form ranks NaN as largest; noted in the `topk_blocks` docstring).
  START_HERE.md gained the `-k "not smve"` command. - reviewer verdict "fix first"
- 2026-10-12: approved; owner OK on threads and the seed check, golden runtime choice left to Claude (row added under Decisions).
- 2026-10-12: drafted.
