---
id: M1-1
title: Index and pipeline, dense first
status: done
approved: 2026-10-11
---

# M1-1: Index and pipeline, dense first

## Goal

An `Index` that is built from precomputed vectors, saved as a folder, reopened, and searched
with a pipeline string such as `dense > maxsim@100`. This is the first time laterank's layers
work together: the kernels and storage from M0, plus new components (the `dense` first stage and
the `maxsim` reranker) and orchestration (the pipeline parser and `Index`). Dense comes first
because it needs no new math, so this checkpoint can focus on the interfaces that every later
stage (lexical, hybrid, BM25, SMVE, MUVERA) plugs into.

## Out of scope

- Encoders and text queries (`index.search("some text")`), the public `laterank.build/open/load`,
  and `EncoderMismatchError`: M1-6. Here a query arrives already encoded.
- Other first stages (lexical, hybrid, BM25, SMVE, MUVERA) and the cross-encoder reranker:
  M1-2 to M1-5, and later.
- `Results` (timings, `to_trec`, `to_pandas`), `Hit.stage_scores`, `Hit.passage`, `evaluate`,
  `tune`, `set_default`: M2 and M3.
- Skipped: stage parameters inside pipeline strings (`hybrid(alpha=0.5)`). Add them when a
  stage needs a per-search override (M1-2 hybrid or M1-4 SMVE presets). Settings live in the
  manifest's `stages` until then.
- Skipped: the `exhaustive` pipeline. Add it with `evaluate`/`tune` (M2), which use it as the
  reference. The M0-4 golden test already checks exhaustive MaxSim through the kernels.
- Skipped: indexes with more than one segment. `Index.open` refuses them. Add a global-row map
  across segments when `add()` lands in 0.2. M1-3 (BM25) designs global statistics.
- Skipped: batched queries. `search` takes one query; a loop is fast enough for SciFact's
  300 queries. Add a batch path when a measurement shows the loop is the bottleneck.
- Skipped: `overwrite=True` and a plugin registry with entry points. `Index` knows its stages
  by a plain dict. Add both when users need them (M3).

## Interfaces

```python
# laterank/errors.py (adds two classes)
class PipelineSyntaxError(LaterankError, ValueError): ...  # a pipeline string doesn't parse,
                                                           # or a depth is larger than the one before it
class IndexExistsError(LaterankError, FileExistsError): ...  # Index.create on a non-empty folder

# laterank/components/base.py
@dataclass(frozen=True)
class Encoded:                      # one encoded query
    tokens: np.ndarray              # (t, dim) float token vectors, t >= 1
    dense: np.ndarray | None = None # (d,) float single vector, if the encoder has one
@dataclass(frozen=True)
class Candidates:
    rows: np.ndarray                # (n,) int64 global document rows, best first
    scores: np.ndarray              # (n,) float32, same length
    # __post_init__: ValueError for a length mismatch or a wrong dtype
class FirstStage(Protocol):
    def search(self, q: Encoded, k: int) -> Candidates: ...    # top min(k, n_docs)
class Reranker(Protocol):
    def rerank(self, q: Encoded, cands: Candidates) -> Candidates: ...  # same rows, new order

# laterank/components/dense.py
class DenseStage:                   # brute-force inner product
    def __init__(self, vectors: np.ndarray): ...   # (n_docs, d) float
    @staticmethod
    def arrays(vectors: np.ndarray) -> dict[str, np.ndarray]:  # {"dense/vectors.npy": float32}
    @classmethod
    def from_segment(cls, seg: Segment) -> "DenseStage": ...   # memory-mapped
    def search(self, q: Encoded, k: int) -> Candidates: ...
    # ValueError if q.dense is None or its length isn't d

# laterank/components/maxsim.py
class MaxSimReranker:               # exact MaxSim on the candidates only
    def __init__(self, flat: np.ndarray, offsets: np.ndarray): ...  # the index's token store
    @classmethod
    def from_segment(cls, seg: Segment) -> "MaxSimReranker": ...    # tokens/{flat,offsets}.npy
    def rerank(self, q: Encoded, cands: Candidates) -> Candidates: ...

# laterank/orchestration/pipeline.py
@dataclass(frozen=True)
class Pipeline:
    first: str                                  # "dense"
    rerankers: tuple[tuple[str, int], ...] = () # (("maxsim", 100),)
    def __str__(self) -> str: ...               # canonical: "dense > maxsim@100"
    def run(self, stages: Mapping[str, FirstStage | Reranker], q: Encoded, k: int) -> Candidates: ...
def parse_pipeline(spec: str) -> Pipeline: ...  # PipelineSyntaxError
# grammar:  pipeline := name ( ">" name "@" depth )*      spaces around tokens are ignored
#           name := [a-z][a-z0-9-]*                       depth := positive integer
# run: the first stage is asked for the first depth (or k when there are no rerankers); each
# reranker sees only the top `depth` of what came before; the result is cut to k.
# ValueError if k < 1, if k > the last depth, or if a name is missing from `stages`.

# laterank/orchestration/index.py
@dataclass(frozen=True)
class Hit:
    id: str; rank: int; score: float            # rank starts at 1; score of the LAST stage
    text: str | None; metadata: dict
class Index:
    path: Path; manifest: Manifest
    @classmethod
    def create(cls, path: Path, docs: Sequence[Mapping], tokens: tuple[np.ndarray, np.ndarray], *,
               encoder: EncoderInfo, dense: np.ndarray | None = None,
               pipeline: str = "dense > maxsim@100") -> "Index": ...
    @classmethod
    def open(cls, path: Path) -> "Index": ...
    @property
    def default_pipeline(self) -> Pipeline: ...
    def search(self, q: Encoded, k: int = 10, *, pipeline: str | Pipeline | None = None) -> list[Hit]: ...
```

```text
my_index/
  manifest.json      stages: {"dense": {"type": "dense", "dim": 1024}}
                     default_pipeline: "dense > maxsim@100"
  seg-000001/
    docs.jsonl, docs.offsets.npy
    tokens/flat.npy, tokens/offsets.npy     dtype as given (float16 or float32)
    dense/vectors.npy                       float32, only when `dense` was given
```

## Decisions

| decision | why | differs from design doc? |
|---|---|---|
| A pipeline is the string from the design, parsed into a frozen `Pipeline`. `parse_pipeline(str(p)) == p` always holds | no library we researched saves a pipeline; a string is readable in Python, the CLI and the manifest, and the manifest format doesn't change | no |
| The first name in the string is the first stage and the rest are rerankers, with two separate Protocols | `maxsim > dense` is caught when the string is used, not halfway through a search | no |
| The depth sits on the reranker (`maxsim@100`), not on the first stage | design grammar; the same as PyTerrier's `% 100` placed just before the reranker | no |
| `run` raises `ValueError` when k is larger than the last depth | PyLate silently raises the depth; RAGatouille aborts with a print. An error is explicit, and the fix is in the message | no |
| The reranker breaks ties by lower **global row**: it sorts the candidates by row before calling `top_k` | the result doesn't depend on the first stage's order. smve-lab's `sorted` kept candidate order; exact float32 ties are rare, and the ±0.002 tolerance covers them | **yes**: smve-lab differs |
| Only the last stage's score is returned, as float32 | dense and MaxSim scores are on different scales | no |
| Rerankers have no entry in the manifest's `stages`; the token store belongs to the index | they own no data folder. `Index.open` checks that the default pipeline's first stage is in `stages` and its rerankers are known | fills a gap in the design |
| Components sit flat in `components/` (`base.py`, `dense.py`, `maxsim.py`) | three files don't need `stages/` and `rerank/` subfolders. Split them when there are more | **yes**: the design has subfolders |
| `Encoded` holds `tokens` and an optional `dense`; M1-2 adds `lexical` | the design's `Encoded` type, with only the heads used so far | no (fewer fields) |
| Dense vectors are stored as float32; tokens are stored in the dtype given | smve-lab's dense vectors are float32 (5183 × 1024, 21 MB); tokens are float16 (3.8 GB), and converting them would double that | no |
| The golden test builds `DenseStage` and `MaxSimReranker` straight from smve-lab's memory-mapped arrays and calls `Pipeline.run`, without `Index.create` | `Index.create` would write a 3.8 GB copy. `Index` itself is tested on small synthetic data against the same `Pipeline.run` | no |

## Acceptance tests

- [x] AT1: `parse_pipeline` accepts `dense`, `dense > maxsim@100` and `  dense>maxsim@100 > maxsim@10`, and
  `str()` gives the canonical form. It raises `PipelineSyntaxError`, with a message naming the
  problem, for `""`, `"dense >"`, `"dense > maxsim"`, `"dense > maxsim@0"`, `"> maxsim@10"`,
  `"Dense"`, `"dense(alpha=1)"` and the increasing depths `"dense > maxsim@10 > maxsim@20"`. A
  hypothesis test checks `parse_pipeline(str(p)) == p`.
- [x] AT2: `Pipeline.run` asks the first stage for the first depth, hands each reranker only the
  top `depth` candidates, and cuts the result to `k`, which a fake stage that records its calls
  confirms. It raises `ValueError` for `k < 1`, for `k` larger than the last depth, and for a
  name missing from `stages`.
- [x] AT3: `DenseStage.search` matches a naive loop (score every document, sort by score
  descending and then by row) on random data. Equal scores come back in row order, and `k`
  larger than the corpus returns every document.
- [x] AT4: `MaxSimReranker.rerank` matches a naive per-document MaxSim loop. Shuffling the input
  candidates doesn't change the output, so ties go to the lower global row.
- [x] AT5: `Index.create` → `Index.open` → `search` returns the same rows and scores as
  `Pipeline.run` on the in-memory arrays, as `Hit`s with 1-based ranks and the stored
  text and metadata. The manifest holds `{"dense": {"type": "dense", "dim": d}}` and the default
  pipeline.
- [x] AT6: errors at the boundary. `create` raises `IndexExistsError` on a non-empty folder and
  `ValueError` (writing nothing) when docs, token offsets and dense rows disagree in count or
  `dim`. `search` raises `ValueError` for a pipeline whose first stage wasn't built, and the
  message lists the stages that were. `open` raises `IndexFormatError` for an unparsable
  default pipeline and for a manifest with two segments.
- [x] AT7: golden. SciFact `dense > maxsim@100` gives nDCG@10 = 0.6966 ± 0.002
  (`results/cross_dataset/summary.md`, line 39) and Recall@100 = 0.8970 ± 0.002. The rerank
  reorders the same 100 documents, so its Recall@100 equals dense's.
- [x] AT8: `tests/test_layers.py` covers the new `components/` and `orchestration/` folders, and
  `uv run pytest -q` and `uv run ruff check .` pass on all three CI operating systems.
- [x] AT9: the reviewer agent's verdict is `merge`.

## Tasks

| # | task | files it owns | depends on | parallel? |
|---|---|---|---|---|
| 1 | errors + component types | `errors.py`, `components/{__init__,base}.py` | M0 | no (others need it) |
| 2 | dense stage + MaxSim reranker | `components/{dense,maxsim}.py`, `tests/components/test_{dense,maxsim}.py` | 1 | yes, with 3 |
| 3 | pipeline parser + runner | `orchestration/{__init__,pipeline}.py`, `tests/orchestration/test_pipeline.py` | 1 | yes, with 2 |
| 4 | Index + Hit | `orchestration/index.py`, `tests/orchestration/test_index.py` | 2, 3 | no |
| 5 | golden test | `tests/golden/test_scifact.py` | 2, 3 | yes, with 4 |

## Changes

- 2026-10-11: review round 2 (verdict "merge") follow-ups: `Index.open` turns `KeyError`/
  `IndexError`/`ValueError` from a malformed manifest stage or stored array into
  `IndexFormatError`. A `# ponytail:` note in `Index.create` names the Windows cleanup ceiling
  (memmaps still open can block the delete). - reviewer low findings
- 2026-10-11: review fixes, no interface added or removed. `Pipeline` checks its own fields
  (names, depths >= 1, depths don't grow), so a hand-made `Pipeline` is held to the grammar too;
  `parse_pipeline` only splits the string. `run` takes `k` through `operator.index` (TypeError
  for 2.5). `Encoded` stores its inputs as arrays. `Index.open` raises `IndexFormatError` when
  token offsets, token dim or a stage's vectors disagree with the documents or the manifest.
  A failed `Index.create` leaves the folder as it found it (absent, or empty). The "stage not
  built" error says how to fix it. Recorded: tokens of any float dtype are stored as given
  (float64 too). - reviewer verdict "fix first"
- 2026-10-11: approved as drafted.
- 2026-10-11: drafted.
