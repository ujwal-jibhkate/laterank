---
id: M2-1
title: Evaluate pipelines, and the exhaustive first stage
status: approved
approved: 2026-11-04
---

# M2-1: Evaluate pipelines, and the exhaustive first stage

## Goal

After this checkpoint a user with relevance labels can compare pipelines on their own index in
one call: `index.evaluate(queries, qrels, ["exhaustive", "bm25 > maxsim@100"]).table()` prints
nDCG@10, Recall@100 and median milliseconds per query for each one. The `exhaustive` pipeline
(exact MaxSim against every document) becomes a first stage that every index has. M2-2 (`tune`)
needs both pieces: exhaustive is its answer key, and evaluate's timing is its latency budget.

## Out of scope

- `tune`, `Budget`, `set_default`: M2-2. Validating tune on 3 datasets: M2-3.
- Skipped: a dataset loader (`datasets.beir("scifact")`). Add it with the CLI in M3. Evaluate
  takes plain dicts, the same format as BEIR and ir_measures.
- Skipped: P, MRR, MAP, other cut-offs, and significance tests. Add them when someone asks;
  none of our reference numbers use them.
- Skipped: dropping a hit whose id equals the query id (BEIR's `ignore_identical_ids`, needed
  for ArguAna). Add it in M2-3, the first checkpoint that evaluates ArguAna.
- Skipped: returning the runs (`to_trec`, `to_pandas`). Add them when a user needs per-query
  output.
- Skipped: timing query encoding. `evaluate` takes encoded queries, as `search` does, so
  encoding happens before it. Add an `encode` column when the index can encode text queries
  itself (`search_text`, M3).
- Skipped: running a shared first stage once for several pipelines (PyTerrier's tree plan).
  Add it when evaluate time matters.

## Interfaces

```python
# components/maxsim.py: MaxSimReranker gains a first-stage method
class MaxSimReranker:
    def search(self, q: Encoded, k: int) -> Candidates:
        """Exact MaxSim against every document: the top min(k, n_docs), ties to the lower row.
        Scores the full token store in place (no subset copy)."""


# orchestration/index.py
EXHAUSTIVE = "exhaustive"  # a first stage every index has: the token store's MaxSimReranker.
# It is not written to manifest.stages, like `maxsim`. It is accepted by search(),
# Index.create(pipeline=...) and a manifest's default_pipeline.


class Index:
    def evaluate(
        self,
        queries: Mapping[str, Encoded],  # {query_id: encoded query}
        qrels: Mapping[str, Mapping[str, float]],  # {query_id: {doc_id: relevance}}
        pipelines: Sequence[str | Pipeline],
        *,
        repeats: int = 3,
    ) -> Evaluation:
        """Run each pipeline on every judged query and score it.

        Only queries in `qrels` run; extra queries are ignored. Each pipeline returns
        k = min(100, its last depth) documents, so Recall@100 of "x > maxsim@10" counts 10.

        Raises:
            PipelineSyntaxError: a pipeline string doesn't parse.
            ValueError: qrels or pipelines is empty, a judged query is missing from `queries`,
                a value isn't an Encoded, repeats < 1, or a pipeline uses a stage this index
                lacks. All checked before any search runs.
        """


# orchestration/evaluate.py (new)
@dataclass(frozen=True)
class Score:
    pipeline: str  # canonical string, str(parse_pipeline(...))
    ndcg_at_10: float
    recall_at_100: float
    p50_ms: float  # median over queries of each query's fastest of `repeats` searches


@dataclass(frozen=True)
class Evaluation:
    scores: tuple[Score, ...]  # in the order the pipelines were given
    n_queries: int

    def table(self) -> str:
        """One line per pipeline, e.g.
        pipeline           nDCG@10   R@100   p50 ms
        exhaustive          0.6991  0.9203    535.0
        bm25 > maxsim@100   0.7086  0.9249      8.1
        """
```

## Decisions

| decision | why | differs from design doc? |
|---|---|---|
| `exhaustive` is a method on `MaxSimReranker`, not a new class | it reads the same token store; one method, no new file | no |
| metrics reuse `kernels/metrics.py` (nDCG@10, Recall@100), no pytrec_eval or ranx | ours already follow BEIR's conventions; ranx pulls in numba, pandas and seaborn | no |
| timing: `time.perf_counter` around each `search`, `repeats` passes over all queries, keep each query's fastest, report the median | ann-benchmarks' approach (best of several runs, report p50). Keeping the fastest makes a separate warmup pass unnecessary: the first pass is the warmup | owner agreed "warmup + best of 3"; this gets the same thing without the extra pass |
| encoding is not timed; queries arrive encoded | `search` takes `Encoded`; smve-lab's latencies also excluded encoding, so the numbers stay comparable | owner agreed "encoding in its own column"; deferred to M3 (see Out of scope) |
| hits from the first pass are scored | search is deterministic; the other passes only time | no |
| `Index.evaluate` delegates to `orchestration/evaluate.py` | keeps `index.py` (340 lines) from growing; M2-2's tune report will live next to it | no |
| golden test checks `exhaustive` through the existing `_run` harness, not through `Index.evaluate` | `Index.evaluate` needs an index folder, and copying SciFact's token store is 3.8 GB. Unit tests check that evaluate's numbers equal `mean_metric` | the session plan said "through evaluate" |

## Acceptance tests

- [x] AT1: `MaxSimReranker.search` equals a brute-force reference (`maxsim_scores` over all docs, then a stable sort by score with ties to the lower row) on a random small token store, including k > n_docs (returns n_docs rows).
- [x] AT2: `exhaustive` works as a pipeline's first stage in `search`, inside `"exhaustive > maxsim@5"`, and as the default pipeline of an index built with no other stage (`Index.create(..., pipeline="exhaustive")`, then `Index.open`).
- [x] AT3: on a small index, `evaluate` with two pipelines gives the same nDCG@10 and Recall@100 as `mean_metric` over a run built with `search`. `scores` keeps the given order, `p50_ms` is finite and > 0, and `table()` has a header plus one line per pipeline.
- [x] AT4: each error in `evaluate`'s Raises list is raised, and no search runs first (checked with a stage that counts calls).
- [x] AT5 (golden, slow): SciFact `exhaustive` through `parse_pipeline("exhaustive").run(...)` gives nDCG@10 0.6991 and Recall@100 0.9203 (±0.002).
- [x] AT6: `uv run pytest -q`, `uv run ruff check .` and `uv run ruff format --check .` pass, and `tests/test_layers.py` still passes (no new imports pointing up).
- [x] AT7: reviewer agent verdict `merge`.

## Tasks

| # | task | files it owns | depends on | parallel? |
|---|---|---|---|---|
| 1 | `MaxSimReranker.search`; accept `exhaustive` in `_check_pipeline` and wire it into `Index`; tests AT1, AT2 | `components/maxsim.py`, `orchestration/index.py`, `tests/components/test_maxsim.py`, `tests/orchestration/test_index.py` | - | no |
| 2 | `Score`, `Evaluation`, `evaluate`, `Index.evaluate`; tests AT3, AT4 | `orchestration/evaluate.py`, `orchestration/index.py` (one method), `tests/orchestration/test_evaluate.py` | 1 | no |
| 3 | golden AT5 | `tests/golden/test_scifact.py` | 1 | yes, with 2 |

## Changes

- 2026-11-05: `Index.evaluate` also raises `ValueError` for a judged query whose token dim differs
  from the index (reviewer finding; a late failure would lose a long run). The engine
  `orchestration/evaluate.py::evaluate(search, doc_id, queries, qrels, pipelines, repeats, dim)`
  takes the index as two callables plus `dim`; it is not part of the public Interfaces.
- 2026-11-05: the "first stages" error text now lists `exhaustive`; three existing test regexes
  updated. `MaxSimReranker.rerank` and `search` share `_check_dim`, run before any copy.
- 2026-11-05: AT5 verified with `LATERANK_GOLDEN_DIR=~/Projects/smve-lab/artifacts/embeddings
  pytest -m slow -k exhaustive` (2 passed, 182 s; `test_exhaustive_pipeline` is the new one).
- 2026-11-05: reviewer verdict `merge` (4 low findings: 1 and 2 fixed; 3 and 4 skipped: a test
  nobody asked for, and an unreachable empty `Evaluation`).
