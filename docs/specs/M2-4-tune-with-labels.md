---
id: M2-4
title: Tune with labels as the recommended mode
status: done
approved: 2026-11-06
---

# M2-4: Tune with labels as the recommended mode

## Goal

After this checkpoint a user who has a few judged queries can ask the index to pick its pipeline by
real relevance: `report = index.tune(Budget(latency_ms=20), queries=q, qrels=qrels)`, then
`index.set_default(report.best)`. It runs every candidate pipeline through `evaluate` and picks
the highest nDCG@10 within the budget (design doc §8's fallback). M2-3 measured label-free tune
missing the 0.01 test on all 3 datasets, so the docs now name the label mode as the recommended
one and describe label-free tune as "agrees with MaxSim", not "finds relevant documents".

## Out of scope

- Changing how label-free tune works (spans, answer key): it stays as M2-2 built it.
- A dataset loader or CLI to read qrels files: M3 (plain dicts, like `evaluate`).
- Skipped: a `repeats` argument. Label mode times like `evaluate`'s default (best of 3); add it
  when a user's query set is too big for 3 passes.
- Skipped: optimising another metric (Recall@100, MRR). Only nDCG@10; add a `metric=` when
  someone needs it.
- Skipped: a held-out split or noise estimate on the pick (300 queries can't tell 0.705 from
  0.708). Add when users report the pick flips between runs.

## Interfaces

```python
# orchestration/tune.py
@dataclass(frozen=True)
class Point:
    pipeline: str  # canonical string
    score: float  # agreement (label-free) or nDCG@10 (labels); was `agreement` (M2-2)
    p50_ms: float


@dataclass(frozen=True)
class TuneReport:
    best: str | None  # highest score with p50_ms <= budget (ties: faster); None if none fits
    frontier: tuple[Point, ...]  # Pareto frontier, fastest first, score strictly rising
    n_queries: int
    budget: Budget
    metric: str  # "agreement" or "nDCG@10"; table()'s header uses it


def _report(points: Sequence[Point], budget: Budget, n_queries: int, metric: str) -> TuneReport:
    """Frontier + best, the rule M2-2's `tune` already uses; both modes call it."""


def tune_with_labels(
    search: Callable[[Pipeline, Encoded, int], Candidates],
    doc_id: Callable[[int], str],
    queries: Mapping[str, Encoded],
    qrels: Mapping[str, Mapping[str, float]],
    pipelines: Sequence[Pipeline],
    budget: Budget,
    dim: int,
    ignore_identical_ids: bool = False,
) -> TuneReport:
    """`evaluate(..., repeats=3)` over `pipelines`, each Score -> Point(pipeline, ndcg_at_10,
    p50_ms), then `_report(..., metric="nDCG@10")`. Raises what `evaluate` raises."""


# orchestration/index.py
class Index:
    def tune(
        self,
        budget: Budget,
        *,
        queries: Mapping[str, Encoded] | None = None,
        qrels: Mapping[str, Mapping[str, float]] | None = None,
        ignore_identical_ids: bool = False,
        encoder: Encoder | None = None,
        n_queries: int | None = None,  # None means 500
        seed: int | None = None,  # None means 0
    ) -> TuneReport:
        """Pick the pipeline for this index within the budget. Same candidates in both modes.

        With labels (recommended): give `queries` and `qrels` like `Index.evaluate`; picks the
        highest nDCG@10.
        Without labels: picks the pipeline that best agrees with exhaustive MaxSim on span
        queries. Agreeing with MaxSim isn't finding relevant documents: it can't pick a
        pipeline that beats MaxSim, and misses `bm25` (M2-3: up to 0.052 nDCG@10 lost).

        Raises:
            ValueError: everything M2-2's tune raises; only one of `queries`/`qrels` given;
                `encoder`, `n_queries` or `seed` given with `qrels`; `ignore_identical_ids`
                given without `qrels`; anything `evaluate` refuses. All before any search.
        """
```

`README.md` gains a short "Choosing a pipeline" section: the label-mode example first, then the
label-free one with the caveat sentence and a link to `docs/dev/tune_validation.md`.
`scripts/validate_tune.py` follows the `agreement` → `score` rename (one line).

## Decisions

| decision | why | differs from design doc? |
|---|---|---|
| one entry point, `Index.tune(budget, queries=, qrels=)` | owner's pick (2026-11-06): one way to tune; labels only swap the answer key | no |
| label mode = `evaluate` + "best nDCG@10 within budget" | §8's fallback; `validate_tune.py` already picks this way, so it's the rule M2-3 measured | no |
| rename `Point.agreement` → `score`, add `TuneReport.metric` | one report type for both modes; `agreement` holding nDCG@10 would mislead. 6 lines use it, nothing released | yes: changes M2-2's interface |
| span-only arguments with `qrels` raise, not ignored | a silently ignored `seed=` misleads; `None` defaults let tune tell "given" from "default" | no |
| timing best of 3 in label mode (label-free stays 1) | same as `evaluate`; judged sets are ~300 queries, so 3 passes cost little next to 500 spans | no |
| caveat lives in `Index.tune`'s docstring and README | where a user reads before calling it; numbers point to `tune_validation.md` | no |

## Acceptance tests

- [x] AT1: on a small index (FakeEncoder, synthetic qrels), label-mode `best` and every frontier point equal a reference built from `Index.evaluate` over `candidates(...)`: highest nDCG@10 with p50 ≤ budget, ties to the faster; `report.metric == "nDCG@10"`; a budget below every p50 gives `best is None`.
- [x] AT2: `ignore_identical_ids=True` reaches `evaluate` (a query whose id is a document id never counts itself).
- [x] AT3: each new error in `Index.tune`'s Raises list (only `queries`, only `qrels`, `encoder`/`n_queries`/`seed` with `qrels`, `ignore_identical_ids` without `qrels`) raises before any search runs.
- [x] AT4: label-free tests pass after the rename (`report.metric == "agreement"`); `table()` header shows the metric name.
- [x] AT5: `README.md` has the label-mode example before the label-free one, and the caveat sentence.
- [x] AT6: `uv run pytest -q`, `ruff check`, `ruff format --check`, `tests/test_layers.py` pass (ruff also covers `scripts/validate_tune.py`).
- [x] AT7: reviewer agent verdict `merge`.

## Tasks

| # | task | files it owns | depends on | parallel? |
|---|---|---|---|---|
| 1 | rename to `score`, `metric`, `_report`, `tune_with_labels`; AT4; script rename | `orchestration/tune.py`, `tests/orchestration/test_tune.py`, `scripts/validate_tune.py` | - | no |
| 2 | `Index.tune` label mode + checks + docstring caveat; AT1-AT3 | `orchestration/index.py`, `tests/orchestration/test_index.py` | 1 | no |
| 3 | README "Choosing a pipeline"; AT5 | `README.md` | - | yes (with 1) |

## Changes

Dated lines for anything changed after approval: `- 2026-10-06: <what> - <why>`.

- 2026-11-06 (reviewer round 1): the loss figure in `Index.tune`'s docstring and the README is
  0.052, not 0.042 - `tune_validation.md` has 0.0524 (SciFact) and 0.0431 (NFCorpus) below 1 ms;
  0.0422 is ArguAna's. The README example imports `Budget` and guards `best is None`.
