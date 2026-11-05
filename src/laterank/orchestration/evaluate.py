"""Evaluate pipelines against relevance labels: nDCG@10, Recall@100 and median latency.

`evaluate` is the engine behind `Index.evaluate`. It gets the index only as two callables
(`search` and `doc_id`), so it can be tested without an index folder.

Timing follows ann-benchmarks: run every query `repeats` times, keep each query's fastest time,
report the median over queries. The first pass doubles as the warmup, because the fastest run
wins. Only the pipeline's own search is timed (not mapping rows to ids, not scoring).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from laterank.components.base import Candidates, Encoded
from laterank.kernels.metrics import mean_metric, ndcg_at_k, recall_at_k
from laterank.orchestration.pipeline import Pipeline

NDCG_K, RECALL_K = 10, 100


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
        w = max(len("pipeline"), *(len(s.pipeline) for s in self.scores))
        lines = [f"{'pipeline':<{w}}  {'nDCG@10':>7}  {'R@100':>6}  {'p50 ms':>8}"]
        lines += [
            f"{s.pipeline:<{w}}  {s.ndcg_at_10:>7.4f}  {s.recall_at_100:>6.4f}  {s.p50_ms:>8.1f}"
            for s in self.scores
        ]
        return "\n".join(lines)


def evaluate(
    search: Callable[[Pipeline, Encoded, int], Candidates],
    doc_id: Callable[[int], str],
    queries: Mapping[str, Encoded],
    qrels: Mapping[str, Mapping[str, float]],
    pipelines: Sequence[Pipeline],
    repeats: int,
    dim: int,
) -> Evaluation:
    """Run each pipeline on every judged query and score it (see `Index.evaluate`).

    Raises:
        ValueError: qrels or pipelines is empty, a judged query is missing from `queries`, a
            value isn't an Encoded or has token dim != `dim`, or repeats < 1. All checked before
            any search runs.
    """
    if not qrels:
        raise ValueError("qrels is empty: there is nothing to evaluate")
    if not pipelines:
        raise ValueError("pipelines is empty: give at least one, e.g. ['exhaustive']")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError(f"repeats must be an integer >= 1, got {repeats!r}")
    missing = [qid for qid in qrels if qid not in queries]
    if missing:
        raise ValueError(
            f"{len(missing)} judged queries are missing from `queries`, e.g. {missing[:3]}: "
            "encode them, or drop them from qrels"
        )
    bad = [qid for qid in qrels if not isinstance(queries[qid], Encoded)]
    if bad:
        raise ValueError(f"queries must be Encoded objects, but {bad[:3]} are not")
    wrong = [qid for qid in qrels if queries[qid].tokens.shape[1] != dim]
    if wrong:
        raise ValueError(f"query tokens must have dim {dim} like the index; wrong for {wrong[:3]}")

    qids = list(qrels)
    scores = []
    for p in pipelines:
        k = min(RECALL_K, p.rerankers[-1][1]) if p.rerankers else RECALL_K
        best = np.full(len(qids), np.inf)
        run: dict[str, list[str]] = {}
        for rep in range(repeats):
            for i, qid in enumerate(qids):
                t0 = time.perf_counter()
                cands = search(p, queries[qid], k)
                best[i] = min(best[i], time.perf_counter() - t0)
                if rep == 0:  # search is deterministic: later passes only time
                    run[qid] = [doc_id(int(r)) for r in cands.rows]
        scores.append(
            Score(
                str(p),
                mean_metric(run, qrels, ndcg_at_k, NDCG_K),
                mean_metric(run, qrels, recall_at_k, RECALL_K),
                float(np.median(best)) * 1000,
            )
        )
    return Evaluation(tuple(scores), len(qids))
