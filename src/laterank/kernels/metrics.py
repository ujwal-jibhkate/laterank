"""Ranking metrics: nDCG@k and Recall@k.

Terminology:
  - ranked: [doc_id, ...] for one query, best first (what the system returned)
  - qrel:   {doc_id: relevance} ground truth for one query; unlisted docs have relevance 0
  - run:    {query_id: ranked};  qrels: {query_id: qrel}

Conventions follow BEIR / pytrec_eval so numbers are comparable with published results:
  - nDCG uses linear gain, rel / log2(rank + 1) (identical to 2^rel - 1 for binary relevance).
  - Recall@k divides by ALL relevant docs for the query, not by k.
  - Scores are computed per query, then averaged over judged queries (macro average). A judged
    query missing from the run scores 0.

Ported from smve-lab's metrics.py (pandas dropped: kernels stay numpy-only).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np


def dcg(relevances: Sequence[float] | np.ndarray) -> float:
    """Discounted cumulative gain: sum_i rel_i / log2(i + 1), with rank i starting at 1."""
    rel = np.asarray(relevances, dtype=np.float64)
    if rel.size == 0:
        return 0.0
    return float(np.sum(rel / np.log2(np.arange(2, rel.size + 2))))


def ndcg_at_k(ranked: Sequence[str], qrel: Mapping[str, float], k: int) -> float:
    """DCG of the top k returned docs, divided by the best achievable DCG (IDCG): the DCG
    you'd get if the k slots held the most relevant docs in order."""
    gains = [qrel.get(doc, 0) for doc in ranked[:k]]
    idcg = dcg(sorted(qrel.values(), reverse=True)[:k])
    return dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: Sequence[str], qrel: Mapping[str, float], k: int) -> float:
    """Fraction of all relevant docs (relevance > 0) that appear in the top k."""
    n_rel = sum(1 for r in qrel.values() if r > 0)
    if n_rel == 0:
        return 0.0
    return sum(1 for doc in ranked[:k] if qrel.get(doc, 0) > 0) / n_rel


Metric = Callable[[Sequence[str], Mapping[str, float], int], float]


def mean_metric(
    run: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, float]],
    metric: Metric,
    k: int,
) -> float:
    """Average `metric@k` over every query in `qrels` (missing from `run` -> 0)."""
    if not qrels:
        raise ValueError("qrels is empty: nothing to evaluate")
    return float(np.mean([metric(run.get(qid, []), qrel, k) for qid, qrel in qrels.items()]))
