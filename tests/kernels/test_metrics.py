"""Hand-computed checks for nDCG@k and Recall@k (ported from smve-lab)."""

import math

import pytest

from laterank.kernels.metrics import dcg, mean_metric, ndcg_at_k, recall_at_k

# Relevant docs are "a" and "c"; the system ranks: x, a, y, c, z
RANKED = ["x", "a", "y", "c", "z"]
QREL = {"a": 1, "c": 1}


def test_dcg():
    assert dcg([1, 0, 1]) == pytest.approx(1 / math.log2(2) + 1 / math.log2(4))
    assert dcg([]) == 0.0


def test_ndcg():
    # DCG = 1/log2(3) + 1/log2(5); IDCG = 1/log2(2) + 1/log2(3)
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(RANKED, QREL, 5) == pytest.approx(expected)
    assert ndcg_at_k(["a", "c"], QREL, 2) == pytest.approx(1.0)
    assert ndcg_at_k(RANKED, QREL, 1) == 0.0


def test_ndcg_ideal_is_truncated_at_k():
    # 3 relevant docs but k=1: a perfect top-1 must still score 1.0.
    assert ndcg_at_k(["a"], {"a": 1, "b": 1, "c": 1}, 1) == pytest.approx(1.0)


def test_ndcg_graded_uses_linear_gain():
    qrel = {"a": 2, "b": 1}
    # ranked b, a: DCG = 1/1 + 2/log2(3); IDCG = 2/1 + 1/log2(3)
    expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert ndcg_at_k(["b", "a"], qrel, 10) == pytest.approx(expected)


def test_recall():
    assert recall_at_k(RANKED, QREL, 2) == pytest.approx(1 / 2)
    assert recall_at_k(RANKED, QREL, 5) == pytest.approx(1.0)
    assert recall_at_k(RANKED, {"a": 0}, 5) == 0.0  # no relevant docs at all


def test_mean_metric_scores_missing_queries_as_zero():
    qrels = {"q1": QREL, "q2": {"b": 1}}
    assert mean_metric({"q1": RANKED}, qrels, recall_at_k, 5) == pytest.approx(0.5)
    assert mean_metric({"q1": ["a", "c"], "q2": ["b"]}, qrels, ndcg_at_k, 10) == pytest.approx(1)
    with pytest.raises(ValueError):
        mean_metric({}, {}, ndcg_at_k, 10)
