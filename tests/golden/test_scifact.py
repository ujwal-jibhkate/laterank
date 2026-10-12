"""SciFact golden tests: laterank's kernels must reproduce smve-lab's numbers on its embeddings.

Expected values: ~/Projects/smve-lab/results/cross_dataset/summary.md (BEIR test split).
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from laterank.kernels.maxsim import maxsim_scores
from laterank.kernels.metrics import mean_metric, ndcg_at_k, recall_at_k
from laterank.kernels.topk import top_k

TOL = 0.002
QRELS_SHA256 = "0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c"


def _check(scores: np.ndarray, data, ndcg10: float, recall100: float) -> None:
    ranked = top_k(scores, 100)
    run = {qid: [data.doc_ids[j] for j in ranked[i]] for i, qid in enumerate(data.query_ids)}
    got = (
        mean_metric(run, data.qrels, ndcg_at_k, 10),
        mean_metric(run, data.qrels, recall_at_k, 100),
    )
    assert got == pytest.approx((ndcg10, recall100), abs=TOL), f"nDCG@10, Recall@100 = {got}"


def test_qrels_fixture_is_the_file_smve_lab_used(scifact_qrels):
    path = Path(__file__).parent / "data" / "scifact-test-qrels.tsv"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == QRELS_SHA256
    assert (len(scifact_qrels), sum(map(len, scifact_qrels.values()))) == (300, 339)


@pytest.mark.slow
def test_exhaustive_maxsim(scifact):
    scores = maxsim_scores(scifact.q_flat, scifact.q_offsets, scifact.d_flat, scifact.d_offsets)
    _check(scores, scifact, ndcg10=0.6991, recall100=0.9203)


@pytest.mark.slow
def test_dense(scifact):
    _check(scifact.q_dense @ scifact.d_dense.T, scifact, ndcg10=0.6542, recall100=0.8970)
