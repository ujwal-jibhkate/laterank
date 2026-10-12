"""SciFact golden tests: laterank's kernels must reproduce smve-lab's numbers on its embeddings.

Expected values: ~/Projects/smve-lab/results/cross_dataset/summary.md (BEIR test split).
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from laterank.components.base import Encoded
from laterank.components.dense import DenseStage
from laterank.components.hybrid import HYBRID_ALPHA, HybridStage
from laterank.components.lexical import LexicalStage
from laterank.components.maxsim import MaxSimReranker
from laterank.kernels.maxsim import maxsim_scores
from laterank.kernels.metrics import mean_metric, ndcg_at_k, recall_at_k
from laterank.kernels.topk import top_k
from laterank.orchestration.pipeline import parse_pipeline

TOL = 0.002
QRELS_SHA256 = "0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c"


def _check(scores: np.ndarray, data, ndcg10: float, recall100: float) -> None:
    ranked = top_k(scores, 100)
    run = {qid: [data.doc_ids[j] for j in ranked[i]] for i, qid in enumerate(data.query_ids)}
    _check_run(run, data, ndcg10, recall100)


def _check_run(run: dict[str, list[str]], data, ndcg10: float, recall100: float) -> None:
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


def _stages(data) -> dict:
    """Every stage, built straight from smve-lab's memory-mapped arrays (no index copy)."""
    dense = DenseStage(data.d_dense)
    lexical = LexicalStage(*LexicalStage.arrays(data.d_lexical).values(), n_docs=len(data.doc_ids))
    return {
        "dense": dense,
        "lexical": lexical,
        "hybrid": HybridStage(dense, lexical, HYBRID_ALPHA),
        "maxsim": MaxSimReranker(data.d_flat, data.d_offsets),
    }


# (pipeline, nDCG@10, Recall@100) from results/cross_dataset/summary.md. A `> maxsim@100` run
# has its first stage's Recall@100: the rerank reorders the same 100 documents.
PIPELINES = [
    ("dense > maxsim@100", 0.6966, 0.8970),
    ("lexical", 0.6376, 0.8980),
    ("hybrid", 0.6842, 0.9237),
    ("lexical > maxsim@100", 0.7048, 0.8980),
    ("hybrid > maxsim@100", 0.6977, 0.9237),
]


@pytest.mark.slow
@pytest.mark.parametrize(("spec", "ndcg10", "recall100"), PIPELINES)
def test_pipeline(scifact, spec, ndcg10, recall100):
    stages, pipeline, off = _stages(scifact), parse_pipeline(spec), scifact.q_offsets
    run = {}
    for i, qid in enumerate(scifact.query_ids):
        q = Encoded(scifact.q_flat[off[i] : off[i + 1]], scifact.q_dense[i], scifact.q_lexical[i])
        run[qid] = [scifact.doc_ids[r] for r in pipeline.run(stages, q, 100).rows]
    _check_run(run, scifact, ndcg10, recall100)
