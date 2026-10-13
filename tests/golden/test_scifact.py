"""SciFact golden tests: laterank's kernels must reproduce smve-lab's numbers on its embeddings.

Expected values: ~/Projects/smve-lab/results/cross_dataset/summary.md (BEIR test split).
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from laterank.components.base import Encoded
from laterank.components.bm25 import BM25Stage
from laterank.components.dense import DenseStage
from laterank.components.hybrid import HYBRID_ALPHA, HybridStage
from laterank.components.lexical import LexicalStage
from laterank.components.maxsim import MaxSimReranker
from laterank.components.smve import SMVE_BALANCED, SMVEStage
from laterank.kernels.maxsim import maxsim_scores
from laterank.kernels.metrics import mean_metric, ndcg_at_k, recall_at_k
from laterank.kernels.rng import make_anchors
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


@pytest.fixture(scope="module")
def stages(scifact) -> dict:
    """Every stage, built straight from smve-lab's memory-mapped arrays (no index copy)."""
    n = len(scifact.doc_ids)
    dense = DenseStage(scifact.d_dense)
    lexical = LexicalStage(*LexicalStage.arrays(scifact.d_lexical).values(), n_docs=n)
    stages = {
        "dense": dense,
        "lexical": lexical,
        "hybrid": HybridStage(dense, lexical, HYBRID_ALPHA),
        "maxsim": MaxSimReranker(scifact.d_flat, scifact.d_offsets),
    }
    if scifact.d_texts is not None:
        a = BM25Stage.arrays(scifact.d_texts)  # smve-lab's joined title + ". " + text
        terms = a["bm25/terms.npy"].tobytes().decode().split("\n")
        stages["bm25"] = BM25Stage(terms, *list(a.values())[1:], n_docs=n)
    return stages


# (pipeline, nDCG@10, Recall@100) from results/cross_dataset/summary.md. A `> maxsim@100` run
# has its first stage's Recall@100: the rerank reorders the same 100 documents.
PIPELINES = [
    ("dense > maxsim@100", 0.6966, 0.8970),
    ("lexical", 0.6376, 0.8980),
    ("hybrid", 0.6842, 0.9237),
    ("lexical > maxsim@100", 0.7048, 0.8980),
    ("hybrid > maxsim@100", 0.6977, 0.9237),
    ("bm25", 0.6896, 0.9249),
    ("bm25 > maxsim@100", 0.7086, 0.9249),
]


@pytest.mark.slow
@pytest.mark.parametrize(("spec", "ndcg10", "recall100"), PIPELINES)
def test_pipeline(scifact, stages, spec, ndcg10, recall100):
    if spec.startswith("bm25") and scifact.q_texts is None:
        pytest.fail(
            "the SciFact texts are missing: run scripts/export_golden_texts.py once inside "
            "smve-lab (see its docstring)"
        )
    _check_run(_run(scifact, stages, spec), scifact, ndcg10, recall100)


def _run(data, stages: dict, spec: str) -> dict[str, list[str]]:
    pipeline, off = parse_pipeline(spec), data.q_offsets
    run = {}
    for i, qid in enumerate(data.query_ids):
        q = Encoded(
            data.q_flat[off[i] : off[i + 1]],
            data.q_dense[i],
            data.q_lexical[i],
            None if data.q_texts is None else data.q_texts[i],
        )
        run[qid] = [data.doc_ids[r] for r in pipeline.run(stages, q, 100).rows]
    return run


@pytest.fixture(scope="module")
def smve_stage(scifact) -> SMVEStage:
    """SMVE balanced over every SciFact document: ~5 minutes, so only SMVE tests ask for it."""
    a, _ = SMVEStage.arrays(scifact.d_flat, scifact.d_offsets, SMVE_BALANCED)
    B = make_anchors(scifact.d_flat.shape[1], SMVE_BALANCED["w"], SMVE_BALANCED["seed"])
    postings = (a[f"smve/{x}.npy"] for x in ("indptr", "rows", "weights"))
    k, reps = SMVE_BALANCED["k"], SMVE_BALANCED["reps"]
    return SMVEStage(*postings, len(scifact.doc_ids), B, k, reps, a["smve/center.npy"])


# SMVE's anchors come from laterank's own generator, so only a band can match smve-lab
# (golden-values skill): alone within its seed range, reranked within 0.6898 +- 0.01.
SMVE_BANDS = [("smve", 0.537, 0.572), ("smve > maxsim@100", 0.6798, 0.6998)]


@pytest.mark.slow
@pytest.mark.parametrize(("spec", "low", "high"), SMVE_BANDS)
def test_smve_pipeline(scifact, stages, smve_stage, spec, low, high):
    run = _run(scifact, {**stages, "smve": smve_stage}, spec)
    ndcg10 = mean_metric(run, scifact.qrels, ndcg_at_k, 10)
    assert low <= ndcg10 <= high, f"nDCG@10 = {ndcg10}"
