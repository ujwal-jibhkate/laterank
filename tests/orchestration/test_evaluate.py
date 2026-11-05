"""Index.evaluate must score what search returns, and refuse bad inputs before any search."""

import math

import numpy as np
import pytest

from laterank.components.base import Encoded
from laterank.errors import PipelineSyntaxError
from laterank.kernels.metrics import mean_metric, ndcg_at_k, recall_at_k
from laterank.orchestration.index import Index
from laterank.storage.manifest import EncoderInfo

DIM = 8
ENCODER = EncoderInfo(name="precomputed", hf_id=None, revision=None, dim=DIM)
PIPELINES = ["exhaustive", "dense > maxsim@20"]


def _setup(tmp_path, n=40, n_queries=6):
    rng = np.random.default_rng(0)
    lengths = rng.integers(1, 6, size=n)
    flat = rng.standard_normal((lengths.sum(), DIM)).astype(np.float32)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    docs = [{"id": f"d{i}", "text": f"t{i}"} for i in range(n)]
    dense = rng.standard_normal((n, 4)).astype(np.float32)
    index = Index.create(tmp_path / "ix", docs, (flat, offsets), encoder=ENCODER, dense=dense)
    queries = {
        f"q{i}": Encoded(rng.standard_normal((3, DIM)).astype(np.float32), rng.standard_normal(4))
        for i in range(n_queries)
    }
    qrels = {qid: {f"d{j}": 1.0 for j in rng.choice(n, size=3, replace=False)} for qid in queries}
    return index, queries, qrels


def test_evaluate_matches_mean_metric_over_search(tmp_path):
    index, queries, qrels = _setup(tmp_path)
    queries["extra"] = queries["q0"]  # not judged: ignored
    ev = index.evaluate(queries, qrels, PIPELINES, repeats=2)

    assert ev.n_queries == 6
    assert [s.pipeline for s in ev.scores] == PIPELINES
    for s, k in zip(ev.scores, (100, 20), strict=True):  # k = min(100, last depth)
        run = {q: [h.id for h in index.search(queries[q], k, pipeline=s.pipeline)] for q in qrels}
        assert s.ndcg_at_10 == mean_metric(run, qrels, ndcg_at_k, 10)
        assert s.recall_at_100 == mean_metric(run, qrels, recall_at_k, 100)
        assert math.isfinite(s.p50_ms) and s.p50_ms > 0
    # "dense > maxsim@20" returns 20 documents, so it can't beat exhaustive's 40-doc recall
    assert ev.scores[1].recall_at_100 <= ev.scores[0].recall_at_100

    lines = ev.table().splitlines()
    assert len(lines) == 3 and lines[0].split()[:2] == ["pipeline", "nDCG@10"]
    assert lines[1].startswith("exhaustive") and lines[2].startswith("dense > maxsim@20")


def test_evaluate_refuses_bad_inputs_before_any_search(tmp_path):
    index, queries, qrels = _setup(tmp_path)
    calls = []
    real = index._stages["exhaustive"]

    class Counting:
        def search(self, q, k):
            calls.append(1)
            return real.search(q, k)

    index._stages["exhaustive"] = Counting()

    cases = [
        (PipelineSyntaxError, dict(pipelines=["exhaustive", "dense >"])),
        (ValueError, dict(pipelines=["exhaustive", "bm25 > maxsim@10"])),  # stage missing
        (ValueError, dict(qrels={})),
        (ValueError, dict(pipelines=[])),
        (ValueError, dict(qrels={**qrels, "nope": {"d1": 1.0}})),  # judged, not encoded
        (ValueError, dict(queries={**queries, "q1": "a string"})),
        (ValueError, dict(queries={**queries, "q1": Encoded(np.ones((2, DIM + 1)))})),  # dim
        (ValueError, dict(repeats=0)),
    ]
    for error, change in cases:
        args = dict(queries=queries, qrels=qrels, pipelines=PIPELINES, repeats=1) | change
        with pytest.raises(error):
            index.evaluate(**args)
        assert calls == [], change
    index.evaluate(queries, qrels, ["exhaustive"], repeats=1)  # the counter does see a good call
    assert len(calls) == len(qrels)
