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


def test_ignore_identical_ids_drops_the_query_itself(tmp_path):
    index, _, _ = _setup(tmp_path, n=150)
    tokens, offsets = index._stages["exhaustive"].flat, index._stages["exhaustive"].offsets
    # each query is a document's own tokens, under that document's id: it finds itself first
    qids = ["d3", "d50", "d120"]
    queries = {
        q: Encoded(np.asarray(tokens[offsets[i] : offsets[i + 1]], np.float32), np.ones(4))
        for q, i in zip(qids, (3, 50, 120), strict=True)
    }
    qrels = {q: {q: 1.0, "d0": 1.0} for q in qids}
    calls = []
    real = index._stages["exhaustive"]

    class Recording:  # keeps every run's rows, to see what evaluate scored
        def search(self, q, k):
            calls.append(real.search(q, k))
            return calls[-1]

    index._stages["exhaustive"] = Recording()
    kept = index.evaluate(queries, qrels, ["exhaustive"], repeats=1)
    assert all(c.rows[0] == i for c, i in zip(calls, (3, 50, 120), strict=True))
    assert kept.scores[0].ndcg_at_10 > 0.5  # the self-hit counts when kept

    calls.clear()
    dropped = index.evaluate(queries, qrels, ["exhaustive"], repeats=1, ignore_identical_ids=True)
    assert all(len(c.rows) == 101 for c in calls)  # one spare hit, so 100 are still scored
    run = {
        q: [index._docs[r]["id"] for r in c.rows if index._docs[r]["id"] != q][:100]
        for q, c in zip(qids, calls, strict=True)
    }
    assert all(q not in run[q] and len(run[q]) == 100 for q in qids)
    assert dropped.scores[0].ndcg_at_10 == mean_metric(run, qrels, ndcg_at_k, 10)
    assert dropped.scores[0].recall_at_100 == mean_metric(run, qrels, recall_at_k, 100)
    with pytest.raises(ValueError):
        index.evaluate(queries, qrels, ["exhaustive"], ignore_identical_ids=1)


def test_ignore_identical_ids_never_asks_past_the_rerank_depth(tmp_path):
    index, _, _ = _setup(tmp_path, n=150)
    real = index._stages["exhaustive"]
    q = Encoded(np.asarray(real.flat[real.offsets[7] : real.offsets[8]], np.float32), np.ones(4))
    asked = []

    class Recording:
        def search(self, q, k):
            asked.append(k)
            return real.search(q, k)

    index._stages["exhaustive"] = Recording()
    index.evaluate(
        {"d7": q},
        {"d7": {"d0": 1.0}},
        ["exhaustive > maxsim@20"],
        repeats=1,
        ignore_identical_ids=True,
    )
    # evaluate must cap its ask at the depth: Pipeline.run raises ValueError for k=21 > 20
    assert asked == [20]
