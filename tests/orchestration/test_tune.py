"""The tune engine: span sampling, agreement with the answer key, the frontier and the pick."""

import numpy as np
import pytest

from laterank.components.base import Candidates, Encoded
from laterank.orchestration import tune as tune_mod
from laterank.orchestration.pipeline import Pipeline
from laterank.orchestration.tune import Budget, span_queries, tune


def _texts():
    lengths = [3, 7, 8, 9, 15, 24, 25, 60, 0, 100, 12, 8]
    return [" ".join(f"w{i}_{j}" for j in range(n)) for i, n in enumerate(lengths)]


def test_span_queries_are_contiguous_spans_of_long_enough_documents():
    texts = _texts()
    eligible = [i for i, t in enumerate(texts) if len(t.split()) >= 8]
    spans = span_queries(texts, 5, seed=3)

    assert spans == span_queries(texts, 5, seed=3)
    assert spans != span_queries(texts, 5, seed=4)
    assert len(spans) == 5 and [r for r, _ in spans] == sorted(r for r, _ in spans)
    for row, span in spans:
        assert row in eligible
        words, part = texts[row].split(), span.split()
        assert 8 <= len(part) <= 24
        assert any(words[s : s + len(part)] == part for s in range(len(words)))

    assert [r for r, _ in span_queries(texts, 100)] == eligible  # n above eligible: all of them
    lengths = {len(span.split()) for _, span in span_queries([texts[9]] * 400, 400)}
    assert lengths == set(range(8, 25))  # both ends of 8-24 are reached


def _query():
    return Encoded(np.ones((1, 4), dtype=np.float32))


ROWS = {  # what each fake first stage returns (11 rows); every query's source is row 10
    "exhaustive": [*range(11)],  # key = rows 0..9 once the source is dropped
    "fast": [*range(20, 31)],  # agreement 0
    "tie": [0, 1, 2, *range(20, 28)],  # 0.3 at mid's speed, listed first: dominated by mid
    "mid": [0, 1, 2, 3, 4, *range(20, 26)],  # 0.5
    "same": [0, 1, 2, 3, 4, *range(20, 26)],  # 0.5 at mid's speed: not a new point
    "slowbad": [10, 0, 1, *range(20, 28)],  # 0.2 (source dropped), slower than mid: dominated
}
MS = {"exhaustive": 50.0, "fast": 1.0, "tie": 5.0, "mid": 5.0, "same": 5.0, "slowbad": 10.0}


@pytest.fixture
def fixed_times(monkeypatch):
    real = tune_mod._timed_runs

    def timed(search, p, queries, k, repeats):
        results, _ = real(search, p, queries, k, repeats)
        return results, np.full(len(queries), MS[p.first] / 1000)

    monkeypatch.setattr(tune_mod, "_timed_runs", timed)


def _search(p, q, k):
    rows = np.array(ROWS[p.first][:k], dtype=np.int64)
    return Candidates(rows, np.zeros(len(rows), dtype=np.float32))


@pytest.mark.parametrize(
    ("ms", "best"), [(6, "mid"), (100, "exhaustive"), (1, "fast"), (0.5, None)]
)
def test_frontier_and_best(fixed_times, ms, best):
    queries = [(10, _query()) for _ in range(4)]
    report = tune(_search, queries, [Pipeline(n) for n in ROWS], Budget(ms))

    assert [(p.pipeline, p.score, p.p50_ms) for p in report.frontier] == [
        ("fast", 0.0, 1.0),
        ("mid", 0.5, 5.0),
        ("exhaustive", 1.0, 50.0),
    ]
    assert report.best == best and report.n_queries == 4 and report.budget == Budget(ms)
    assert report.metric == "agreement"
    lines = report.table().splitlines()
    assert len(lines) == 4 and lines[0].split() == ["pipeline", "agreement", "p50", "ms"]
    assert [line.startswith("*") for line in lines[1:]] == [
        p.pipeline == best for p in report.frontier
    ]


@pytest.mark.parametrize("ms", [0, -1, float("nan"), float("inf"), True, "5", None])
def test_budget_refuses_bad_latency(ms):
    with pytest.raises(ValueError, match="latency_ms"):
        Budget(ms)
    for ok in (20, 0.5, np.float64(20), np.float32(5), 10**400):
        Budget(ok)


def test_candidates_is_the_list_index_tune_measures():
    got = [str(p) for p in tune_mod.candidates({"lexical", "dense", "bm25"}, n_docs=150)]
    expected = ["exhaustive"]
    for name in ("bm25", "dense", "lexical"):
        expected += [name, f"{name} > maxsim@20", f"{name} > maxsim@50", f"{name} > maxsim@100"]
    assert got == expected
