"""Pipeline strings parse to one canonical form, and run() hands each step the right depth."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from laterank.components.base import Candidates, Encoded
from laterank.errors import PipelineSyntaxError
from laterank.orchestration.pipeline import Pipeline, parse_pipeline

Q = Encoded(np.ones((1, 2), dtype=np.float32))


@pytest.mark.parametrize(
    ("spec", "canonical"),
    [
        ("dense", "dense"),
        ("dense > maxsim@100", "dense > maxsim@100"),
        ("  dense>maxsim@100 > maxsim@10", "dense > maxsim@100 > maxsim@10"),
        (
            "smve-balanced > maxsim@50 > cross-encoder@50",
            "smve-balanced > maxsim@50 > cross-encoder@50",
        ),
    ],
)
def test_valid_strings_print_canonically(spec, canonical):
    assert str(parse_pipeline(spec)) == canonical


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("", "first-stage name"),
        ("dense >", "reranker with a depth"),
        ("dense > maxsim", "reranker with a depth"),
        ("dense > maxsim@0", "reranker with a depth"),
        ("> maxsim@10", "first-stage name"),
        ("Dense", "first-stage name"),
        ("dense(alpha=1)", "first-stage name"),
        ("dense > maxsim@10 > maxsim@20", "grow"),
    ],
)
def test_invalid_strings_say_what_is_wrong(spec, message):
    with pytest.raises(PipelineSyntaxError, match=message):
        parse_pipeline(spec)


_names = st.from_regex(r"[a-z][a-z0-9-]{0,8}", fullmatch=True)


@given(_names, st.lists(st.tuples(_names, st.integers(1, 10**6)), max_size=4))
def test_round_trip(first, rerankers):
    rerankers.sort(key=lambda r: -r[1])  # depths may not grow
    p = Pipeline(first, tuple(rerankers))
    assert parse_pipeline(str(p)) == p


class _Recorder:
    """A fake stage: returns rows 0..n-1 (first stage) or reverses its input (reranker)."""

    def __init__(self, log, name):
        self.log, self.name = log, name

    def search(self, q, k):
        self.log.append((self.name, k))
        rows = np.arange(k, dtype=np.int64)
        return Candidates(rows, np.zeros(k, dtype=np.float32))

    def rerank(self, q, cands):
        self.log.append((self.name, len(cands.rows)))
        return Candidates(cands.rows[::-1].copy(), cands.scores[::-1].copy())


def test_run_hands_each_step_its_depth():
    log = []
    stages = {n: _Recorder(log, n) for n in ("dense", "maxsim", "other")}
    got = parse_pipeline("dense > maxsim@100 > other@10").run(stages, Q, k=5)
    assert log == [("dense", 100), ("maxsim", 100), ("other", 10)]
    # maxsim reversed 0..99, so `other` saw 99..90 and reversed it to 90..99; then cut to 5.
    assert got.rows.tolist() == [90, 91, 92, 93, 94]


def test_run_without_rerankers_asks_for_k():
    log = []
    parse_pipeline("dense").run({"dense": _Recorder(log, "dense")}, Q, k=7)
    assert log == [("dense", 7)]


def test_run_errors():
    stages = {"dense": _Recorder([], "dense"), "maxsim": _Recorder([], "maxsim")}
    p = parse_pipeline("dense > maxsim@100")
    with pytest.raises(ValueError, match="at least 1"):
        p.run(stages, Q, k=0)
    with pytest.raises(ValueError, match="k=101"):
        p.run(stages, Q, k=101)
    with pytest.raises(ValueError, match=r"\['bm25'\]"):
        parse_pipeline("bm25 > maxsim@100").run(stages, Q, k=10)


@pytest.mark.parametrize(
    "args",
    [
        ("dense > maxsim@5",),
        ("dense", (("maxsim", 10), ("maxsim", 20))),
        ("dense", (("maxsim", 0),)),
        ("dense", (("maxsim", 1.5),)),
        ("dense", ("maxsim",)),
    ],
)
def test_hand_made_pipelines_are_checked_too(args):
    with pytest.raises(PipelineSyntaxError):
        Pipeline(*args)


def test_k_must_be_an_integer():
    with pytest.raises(TypeError):
        parse_pipeline("dense").run({"dense": _Recorder([], "dense")}, Q, k=2.5)
