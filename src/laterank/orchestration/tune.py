"""Tune without labels: pick the pipeline that best agrees with exhaustive MaxSim in a budget.

Design doc §8. Queries are random 8-24-word spans cut out of documents (`span_queries`). The
answer key for each one is exhaustive MaxSim's top 10, with the span's own document removed so
it can't trivially find itself. A candidate's agreement is the overlap of its top 10 (source
removed too) with the key; for `x > maxsim@d` that equals the fraction of the key inside x's
top d, because the MaxSim rerank is exact.

Only the Pareto frontier is kept (FAISS's operating points): no kept point is beaten by another
that is both at least as fast and at least as good. The best point within the budget is always
on it. `tune` is the engine behind `Index.tune`; like `evaluate` it gets the index as a callable.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from laterank.components.base import Candidates, Encoded
from laterank.kernels.rng import sample_indices, span_bits
from laterank.orchestration.evaluate import _timed_runs
from laterank.orchestration.pipeline import Pipeline

DEPTHS = (20, 50, 100, 200, 500, 1000)  # rerank depths tried; only those < n_docs
SPAN_WORDS = (8, 24)  # span length in words, both ends included
KEY_K = 10  # agreement = |top 10 ∩ exhaustive top 10| / 10
# A first stage every index has: the token store's MaxSimReranker, scoring every document. Like
# `maxsim` it is not written to manifest.stages.
EXHAUSTIVE = "exhaustive"


@dataclass(frozen=True)
class Budget:
    latency_ms: float  # p50 per query, encoding not included; finite and > 0

    def __post_init__(self) -> None:
        ms = self.latency_ms
        # numpy floats are Real; bool is too, so it's refused by name. A huge int compares
        # exactly (no float overflow), so 10**400 is a valid "no limit".
        if not (isinstance(ms, numbers.Real) and not isinstance(ms, bool) and 0 < ms < math.inf):
            raise ValueError(f"latency_ms must be a finite number > 0, got {ms!r}")


@dataclass(frozen=True)
class Point:
    pipeline: str  # canonical string
    agreement: float  # mean over queries, in [0, 1]
    p50_ms: float  # median over queries of one timed search each


@dataclass(frozen=True)
class TuneReport:
    best: str | None  # highest agreement with p50_ms <= budget (ties: faster); None if none fits
    frontier: tuple[Point, ...]  # Pareto frontier, fastest first, agreement strictly rising
    n_queries: int
    budget: Budget

    def table(self) -> str:
        """One line per frontier point (pipeline, agreement, p50 ms), `best` marked with '*'."""
        w = max(len("pipeline"), *(len(p.pipeline) for p in self.frontier))
        lines = [f"  {'pipeline':<{w}}  {'agreement':>9}  {'p50 ms':>8}"]
        lines += [
            f"{'*' if p.pipeline == self.best else ' '} {p.pipeline:<{w}}  "
            f"{p.agreement:>9.4f}  {p.p50_ms:>8.1f}"
            for p in self.frontier
        ]
        return "\n".join(lines)


def span_queries(texts: Sequence[str], n: int, seed: int = 0) -> list[tuple[int, str]]:
    """(source row, span) pairs: min(n, eligible) documents with >= 8 whitespace words, chosen
    with `rng.sample_indices`; length and start drawn from `rng.span_bits`. Sorted by row."""
    lo, hi = SPAN_WORDS
    words = [t.split() for t in texts]
    eligible = [i for i, w in enumerate(words) if len(w) >= lo]
    m = min(n, len(eligible))
    chosen = [eligible[j] for j in sample_indices(len(eligible), m, seed)]
    r = [int(x) for x in span_bits(seed, 2 * m)]  # modulo bias: range / 2**64
    spans = []
    for j, row in enumerate(chosen):
        w = words[row]
        length = lo + r[2 * j] % (min(hi, len(w)) - lo + 1)
        start = r[2 * j + 1] % (len(w) - length + 1)
        spans.append((row, " ".join(w[start : start + length])))
    return spans


def candidates(first_stages: Iterable[str], n_docs: int) -> list[Pipeline]:
    """[exhaustive] + for each name in sorted order: name, then name > maxsim@d for d in DEPTHS
    with d < n_docs. The list `Index.tune` measures."""
    pipelines = [Pipeline(EXHAUSTIVE)]
    for name in sorted(first_stages):
        pipelines += [Pipeline(name)]
        pipelines += [Pipeline(name, (("maxsim", d),)) for d in DEPTHS if d < n_docs]
    return pipelines


def tune(
    search: Callable[[Pipeline, Encoded, int], Candidates],
    queries: Sequence[tuple[int, Encoded]],
    pipelines: Sequence[Pipeline],
    budget: Budget,
) -> TuneReport:
    """Measure every pipeline and keep the frontier (see `Index.tune`).

    `queries` are (source row, encoded span) pairs. `pipelines[0]` must be `exhaustive`: its
    results are the answer key, and it is a candidate like the others (agreement 1.0).
    """
    sources, encoded = zip(*queries, strict=True)
    points, key = [], None
    for p in pipelines:
        results, times = _timed_runs(search, p, encoded, KEY_K + 1, 1)
        # the source dropped, then the top 10
        tops = [
            set([int(r) for r in c.rows if r != s][:KEY_K])
            for c, s in zip(results, sources, strict=True)
        ]
        key = key or tops
        overlap = [len(t & k) / KEY_K for t, k in zip(tops, key, strict=True)]
        points.append(Point(str(p), float(np.mean(overlap)), float(np.median(times)) * 1000))

    frontier: list[Point] = []
    for pt in sorted(points, key=lambda pt: (pt.p50_ms, -pt.agreement)):
        if not frontier or pt.agreement > frontier[-1].agreement:
            frontier.append(pt)
    fits = [pt for pt in frontier if pt.p50_ms <= budget.latency_ms]
    return TuneReport(fits[-1].pipeline if fits else None, tuple(frontier), len(queries), budget)
