---
id: M2-2
title: Label-free tune and set_default
status: done
approved: 2026-11-05
---

# M2-2: Label-free tune and set_default

## Goal

After this checkpoint a user with no relevance labels can ask the index to pick its pipeline:
`report = index.tune(Budget(latency_ms=20))`, then `index.set_default(report.best)`. Tune cuts
random 8–24-word spans out of documents as queries, uses exhaustive MaxSim's top 10 as the answer
key, and measures each candidate pipeline's agreement with that key and its median latency
(design doc §8). M2-3 then checks on 3 datasets whether the pick matches what real labels pick.

## Out of scope

- Validating tune against real labels (SciFact, NFCorpus, ArguAna): M2-3.
- Skipped: tuning on the user's own query texts. Add it when M2-3 shows spans mislead, or a
  user asks (design §8 lists it as an option).
- Skipped: a memory (MB) budget. Every built stage is already on disk, so tune can't save
  space; add it when tune also decides which stages to build.
- Skipped: tuning stage settings (SMVE `w`/`k`, MUVERA reps). Only `balanced` / `MUVERA_DEFAULT`
  can be built today; add when `Index.create` takes other settings.
- Skipped: reporting dominated points. Only the Pareto frontier is kept (FAISS's operating
  points); add when someone needs the full grid.
- Skipped: best-of-`repeats` timing. Tune times each query once (see Decisions).

## Interfaces

```python
# orchestration/tune.py (new)
DEPTHS = (20, 50, 100, 200, 500, 1000)  # rerank depths tried; only those < n_docs
SPAN_WORDS = (8, 24)  # span length in words, both ends included
KEY_K = 10  # agreement = |top 10 ∩ exhaustive top 10| / 10


@dataclass(frozen=True)
class Budget:
    latency_ms: float  # p50 per query, encoding not included; finite and > 0 (else ValueError)


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


def span_queries(texts: Sequence[str], n: int, seed: int = 0) -> list[tuple[int, str]]:
    """(source row, span) pairs: min(n, eligible) documents with >= 8 whitespace words, chosen
    with `rng.sample_indices`; length and start drawn from `rng.splitmix64`. Sorted by row."""


# orchestration/index.py
class Index:
    def tune(
        self, budget: Budget, *, encoder: Encoder | None = None, n_queries: int = 500, seed: int = 0
    ) -> TuneReport:
        """Pick the pipeline that best agrees with exhaustive MaxSim within the budget.

        Candidates: `exhaustive`, every built first stage alone, and every built first stage
        `> maxsim@d` for d in DEPTHS below n_docs. The span's source document is dropped from
        every result list (each search asks for 11, keeps the first 10 that aren't the source).
        `encoder` defaults to `load_encoder(self.manifest.encoder)`.

        Raises:
            ValueError: n_queries not an int >= 1, seed out of range, fewer than 11 documents,
                no document has 8 words, `encoder.info != self.manifest.encoder`, or the index
                has no loadable encoder and none was given. All checked before any search.
        """

    def set_default(self, pipeline: str | Pipeline) -> None:
        """Make `pipeline` the default: checked like `search` checks it, then written to the
        manifest atomically (`write_manifest`) and used by this Index at once.

        Raises:
            PipelineSyntaxError: the string doesn't parse.
            ValueError: the pipeline uses a stage this index lacks.
        """
```

`orchestration/evaluate.py` gains a private `_timed_runs(search, p, queries, k, repeats)` that
both `evaluate` and tune call; `evaluate`'s behaviour doesn't change.

## Decisions

| decision | why | differs from design doc? |
|---|---|---|
| agreement = overlap of the final top 10 with exhaustive's top 10, the same rule for every candidate | for `x > maxsim@d` it equals "fraction of the key in x's top d" (§8's rule), because the rerank is exact; one rule also covers `x` alone | no |
| the source document is dropped from the key and from every result, then the top 10 are taken | §8: a span must not find its own document. Asking for 11 is enough; depths start at 20 so every pipeline can return 11 | no |
| keep only the Pareto frontier; `best` = its last point within budget | FAISS operating points (researcher, 11-04). The best in-budget point is always on the frontier | no |
| `exhaustive` is a candidate | if the budget allows exact search, it is the right answer (agreement 1.0) | no |
| default 500 span queries, each timed once | about 0.01 standard error on agreement, M2-3's tolerance. FAISS asks ≥ 1000, but tune runs up to 43 pipelines on SciFact (~0.5 s per query in all, estimate, not measured), so 1000 × 3 repeats would be ~25 min. A median over 500 single runs barely moves from the warmup | yes: FAISS's 1000; evaluate's best of 3 |
| `Budget` is a frozen dataclass with one field, validated in `__post_init__` | the design doc's API (`Budget(latency_ms=20)`); a memory field can join it later without a new argument | no |
| `encoder` argument, defaulting to the manifest's model | tune makes text queries, so it must encode them. Indexes built from precomputed vectors (tests, golden) pass one | no |
| span sampling uses the portable RNG; ints are `splitmix64 % range` | same spans on every machine for a seed (never-cut rule). The modulo bias is range / 2⁶⁴, far below anything measurable | no |

## Acceptance tests

- [x] AT1: `span_queries` is the same for the same seed and differs for another; every span is 8–24 contiguous words of its source text; documents under 8 words are never picked; `n` above the eligible count returns all eligible.
- [x] AT2: on a small index (FakeEncoder), tune's agreement for each candidate equals a brute-force reference (`maxsim_scores` over all docs, source removed, top 10 vs each pipeline's top 10 with the source removed). `exhaustive` scores 1.0, and `x > maxsim@d` never scores less than `x > maxsim@d'` for d > d'.
- [x] AT3: `frontier` is sorted by p50 with strictly rising agreement and no point dominated; `best` is the highest-agreement point within budget; a budget below every p50 gives `best is None`; `table()` has a header plus one line per point.
- [x] AT4: each error in `tune`'s Raises list is raised before any search runs (counting stage), and `Budget(0)`, `Budget(nan)`, `Budget(True)` raise `ValueError`.
- [x] AT5: `set_default` changes `default_pipeline` for this Index and after `Index.open`; a bad pipeline raises and leaves `manifest.json` byte-for-byte unchanged.
- [x] AT6: existing `evaluate` tests pass unchanged after the `_timed_runs` refactor; `uv run pytest -q`, `ruff check`, `ruff format --check` and `tests/test_layers.py` pass.
- [x] AT7: reviewer agent verdict `merge`.

## Tasks

| # | task | files it owns | depends on | parallel? |
|---|---|---|---|---|
| 1 | `_timed_runs` refactor; `Budget`, `Point`, `TuneReport`, `span_queries`, the tune engine; tests AT1, AT3, Budget part of AT4 | `orchestration/evaluate.py`, `orchestration/tune.py`, `tests/orchestration/test_tune.py` | - | no |
| 2 | `Index.tune`, `Index.set_default`; tests AT2, AT4, AT5 | `orchestration/index.py`, `tests/orchestration/test_index.py` | 1 | no |

## Changes

Dated lines for anything changed after approval: `- 2026-10-06: <what> - <why>`.

- 2026-11-05: `Index.tune` also raises `ValueError` when `budget` isn't a `Budget` (e.g. a
  bare `20`) - validation at the public boundary; otherwise it fails as an `AttributeError`
  deep in the engine. `search`, `evaluate` and `set_default` share a private `Index._pipeline`
  (parse + check), so the check is written once.
- 2026-11-05 (reviewer round 1, owner OK "fix all"): new `kernels/rng.py::span_bits(seed, n)
  -> (n,) uint64`, drawn from the stream derived from (seed, "spans"); `span_queries` uses it
  instead of the raw `splitmix64(seed)` stream - `rng.py`'s rule gives the raw stream to
  anchors only (SMVE's seed is also 0). `Budget` accepts any real number but `bool` (numpy
  floats included); a huge int like `10**400` is valid ("no limit": it compares exactly, no
  overflow). Task 1 also owns
  `kernels/rng.py` and `tests/kernels/test_rng.py`.
