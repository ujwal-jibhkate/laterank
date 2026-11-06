---
id: M2-3
title: Validate label-free tune against real labels
status: done
approved: 2026-11-05
---

# M2-3: Validate label-free tune against real labels

## Goal

Answer design §8's open question with numbers: on SciFact, NFCorpus and ArguAna, does `tune`
(no labels) pick a pipeline whose real nDCG@10 is within 0.01 of the pipeline real labels would
pick under the same budget? After this checkpoint a script answers it per dataset, the answer is
written down in `docs/dev/tune_validation.md`, and `evaluate` can score ArguAna correctly
(`ignore_identical_ids`, deferred from M2-1).

## Out of scope

- Acting on a "no": design §8 says then ship tune-with-labels as the recommended mode. If a
  dataset fails, that is the next spec, decided with the owner; this one only measures.
- Running the validation in CI or `pytest -m slow`: it needs ~7 GB of embeddings, BGE-M3 and
  (estimate) tens of minutes per dataset. It is a script, run by hand.
- Skipped: a `tune(qrels=...)` API. `evaluate` + "best nDCG within budget" already is
  tune-with-labels; add the API if the validation says to recommend it.
- Skipped: committing NFCorpus/ArguAna qrels as test fixtures. They are exported next to the
  embeddings, like the texts; add fixtures if a test ever needs them.
- Skipped: other seeds, other `n_queries`. The defaults are what users get; add a sweep if the
  result is borderline (a gap within 0.005 of 0.01).

## Interfaces

```python
# orchestration/index.py: Index.evaluate gains one keyword (engine `evaluate` too)
def evaluate(
    self, queries, qrels, pipelines, *, repeats: int = 3, ignore_identical_ids: bool = False
) -> Evaluation:
    """ignore_identical_ids: drop a hit whose doc id equals its query id (BEIR's rule; ArguAna's
    queries are corpus documents). Each search then asks for min(101, last depth) and keeps the
    first 100 that aren't the query itself. Default False: behaviour unchanged."""


# orchestration/tune.py: the candidate list, now shared by Index.tune and the script
def candidates(first_stages: Iterable[str], n_docs: int) -> list[Pipeline]:
    """[exhaustive] + for each name in sorted order: name, then name > maxsim@d for d in DEPTHS
    with d < n_docs. Exactly the list Index.tune builds today."""
```

```text
scripts/export_golden_texts.py   also writes <name>_bgem3/qrels_test.json ({qid: {doc_id: rel}},
                                 rel > 0 as smve-lab's load_qrels); run with: scifact nfcorpus arguana
scripts/validate_tune.py --dataset {scifact,nfcorpus,arguana} [--n-queries 500] [--seed 0]
    needs LATERANK_GOLDEN_DIR, `uv sync --extra bge-m3`, HF_HUB_OFFLINE=1
    1. builds dense, lexical, hybrid, bm25, smve (balanced), muvera (default) from smve-lab's memmaps
    2. evaluate(): every candidate on the real test queries (smve-lab's stored query vectors),
       repeats=1, ignore_identical_ids for ArguAna  -> nDCG@10, R@100, real p50
    3. tune engine: 500 spans encoded with BGEM3Encoder -> frontier (agreement, span p50)
    4. for each budget B in {every frontier p50} + {no limit}:
         tune pick  = report.best for Budget(B)
         label pick = highest nDCG@10 among all candidates with real p50 <= B
         gap        = nDCG(label pick) - nDCG(tune pick)
    5. prints and appends a section to docs/dev/tune_validation.md: both tables, the gaps, the
       verdict "holds" (every gap <= 0.01) or "fails at B = ...", and the run time
```

## Decisions

| decision | why | differs from design doc? |
|---|---|---|
| a script over smve-lab's memmaps, not `Index.tune` on a built index | a built index copies 3.8–7 GB of tokens per dataset (M2-1's reason); the script calls the same engines (`tune`, `evaluate`, `candidates`) so it measures the same code | no |
| budgets = tune's own frontier p50s plus "no limit" | every point tune could ever pick gets checked, and no machine-specific ms list is invented | no |
| label side uses real-query p50, tune side span p50 | that is what each mode would see in practice; span vs real latency is part of the question (ArguAna's queries are whole arguments) | no |
| pass = every gap <= 0.01 on that dataset | §8's wording, applied to every budget, not one | no |
| sanity gate: the script stops if `exhaustive` nDCG@10 is off golden by > 0.002 (0.6991 / 0.3441 / 0.4841) | proves the new datasets, qrels and the ArguAna rule load right before any conclusion | no |
| `ignore_identical_ids` asks for one extra hit, then drops the self-hit | matches smve-lab (mask self, then top 100) for single stages, and tune's own rule | no |

**Expect ArguAna to fail.** Its golden `exhaustive` (0.4841) is below `dense` alone (0.5266).
Tune scores agreement with exhaustive, so with a loose budget it picks near-exhaustive while labels
pick `dense`: a gap of about 0.04. That is the finding §8 asked about, not a bug.

## Acceptance tests

- [x] AT1: `evaluate(..., ignore_identical_ids=True)` never returns a hit whose id equals its query id, still scores 100 hits when the index has > 100 docs, and the default (False) gives the same scores as before (existing tests unchanged).
- [x] AT2: `candidates()` equals the list in its docstring on a 3-stage example; `Index.tune` uses it and the existing tune tests pass unchanged.
- [x] AT3: `validate_tune.py` passes its sanity gate on all three datasets (`exhaustive` nDCG@10 within ±0.002 of 0.6991 / 0.3441 / 0.4841).
- [x] AT4: `docs/dev/tune_validation.md` has, per dataset: the candidate table (nDCG@10, R@100, real p50), tune's frontier (agreement, span p50), the gap per budget, a verdict line, and the run time.
- [x] AT5: the outcome is in the progress log and the `golden-values` skill (NFCorpus/ArguAna laterank `exhaustive` numbers), and the owner has decided the follow-up for any failing dataset.
- [x] AT6: `uv run pytest -q`, `ruff check`, `ruff format --check` and `tests/test_layers.py` pass.
- [x] AT7: reviewer agent verdict `merge`.

## Tasks

| # | task | files it owns | depends on | parallel? |
|---|---|---|---|---|
| 1 | `ignore_identical_ids`; `candidates()` and `Index.tune` using it; tests AT1, AT2 | `orchestration/{evaluate,tune,index}.py`, `tests/orchestration/{test_evaluate,test_tune}.py` | - | no |
| 2 | qrels export; `validate_tune.py`; smoke run with `--n-queries 20` on SciFact | `scripts/export_golden_texts.py`, `scripts/validate_tune.py` | 1 | no |
| 3 | full runs on 3 datasets (owner runs or OKs the time), write results; AT3–AT5 | `docs/dev/tune_validation.md`, golden-values skill, `docs/dev/progress.md` | 2 | no |

## Changes

Dated lines for anything changed after approval: `- 2026-10-06: <what> - <why>`.

- 2026-11-05: known difference (reviewer finding 3): with `ignore_identical_ids`, a rerank
  `x > maxsim@d` asks for d (it can't ask past its depth), so when the query document is in x's
  top d only d-1 others are scored; smve-lab masked it before its top 100. Seen on ArguAna only:
  `bm25 > maxsim@100` R@100 0.9716 vs `bm25` 0.9730. Test added: the ask is capped at the depth.
- 2026-11-05: recorded (reviewer finding 5): budgets where no candidate's real p50 fits ("none
  fits") are left out of the verdict, and tune's pick is judged by its span p50, so it can be
  slower than B on real queries (NFCorpus, B = 3.3 ms: `hybrid > maxsim@20` real p50 4.2 ms).
  This favours tune; every dataset fails anyway, so the conclusion stands.
- 2026-11-05: ran all three datasets: fails on all (`docs/dev/tune_validation.md`); owner chose
  "tune with labels as the recommended mode" as M2-4.
