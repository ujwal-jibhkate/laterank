# laterank progress log

This file is the project's memory between Claude Code sessions. `/start-session` reads it, and
`/wrap-up` updates it. Newest entries go first.

## Current state

- Milestones M0-M4 are complete. `laterank 0.1.0b1` is on PyPI (tag `v0.1.0b1` = `a820d5a`). No spec approved. M5 is planned (checklist below).
- Works: `pip install "laterank[colbert]"` from PyPI (no `--pre`; quick-start checked in a new venv); `from laterank import Index, Budget, ColBERTEncoder, BGEM3Encoder, load_encoder`; `laterank build`, `search`, `evaluate`, `tune` (`api/cli.py`); `load_beir`/`download_beir`; `Index.build`, `evaluate`, `tune`, `set_default`; first stages `dense`, `lexical`, `hybrid`, `bm25`, `smve`, `muvera`, `exhaustive`, alone or `> maxsim@N`; `ColBERTEncoder` refuses non-ColBERT checkpoints.
- Tune with labels is the recommended mode (label-free fails the 0.01 test on all 3 datasets). Recommended ColBERT model: **mxbai-edge-colbert-v0-32m** (not hard-coded: `--hf-id`).
- Nothing is known broken. 484 passed, 31 golden skip without `LATERANK_GOLDEN_DIR` (some also need `uv sync --extra bge-m3 --extra colbert`). A plain `uv sync` removes those extras.
- Chunking is cut from 0.1 (owner, 2026-11-08): the README tells users to split long documents before `build`.
- Known wrong on PyPI until `0.1.0b2`: the 0.1.0b1 page links to `START_HERE.md`, which is now untracked (404).

## M0 checklist

- [x] 1. Skeleton: uv + hatchling/hatch-vcs, Apache-2.0, extras declared, CI (3 OS × py3.12/3.13)
- [x] 2. Kernels: MaxSim, top-k, CSC scoring, SMVE encode, portable SplitMix64 anchors + SHA-256, metrics
- [x] 3. Storage: manifest.json (atomic write), segments, `.npy` with mmap and no pickle, docs.jsonl + offsets
- [x] 4. Golden harness: `LATERANK_GOLDEN_DIR`, SciFact qrels fixture, exhaustive 0.6991 and dense 0.6542

## M1 checklist

- [x] 1. Index + pipeline, dense first (`dense > maxsim@100` = 0.6966)
- [x] 2. Lexical + hybrid (`hybrid > maxsim@100` = 0.6977)
- [x] 3. BM25 (0.6896; `bm25 > maxsim@100` = 0.7086)
- [x] 4. SMVE (band [0.537, 0.572])
- [x] 5. MUVERA (band: alone [0.500, 0.580], `muvera > maxsim@100` 0.6888 ± 0.01)
- [x] 6. BGE-M3 encoder + `Index.build` (`2a3cdfc`; spec split: presets dropped for now)
- [x] 7. ColBERT encoder (`MultiVectorEncoder`) + the two-model measurement (`92b6c70`; answerai
  0.7384, mxbai 0.7453 with `bm25 > maxsim@100`; recommended default: mxbai)

## M2 checklist

- [x] 1. Evaluate + `exhaustive` first stage (SciFact `exhaustive` = 0.6991)
- [x] 2. Label-free `tune` + `set_default`: span queries, exhaustive top-10 as answer key, Pareto
  frontier, `Budget(latency_ms)`
- [x] 3. Validate tune on SciFact/NFCorpus/ArguAna against real labels (within 0.01 nDCG@10?):
  measured, answer **no** on all 3 (worst ArguAna, 0.042: MaxSim itself loses to `dense`)
- [x] 4. Tune with labels as the recommended mode: `Index.tune(budget, queries=, qrels=)` (owner's decision after M2-3)

## M3 checklist

- [x] 1. BEIR dataset loader + `api/` layer: `load_beir`, `download_beir` (`0a35e76`, CI green)
- [x] 2. CLI: `laterank build`, `search`, `evaluate`, `tune` (`92fe519`, CI green)
- [x] Chunking: cut from 0.1 (owner, 2026-11-08)

## M4 checklist

- [x] 1. README install guide + quick-start, top-level imports (`042ddbd`, CI green)
- [x] 2. PyPI release: `0.1.0b1` on PyPI from tag `v0.1.0b1` (`a820d5a`, CI green); Beta classifier, empty `rerank`/`pandas` extras removed
- [x] 3. `ColBERTEncoder` refuses non-ColBERT checkpoints (`4492766`, CI green)

## M5 checklist

- [ ] 1. `0.1.0b2`: test the `scipy` floor (raise it if 1.15 is broken), release so the PyPI page loses the dead `START_HERE.md` link (already fixed in the repo's README)
- [ ] 2. 0.2 design: how `add` and `delete` change an index (segments, deletes, BM25 statistics, crash safety), from `docs/dev/add_delete_research.md`. Owner chose option A (rewrite into one new segment, 2026-11-10); measure a SciFact BGE-M3 rewrite first
- [ ] 3. `Index.add` (new documents in a new segment)
- [ ] 4. `Index.delete` (and how deleted documents leave search results)
- [ ] 5. CLI `laterank add` / `laterank delete`, README, `0.2.0` release

## Ported from smve-lab

| smve-lab file | laterank file | what changed |
|---|---|---|
| `maxsim.py` (`maxsim_scores`, `subset_ragged`) | `kernels/maxsim.py` | asserts → `ValueError`; progress bar off by default; empty query/doc sets allowed |
| `maxsim.py` + `inverted_index.py` (`top_k`) | `kernels/topk.py` | returns indices, not id dicts; ties broken by lower row (argpartition alone was arbitrary at the k-th place); 1-D and 2-D |
| `inverted_index.py` | `kernels/inverted.py` | class → pure functions over `(indptr, rows, weights)`; scores returned as float32 |
| `smve.py` | `kernels/smve.py` | torch → numpy (`argpartition`, `bincount`, `np.unique`); anchors from `kernels/rng.py`; `token_mean` samples with the portable RNG and accumulates in float64 |
| `smve.make_anchors` (torch.Generator) | `kernels/rng.py` | new: SplitMix64 + Box–Muller with polynomial log/sin/cos, bit-identical on every platform; `anchor_sha256` |
| `storage.py` (save/load embeddings) | `storage/{arrays,manifest,docs,segment}.py` | new index format: manifest + segment folders, atomic writes, `allow_pickle=False`, no duplicate `.npz` copy |
| `metrics.py` (nDCG, Recall) | `kernels/metrics.py` | pandas dropped; `mean_metric` replaces `evaluate_run`; P/MRR/MAP wait for M2 |
| `scripts/evaluate_bgem3_single_vector.py` (lexical, dense + 0.3·lexical) | `components/{lexical,hybrid}.py` | scipy sparse product → postings scored by `kernels/inverted.py`; hybrid stores nothing and reuses the dense and lexical stages |
| `bm25.py` (`tokenize`, `BM25`) | `components/{bm25,stopwords}.py` | stores raw counts (tf postings + sorted terms), weights computed at open in smve-lab's float32 order (bit-identical); sklearn stopwords copied (no sklearn dependency); vocab sorted, not insertion order |
| `scripts/evaluate_smve.py` (setting w65536_k32_center) | `components/smve.py` | anchors regenerated at open and hash-checked (not stored, 256 MB); doc vectors as postings; `topk_blocks` threaded (argpartition was 89% of encode time) |
| `scripts/compare_first_stages.py` (rerank the top depth with MaxSim) | `components/maxsim.py`, `orchestration/pipeline.py` | a `Pipeline` object + string grammar; ties go to the lower global row (smve-lab kept candidate order) |
| `muvera.py` (`MuveraFDE`) | `kernels/muvera.py`, `components/muvera.py` | torch → numpy (`np.bincount` per projected column replaces `index_add_`); planes from the portable RNG (`make_planes`, new `rng.random_signs`) and **stored** in the index (6 MB), not regenerated; `center` and `fill_empty=False` dropped; an empty item gives zeros (smve-lab crashed); `fde_encode` agrees with smve-lab's `encode` to 1.9e-6 |
| `bge_m3.py` (`load_model`, `encode_texts`) | `components/{encoder,bgem3}.py` | `BGEM3Encoder` (FlagEmbedding, lazy import) behind an `Encoder` Protocol; model pinned by `snapshot_download(revision=sha)` and the sha recorded; ragged outputs → `EncodedDocs` (float16 tokens, CSR lexical); queries via `encode_query`; vectors match smve-lab's (dense cosine ≥ 0.9999999 on 20 docs) |

## Log

### 2026-11-09 (M4-2: 0.1.0b1 on PyPI)
- **Done**: drafted, approved, built and closed M4-2. `pyproject.toml` (Beta classifier, Repository
  URL, `rerank`/`pandas` extras removed, sdist allowlist), `uv.lock`, README (PyPI install line, full
  GitHub links). Owner tagged `v0.1.0b1` on `a820d5a`, uploaded with `uv publish`, pushed the tag.
  Verified: twine check, sdist 6 entries, wheel 44 files, clean-venv install, PyPI install + quick-start
  (doc `a` first, 8.559). Reviewer: "merge" (rehearsed the tag in a throwaway clone).
- **Also**: `tests/orchestration/test_index.py` score asserts use `pytest.approx(rel=1e-6)`.
  `START_HERE.md` is now gitignored (owner's local guide); new public `CONTRIBUTING.md`; README's
  "Working on laterank" became "Contributing"; CLAUDE.md's rule updated to match.
- **Decisions**: upload by hand, no TestPyPI, no publish workflow yet (add Trusted Publishing at
  0.1.0). Tag locally, push it only after the upload. Kept the `scipy>=1.15` floor and author email.
- **Shortcuts with a ceiling**: none added (`grep ponytail:` still 9).
- **Surprises**: CI failed once on macOS: Apple Accelerate rounds a float32 matmul differently by
  memory address (1 ULP between a memmap and an in-memory copy). A GitHub Actions outage cost one
  rerun. scipy 1.15 fails to load on this Mac (1.16.3 works), so the floor may be too low.
- **Token**: the account-wide PyPI token used for the first upload is deleted; future uploads use a
  token scoped to the `laterank` project (owner, 2026-11-09).
- **Next**: plan what comes after the beta: list for `0.1.0b2` the PyPI README's dead `START_HERE.md` link and the
  scipy floor, and draft the next milestone's checklist (0.2 brings add/delete documents).

### 2026-11-08 (M4-3: refuse non-ColBERT models)
- **Done**: drafted, approved, built and closed M4-3. `components/colbert.py` gains
  `_is_colbert_checkpoint` and a `ValueError` in `ColBERTEncoder.__init__` (after the download,
  before the load). Tests: `tests/components/test_colbert.py`, and a golden `laterank build` with
  real BGE-M3 in `tests/golden/test_colbert.py`. Reviewer: "merge", 2 findings, both fixed (no
  second pass after the `model_type` fix, recorded in the spec).
- **Measured**: BGE-M3 loaded twice through `MultiVectorEncoder`: mean cosine -0.09 (mxbai 1.0000).
  Golden numbers unchanged: answerai 0.7384, mxbai 0.7453, command line 0.7459.
- **Decisions**: read the checkpoint's files, because loading erases the evidence. Accept
  `modules.json` with `Dense`, no `Pooling` and an allowed `model_type`, or no `modules.json` with
  `HF_ColBERT`. `model_type` can only refuse: LightOn's PyLate models have none. The check is in the
  encoder, so the CLI, `Index.build` users and `load_encoder` all get it. `ForRetrieval` models
  (ColPali) are refused: text only.
- **Shortcuts with a ceiling**: none added (`grep ponytail:` still 9). A wrong model still downloads
  once before the refusal (BGE-M3: 4.3 GB); upgrade: a config-only `snapshot_download` first.
- **Surprises**: the library adds a random projection and logs it only at `info`. Other surprises:
  a foreign `model_type` makes it discard the declared modules (reviewer found this). Importing the
  model libraries under `capsys` leaves transformers logging to a closed stream, so the golden test
  uses `redirect_stderr`. answerai is cached by sha only (no `refs/main`). Only mxbai, answerai
  and BGE-M3 were loaded; the other models' rule results come from their config files.
- **Not done**: the 4.3 GB BGE-M3 note is in the golden test's docstring, not in `CLAUDE.md`.
- **Next**: owner pushes and confirms CI is green. Then `/spec M4-2`: start by reading
  `pyproject.toml` (classifiers, the empty `rerank`/`pandas` extras, hatch-vcs version from a tag) and
  the README's install line, and decide with the owner whether to try TestPyPI before PyPI.

### 2026-11-08 (M4-1: install guide, quick-start, top-level imports)
- **Done**: drafted, approved, built and closed M4-1 (`042ddbd`, CI green). `src/laterank/__init__.py`
  exports five names; `README.md` gained Install and Quick start, lost its "not usable" banner and
  the checkout-only `uv sync` line, and its tune example now runs as written. Reviewer: "merge", 4
  low findings fixed. All acceptance tests re-run on the venv installed from the real git URL.
- **Measured**: README's `pip install` with no cache: 51 s, 1.1 GB. `import laterank`: 0.11 s.
- **Decisions**: chunking cut from 0.1 (owner; also in CLAUDE.md), the README says to split long
  documents. Top-level exports (owner), with `load_encoder` added because it is the only way back
  to an index's pinned model. Install from the git URL until PyPI. No docs site, the README is
  enough for a beta (differs from the design doc). No new torch test: `test_every_module_imports_with_the_model_libraries_blocked` covers it.
- **Shortcuts with a ceiling**: none added (`grep ponytail:` still 9).
- **Surprises**: `Index.build` with no pipeline fails on ColBERT after encoding (its default needs
  `hybrid`), so the quick-start passes `bm25=True` and a pipeline. A wrong document key or non-string
  `id` also fails only after encoding; the README now states the rules. The 512-token cut applies
  to model vectors only, BM25 indexes the full text. `PIPESTATUS` is empty in zsh and
  `gh run list --commit` needs the full sha; both made my checks report wrongly until re-run.
- **Not done (reviewer, low)**: a second quick-start run raises `IndexExistsError`; label-free tune
  needs at least 11 documents; `api/cli.py:34,37` local imports can move to the top; install-hint
  errors say `pip install "laterank[...]"`, which only works after M4-2.
- **Next**: `/spec M4-3` (`--encoder colbert` refuses non-ColBERT models, so the beta ships with it).
  Start by reading `components/colbert.py:60` and checking what a non-ColBERT checkpoint looks like
  once loaded by `MultiVectorEncoder`. Then `/spec M4-2` (PyPI: classifier, remove the empty
  `rerank`/`pandas` extras, tag `v0.1.0b1`, swap the README's install line).

### 2026-11-08 (M3-2: command line)
- **Done**: researcher report on CLI shapes (bm25s, tantivy, Pyserini, ir_datasets, llm, colgrep).
  Drafted, approved, built and closed M3-2 (`92fe519`; `ab6215a` marks M3-1 done, whose session had
  no log entry). New `api/cli.py`, `api/datasets.py::load_corpus`, `[project.scripts]`,
  `tests/api/test_cli.py`, golden `tests/golden/test_cli.py`, README "Command line". Reviewer: round
  1 "fix first" (7 fixed), round 2 "merge" (3 notes, fixed), round 3 "merge" after the `--split` change.
- **Measured** (SciFact, mxbai at `bb13a29e`, `bm25 > maxsim@100`): nDCG@10 0.7459, Recall@100 0.9249
  (0.7453 on stored texts), query encoding p50 ~7 ms, one `laterank search` 5.3-6.8 s, nearly all
  of it startup and model load. First real ColBERT build followed by `Index.open`.
- **Decisions**: `--encoder` required on `build` only (no hard-coded default). Default pipeline
  follows the encoder (ColBERT has no `hybrid`), and a pipeline's first stage builds itself.
  `--dataset` takes a BEIR name or a folder holding `corpus.jsonl`, no `download` command (a name
  is the explicit request). `--docs` reads BEIR's row shape (`_id`). `build` reads only the corpus,
  no `--split` (owner). Query encoding is one line, not a column (same number on every row).
- **Shortcuts with a ceiling**: none added (`grep ponytail:` 9, all older). `cli.py` calls private
  `Index._pipeline` and `_check_free` to fail before the model loads; make them public if a third appears.
- **Surprises**: the researcher's "a path wins" rule would let an index folder named `scifact` hide
  the dataset `scifact`. Truthy checks made `tune --dataset "$UNSET"` silently go label-free. One
  Greek letter aborted `search` on a cp1252 stream, and Hugging Face errors ran to 7 lines. A plain
  `uv sync` removed the model extras from the venv.
- **Then**: pushed; CI green on all 6 jobs and lint (`27c288e`). AT9 (real model) has only run on macOS.
- **Next**: frame what's left with the owner: chunking (a spec, or cut from 0.1) and M4 (docs,
  `0.1.0b1`). Also open: `--encoder colbert` accepts non-ColBERT models (M1-7 code, needs its own spec).

### 2026-11-06 (M2-4: tune with labels)
- **Done**: drafted, approved, built and closed M2-4 (spec `done`, not yet committed).
  `Index.tune(budget, queries=, qrels=, ignore_identical_ids=)` runs `tune_with_labels` (new, in
  `orchestration/tune.py`: `evaluate` with best-of-3 timing, then the shared `_report` frontier
  and pick). `Point.agreement` -> `score`, `TuneReport.metric` ("agreement" or "nDCG@10"); the
  rename also touched `scripts/validate_tune.py`. README gained "Choosing a pipeline"; the
  `Index.tune` docstring carries the label-free caveat. Tests in `tests/orchestration/test_{tune,index}.py`.
  Reviewer: round 1 "fix first" (3 findings, all fixed), round 2 "merge".
- **Decisions**: one entry point (owner OK: `Index.tune`, not a method on `Evaluation`). Mixing
  modes raises (`seed=` with `qrels` is not silently ignored), so `n_queries`/`seed` default to
  `None`. Label mode times best of 3, label-free stays at 1.
- **Shortcuts with a ceiling**: none added (`grep ponytail:` still 7). Not done: no noise
  estimate on the label pick (300 queries can't tell 0.705 from 0.708); listed as skipped in the spec.
- **Surprises**: my first README/docstring said "up to 0.042" lost; the real worst gap is 0.0524
  (SciFact below 1 ms), 0.042 is ArguAna above 1 ms (reviewer caught it). My first refusal test
  patched the engine, so `evaluate`'s own errors (raised inside it) never ran; it is now two tests.
  BSD `sed -i` needs `''`, so a `sed` tick of the spec failed silently once.
- **Next**: frame M3 with the owner: the dataset loader (BEIR-style qrels/corpus to plain dicts)
  and a CLI (`build`, `evaluate`, `tune`, `search`; design doc, plus an encode-time column in
  `evaluate`). Start with the `researcher` agent on how others load BEIR data and shape a retrieval
  CLI, then `/spec M3-1`. Also open: the `README` has no install or quick-start yet.

### 2026-11-06 (M2-3: validate tune against real labels)
- **Done**: drafted, approved, built, ran and closed M2-3 (`c54d651`). `ignore_identical_ids` on
  `evaluate` / `Index.evaluate` (`orchestration/{evaluate,index}.py`); `candidates()` and
  `EXHAUSTIVE` moved to `orchestration/tune.py`. `scripts/export_golden_texts.py` also writes
  `qrels_test.json` (run for all 3 datasets). New `scripts/validate_tune.py`; results in
  `docs/dev/tune_validation.md`. Reviewer: round 1 "fix first" (2 wrong lines in the skill), round 2 "merge".
- **Result**: tune fails the 0.01 test everywhere. SciFact worst 0.014 (≥ 1 ms), NFCorpus holds
  above 1 ms, ArguAna ~0.042 at every budget. Below 1 ms tune picks `lexical`/`dense` over `bm25`.
- **Decisions**: a script over smve-lab's memmaps, not a test (7 GB, 34–111 min per dataset).
  Label side uses real-query p50, tune side span p50 (favours tune; noted in spec Changes).
  Owner: M2-4 makes tune-with-labels the recommended mode; label-free tune stays, documented.
- **Shortcuts with a ceiling**: none added (`grep ponytail:` still 7).
- **Surprises**: two causes, not one. (1) BGE-M3 spans favour BGE-M3's own `dense`/`hybrid`, so
  tune never picks `bm25 > maxsim@d`, the label winner. (2) On ArguAna MaxSim itself (0.4854) loses
  to `dense` (0.5274), so no agreement-with-MaxSim method can win there. laterank's `bm25` alone
  runs slightly above smve-lab's (ArguAna +0.0022, cause not found). An ArguAna rerank scores d-1
  docs when the query is in its top d.
- **Next**: `/spec M2-4`. Start from design §8's fallback: tune-with-labels = `evaluate` + "best
  nDCG@10 within budget". Decide with the owner whether it's `Index.tune(budget, qrels=...)` or a
  method on `Evaluation`, and where label-free tune's caveat is documented.

### 2026-11-05 (M2-2: label-free tune + set_default)
- **Done**: drafted, approved, built and closed M2-2 (`8cfae63`). New `orchestration/tune.py`
  (`Budget`, `Point`, `TuneReport`, `span_queries`, the `tune` engine); `Index.tune`,
  `Index.set_default` and a shared `Index._pipeline` (`orchestration/index.py`); `evaluate.py`'s
  timing loop is now `_timed_runs`, shared with tune; `kernels/rng.py::span_bits`. Tests in
  `tests/orchestration/{test_tune,test_index}.py`, `tests/kernels/test_rng.py`. Reviewer: round 1
  "fix first" (5 findings, all fixed), round 2 "merge".
- **Decisions**: agreement = overlap of the final top 10 with exhaustive's top 10, source document
  dropped from both (one rule for every candidate; equals §8's "key in x's top d"). Only the
  Pareto frontier is kept; `best` = its last point within budget. Default 500 span queries, each
  timed once (1000 x best-of-3 estimated ~25 min on SciFact; not measured). Spans only, no user
  query texts (add when M2-3 shows spans mislead). No MB budget (stages are already built).
  Spans get their own RNG stream (`"spans"` tag): rng.py's rule gives the raw stream to anchors only.
- **Shortcuts with a ceiling**: none added (`grep ponytail:` still 7, all older).
- **Surprises**: my first `span_queries` read the raw `splitmix64(seed)` stream, the same bits as
  SMVE's anchors at seed 0 (reviewer caught it). `Budget(np.float64(20))` was refused by my
  `type(...) in (int, float)` check. `10**400` doesn't overflow in `0 < ms < math.inf` (Python
  compares int and float exactly), so the planned overflow guard was dead code. `EncoderInfo`
  with an `hf_id` needs a 40-char revision, so test encoders differ by `vocab` instead.
- **Next**: `/spec M2-3`. First decide how tune runs on SciFact/NFCorpus/ArguAna: `Index.tune` needs
  an index folder (copying SciFact's tokens is 3.8 GB, the reason M2-1's golden test avoids
  `Index.evaluate`), so either call the `tune` engine on stages built from smve-lab's memmaps
  (`~/Projects/smve-lab/artifacts/embeddings/{scifact,nfcorpus,arguana}_bgem3`) with BGE-M3
  encoding the spans (`--extra bge-m3`, `HF_HUB_OFFLINE=1`), or build real indexes. Also: ArguAna
  needs BEIR's `ignore_identical_ids` (deferred from M2-1), and tune's real runtime is unmeasured.

### 2026-11-05 (M2-1: evaluate + exhaustive)
- **Done**: `MaxSimReranker.search` and `_check_dim` (`components/maxsim.py`); `EXHAUSTIVE` accepted
  by `_check_pipeline`, registered in `Index._stages`, plus `Index.evaluate` (`orchestration/index.py`);
  new `orchestration/evaluate.py` (`Score`, `Evaluation.table()`, `evaluate`); tests in
  `tests/components/test_maxsim.py`, `tests/orchestration/{test_index,test_evaluate}.py`; golden
  `test_exhaustive_pipeline` (SciFact 0.6991 / 0.9203). Reviewer: "merge", findings 1-2 fixed.
- **Decisions**: the engine takes `search` and `doc_id` callables plus `dim`, so `index.py` stays
  small (spec Changes). Query dim is checked before any search (a late failure loses a long run).
  Skipped reviewer findings 3-4 (tests nobody asked for; an empty `Evaluation` is unreachable).
- **Shortcuts with a ceiling**: none added (`grep ponytail:` still 7, all older).
- **Surprises**: my first AT3 test asked a `maxsim@20` pipeline for k=100 (the pipeline refuses it);
  the test now uses evaluate's rule k = min(100, last depth). The two `exhaustive` golden tests took
  182 s together, not ~27 s as logged in October (cause not checked).
- **Next**: `/spec M2-2` (label-free `tune`): start
  with the research takeaways in the 2026-11-04 entry (Pareto frontier, `Budget`, span queries,
  exhaustive top-10 as answer key) and decide the `Budget` shape with the owner.

### 2026-11-04 (framing M2)
- **Done**: researcher report on evaluation APIs (PyTerrier, BEIR, ir_measures, ranx, PyLate,
  bm25s) and label-free tuning (FAISS AutoTune, ann-benchmarks, PLAID repro). Split M2 into
  M2-1 evaluate, M2-2 tune, M2-3 validation. Drafted and approved
  `docs/specs/M2-1-evaluate.md`. No code changed.
- **Decisions** (owner OK): no dataset loader in M2 (`evaluate` takes plain dicts; loader with
  the CLI in M3); only nDCG@10 + R@100; no significance tests; timing = `perf_counter` per
  query, 3 repeats, keep each query's fastest, report p50 (the first pass is the warmup, no
  extra pass); encoding not timed (queries arrive `Encoded`, like `search`; an encode column
  in M3); `exhaustive` = a `search` method on `MaxSimReranker`, always available. Golden checks
  `exhaustive` through the `_run` harness, not `Index.evaluate` (avoids a 3.8 GB token copy).
- **Research takeaways for M2-2/M2-3**: copy FAISS's operating points (keep the Pareto
  frontier, pick the best point within budget); FAISS asks for ≥1000 tune queries; agreement
  with exhaustive ≠ relevance (PLAID repro: BM25+LADR has lower overlap but higher nDCG on
  DL19), so the M2-3 check against real labels is a release gate; verbatim document spans may
  favour bm25 (unverified). Source files cached in this session's scratchpad (temporary).
- **Surprises**: `ruff format --check` also formats Python blocks inside the spec markdown.
  The schedule slipped ~1 week (design doc had M2 at Oct 26–30).
- **Next**: build M2-1 task 1: add `MaxSimReranker.search` in `components/maxsim.py` with the
  AT1 brute-force test in `tests/components/test_maxsim.py`, then accept `exhaustive` in
  `orchestration/index.py::_check_pipeline` (AT2).

### 2026-11-04 (M1-7: ColBERT encoder + two-model measurement)
- **Done**: researcher report (ST 6 `MultiVectorEncoder` vs PyLate, candidates). Drafted, approved,
  built and closed M1-7 (`92b6c70`, CI green, spec done): `components/colbert.py`
  (`ColBERTEncoder`), `colbert` entry in `ENCODERS`, `colbert` extra (`sentence-transformers>=6`),
  `tests/components/test_colbert.py` (stubs), golden `tests/golden/test_colbert.py`, the missing
  MUVERA `# ponytail:` comment, a ColBERT section in the `golden-values` skill. Ignored the
  Wheelhouse files (`2d32d3e`, `Spec: none`). Reviewer: "merge" (low findings, all fixed or noted).
- **Measured** (SciFact, `bm25 > maxsim@100`, Recall@100 0.9249 for both): answerai-colbert-small-v1
  (96-d) nDCG@10 0.7384, 44.1 KB/doc, encode 31 s; mxbai-edge-colbert-v0-32m (64-d) 0.7453, 33.4
  KB/doc, 46 s. BGE-M3 on the same pipeline: 0.7086, 718 KB/doc. One run each, noise not measured.
- **Decisions**: mxbai is the recommended default (owner OK: 24% smaller, nDCG@10 0.007 higher;
  answerai encodes faster), not hard-coded: `ColBERTEncoder` takes `hf_id`. ST
  `MultiVectorEncoder`, not PyLate (pins ST 5.3.0). `settings` read off the loaded model, incl.
  `query_expansion` (with its `token` key) and `skiplist_tasks` (owner OK after review). `hf_id`
  required; ONNX files skipped in the download (134 MB, not 993).
- **Shortcuts with a ceiling** (`# ponytail:`): the MUVERA `fdes` build holds the matrix in RAM
  (425 MB SciFact, ~8 GB at 100k docs); upgrade: chunked `.npy` memmap. Not marked in code: the
  nDCG floor 0.60 in the golden test only catches broken vectors; tighten once a band is agreed.
- **Surprises**: mxbai does not pad queries (8 vectors) while answerai pads to 32. The design doc's
  "64-d, 19x smaller" holds only for mxbai. My first BGE-M3 size (735 KB) was in decimal KB, the
  table is KiB: 718. First `ruff` run caught a mutable class attribute in my test stub.
- **Next**: draft the next spec (start of M2); the default-model question is closed.
  Open: no real build-then-`Index.open` test with a real ColBERT model (settings JSON round trip);
  the `--extra bge-m3` and `--extra colbert` golden commands are in CLAUDE.md.
### 2026-11-03 (M1-6: BGE-M3 encoder + Index.build)
- **Done**: researcher report (sentence-transformers 6.1, PyLate, FlagEmbedding: none has presets,
  none pins a revision). Drafted, approved and built M1-6 (`2a3cdfc`, CI green): `EncoderInfo.settings`
  (`storage/manifest.py`), `components/encoder.py` (`Encoder`, `EncodedDocs`, `load_encoder`),
  `components/bgem3.py` (`BGEM3Encoder`), `Index.build` (`orchestration/index.py`), `bge-m3` extra
  (`FlagEmbedding>=1.4.2`), `tests/conftest.py` (`FakeEncoder`), golden `tests/golden/test_encoder.py`
  (20 docs + 5 queries vs smve-lab). Reviewer: round 1 "merge" (low findings), round 2 "merge".
  Spec closed `done`.
- **Decisions**: spec split in two: presets dropped (no prior art, only SMVE/MUVERA have a band),
  ColBERT encoder moved to M1-7 (no weights cached, default model decided by measurement). Model
  pinned by downloading one sha ourselves (FlagEmbedding takes `main`); `settings` records
  `max_length` + `fp16`. Text sent to the encoder is smve-lab's `"title. text"`. Lexical golden
  compares weights within 0.002, not term ids (a 0.00014 term dropped under fp16 noise).
- **Shortcuts with a ceiling** (none marked `# ponytail:` in code, no spec approved to add them):
  `Index.build` finds a pipeline needing a missing head only after encoding (upgrade: encoders
  declare their heads); `BGEM3Encoder` downloads the whole repo snapshot, 30 files, size not
  measured (upgrade: `allow_patterns`); `EncoderInfo` is frozen but holds a dict, so it isn't
  hashable (upgrade when something hashes it).
- **Surprises**: smve-lab's golden lexical had a weight-0.00014 term ours lacked, so exact ids
  were a trap. Adding the `bge-m3` extra made `uv.lock` grow by 2,600 lines (torch and friends). My `_pick_device` imported torch
  even for `device="cpu"`; the reviewer caught that no test covered it.
- **Next**: `/spec M1-7`. Start with the `researcher` agent: which small ColBERT checkpoint
  (e.g. a 64-d one) loads in `MultiVectorEncoder`, and whether `revision=<sha>` takes a full sha
  (round 1 left that unverified). Pick with the owner how ColBERT's prefixes, query expansion and
  skiplist go in `EncoderInfo.settings`. Still open from M1-5: add the missing MUVERA `# ponytail:`
  comment for the in-RAM `fdes` build (commit `Spec: none`), and commit or ignore the Wheelhouse
  files (`.gitignore`, `wheelhouse.json`, `scripts/git-hooks/post-*`).

### 2026-11-03 (M1-5: MUVERA)
- **Done**: drafted, approved and built M1-5: `kernels/muvera.py` (`make_planes`, `fde_encode`),
  `kernels/rng.py::random_signs`, `components/muvera.py` (`MuveraStage`, `MUVERA_DEFAULT`),
  `Index.create(muvera=True)`, golden `test_muvera_pipeline` over seeds 0-2. Reviewer: "merge"
  (low findings fixed). Commits `ea95aa5` (code) and `51a429d` (close), CI green, spec done.
  SciFact seeds 0/1/2: `muvera` 0.5335/0.5378/0.5118, `muvera > maxsim@100` 0.6895/0.6870/0.6904,
  Recall@100 0.842-0.856; table in the golden-values skill.
- **Decisions**: setting r40_k4_p32 uncentered (owner's pick: 425 MB, ~10 ms in smve-lab). Planes
  stored, not regenerated (6 MB vs SMVE's 256 MB anchors). Alone band widened to [0.500, 0.580]
  after seed 2 (owner OK); reranked band stays 0.6888 ± 0.01 (spread only 0.003). `fdes` finite
  check is at build (`arrays`), not on every `open` (would read 425 MB). Golden test runs 3 seeds.
- **Shortcuts with a ceiling**: the build holds the whole `fdes` matrix in RAM (425 MB on SciFact,
  8 GB at 100k docs); upgrade: write chunks into a `.npy` memmap. The spec promised a
  `# ponytail:` comment for it, but none is in the code (no spec was approved to add one).
- **Surprises**: seed 2 fell below my first band (0.5118 vs 0.533): MUVERA's alone spread (0.026)
  is 2.6x SMVE's. Build takes 22-25 s, not the ~10 s I estimated from synthetic data (torch:
  13.7 s). The reviewer found my spec paired two settings' centering numbers; fixed from `runs.csv`.
- **Next**: `/spec M1-6` (encoders + presets, the last M1 item). Start with the `researcher` agent
  on how sentence-transformers, PyLate and FlagEmbedding load BGE-M3 and ColBERT models, and pick
  with the owner how a preset maps to first stage + settings (`smve(preset=...)`, `muvera(...)`).
  Also decide with the owner: add the missing `# ponytail:` comment (commit `Spec: none`), and
  commit or ignore the Wheelhouse files (`.gitignore`, `wheelhouse.json`, `scripts/git-hooks/post-*`).

### 2026-10-13 (M1-4: SMVE)
- **Done**: drafted, approved (10-12) and built M1-4: threaded `kernels/smve.py::topk_blocks`,
  `components/smve.py` (new, `SMVEStage`, `SMVE_BALANCED`), `Index.create(smve=True)`, golden
  `test_smve_pipeline` with its own fixture. Seeds 0/1/2: `smve` 0.5502/0.5469/0.5572, `smve >
  maxsim@100` 0.6913/0.6864/0.6960, all in band; table in the golden-values skill. Reviewer:
  round 1 "fix first" (fork hang, 4-thread cap, no NaN check), round 2 "fix first" (NaN check
  made a 1.8 GB temporary), round 3 "merge". Commands added to CLAUDE.md and START_HERE.md.
- **Decisions**: anchors regenerated at open and SHA-256-checked (storing them = 256 MB, more than
  the index). Settings fixed to `balanced`, all in the manifest. Postings validated by an inner
  `LexicalStage`. Thread pool per call (fork-safe), one slice per core. SMVE golden stays under
  `-m slow`; skip with `-k "not smve"` (owner left the pick to Claude). ±0.01 rerank tolerance
  confirmed across seeds (spread 0.0096, so don't tighten it).
- **Shortcuts with a ceiling** (`# ponytail:`): SciFact build ~5.2 min vs torch's 3.3; upgrade:
  group-max prefilter for top-k. Anchors rebuilt on every open (~2.2 s); upgrade: build on first
  query or cache.
- **Surprises**: `argpartition` was 89% of encode time and single-threaded (SciFact ~20 min at
  first). My first threading capped itself at 4 threads (256-row floor). `argpartition(P, w-k)`
  ranks NaN as largest, unlike the old `-P` form. laterank's Recall@100 is ~0.01 below smve-lab's.
- **Then**: committed `198728b` (`Spec: M1-4`), CI green on all 6 jobs, M1-4 marked done.
- **Next**: `/spec M1-5` (MUVERA). Start by reading smve-lab's `src/smve_lab/muvera.py` and its 54 runs in
  `results/scifact_bgem3/muvera/` (one seed each). golden-values has no MUVERA band yet, so first
  pick the setting with the owner and decide how to set a band.

### 2026-10-11 (M1-3: BM25)
- **Done**: researcher report (bm25s, Lucene, Tantivy). Drafted, approved and built M1-3:
  `components/{bm25,stopwords}.py` (new), `Encoded.text`, `Index.create(bm25=True)`,
  `scripts/export_golden_texts.py` (run once in smve-lab; wrote `scifact_bgem3/{docs,queries}_texts.json`
  into its git-ignored artifacts), golden `bm25` 0.68999 / 0.92489 and `bm25 > maxsim@100`
  0.70864 / 0.92489. Reviewer: "merge" with 4 small fixes, all applied (spec Changes).
- **Decisions**: segments store raw tf postings + sorted terms (`"\n"`-joined uint8); df,
  lengths, N, avg length derived at open, so 0.2 can add segments (Lucene/Tantivy do the same).
  Weights follow smve-lab's float32 order: bit-identical on all 445,237 SciFact postings.
  sklearn's 318 stopwords copied with BSD-3 notice (no sklearn dep). Unknown query words dropped.
  `bm25` built only with `bm25=True` (presets in M1-6 turn it on). Counts built with integers.
- **Shortcuts with a ceiling** (`# ponytail:`): `BM25Stage` computes every weight at open and
  holds them in RAM (O(nnz); SciFact ~2 MB). Upgrade: weight only the touched postings per query
  when open time or RAM on a big corpus asks for it.
- **Surprises**: smve-lab's tokenizer also drops stopwords and stems (the brief said regex only).
  bm25 nDCG@10 is 0.68999 vs smve-lab's 0.68956: ranking smve-lab's own scores with laterank's
  `top_k` also gives 0.68999, so it's tie order in one query's top 10, not scoring. My first
  golden "texts optional" fix still crashed the non-bm25 cases; caught by running without texts.
- **Next**: owner commits and pushes; when CI is green on all 6 jobs, tick AT6 and run
  `/spec M1-3 done`. Then `/spec M1-4` (SMVE, band [0.537, 0.572] for `balanced` alone, ±0.01
  around 0.6898 with `> maxsim@100`): start by reading smve-lab's `smve.py` call sites and
  laterank's `kernels/smve.py`, and decide how the anchor matrix is stored or regenerated
  (building 1024 × 131,072 anchors takes ~4.5 s, so avoid it on every `open`).

### 2026-10-11 (M1-2: lexical + hybrid)
- **Done**: closed M1-1 (`0b3a7cc`, CI green). Drafted, approved and built spec M1-2:
  `components/{lexical,hybrid}.py` (new), `Encoded.lexical`, `DenseStage.scores` and
  `from_segment(seg, cfg, n_docs)`, `Index.create(lexical=...)`, golden fixture reads smve-lab's
  sparse arrays, golden tests parametrized over 5 pipelines. Reviewer: round 1 "fix first"
  (unchecked `alpha`, postings dtypes, NaN weights, loose `Encoded.lexical`), round 2 "fix
  first" (my NaN fix broke `lil`/`dok` input), round 3 "merge". Then `Encoded.dense` refuses
  NaN/inf too (owner OK). `golden-values` skill gained the `lexical` Recall@100 row.
- **Decisions**: lexical weights stored as postings (CSC). `hybrid` stores nothing, is built
  whenever dense + lexical are given, alpha fixed at 0.3 (paper value, smve-lab's golden) and
  recorded in the manifest; it finds its partners by the stage names `dense`/`lexical`. Every
  stage has `scores(q)` beside `search`. Stored stages check their own arrays in `from_segment`.
  Non-finite weights/vectors are refused at `create` and in queries (a NaN silently empties results).
- **Shortcuts with a ceiling**: none added today (the 3 existing `# ponytail:` notes stand).
- **Surprises**: lexical ties at score 0 didn't move any golden value. A validation fix can
  break valid inputs: reading `.data` before conversion crashed on `lil`/`dok` matrices; the
  new all-formats test pins it. `ruff format` also reformats Python blocks inside spec Markdown.
- **Next**: owner commits and pushes; when CI is green, tick AT7 and run `/spec M1-2 done`.
  Then `/spec M1-3` (BM25: 0.6896 / 0.9249, `bm25 > maxsim@100` 0.7086). Start by reading
  smve-lab's `src/smve_lab/bm25.py` and its ASCII tokenizer, and decide how corpus-wide stats
  (IDF, average doc length) are stored so new segments can update them (design §6: store raw
  term counts, compute BM25 weights at query time).

### 2026-10-11 (M1-1: Index + pipeline, dense first)
- **Done**: researcher report (PyTerrier, PyLate, RAGatouille, bm25s, sentence-transformers: none
  saves a pipeline; only PyTerrier has a pipeline object). Drafted, approved and built spec M1-1:
  `errors.py` (+`IndexExistsError`, `PipelineSyntaxError`), `components/{base,dense,maxsim}.py`,
  `orchestration/{pipeline,index}.py`, tests in `tests/{components,orchestration}/`, golden
  `test_dense_then_maxsim_at_100`. Reviewer: round 1 "fix first" (6 boundary gaps, all fixed),
  round 2 "merge" (2 low notes, fixed). Spec Changes lists every fix.
- **Decisions**: pipeline stays a string in the manifest, parsed into a frozen `Pipeline` that
  validates itself (a hand-made one obeys the grammar too). Two Protocols, `FirstStage` and
  `Reranker`, so `maxsim > dense` fails at parse/check time. k > last depth is an error.
  Rerankers have no manifest `stages` entry (the token store belongs to the index). The golden
  test builds components from smve-lab's memmaps (no 3.8 GB copy); `Index` is tested on synthetic data.
- **Shortcuts with a ceiling** (`# ponytail:`): `Index.open` reads single-segment indexes only
  (upgrade: global-row map across segments when `add()` lands in 0.2). `Index.create` cleanup
  can leave files on Windows if memmaps are still open (only if `open` rejects what `create`
  wrote; upgrade: close arrays before deleting).
- **Surprises**: the tie rule change (global row vs candidate order) didn't move the golden
  value at all (0.69658 vs 0.6966). `/spec` can't be started by Claude; the owner types it.
- **Next**: owner commits and pushes; when CI is green on all 6 jobs, tick AT8 and run
  `/spec M1-1 done`. Then `/spec M1-2` (lexical + hybrid; golden `hybrid > maxsim@100` = 0.6977,
  `hybrid` alone 0.6842): first decide where `Encoded.lexical` lives and how lexical weights are
  stored (postings via `kernels/inverted.py::to_postings`).

### 2026-10-11 (wrap-up: M0 closed, M1 order agreed)
- **Done**: closed M0-3 and M0-4 after CI confirmed them (`d3afe51`). The spec hook now also asks
  before a Bash command sets a spec status (`scripts/specs.py`, `.claude/settings.json`), and
  `.gitattributes` keeps the qrels TSV byte-exact (`0f3ba0d`).
- **Decisions**: M1 order, proposed by Claude, owner said "you pick": M1-1 Index + pipeline with
  `dense` first (precomputed vectors, no torch; golden `dense > maxsim@100` = 0.6966) → M1-2
  lexical + hybrid (0.6977) → M1-3 BM25 (0.6896 / 0.7086) → M1-4 SMVE (band [0.537, 0.572])
  → M1-5 MUVERA → M1-6 encoders + presets. Why dense first: no new math, so M1-1 can focus
  on the `Stage`/`Index`/`Pipeline` interfaces. BM25 carries the most new design (tokenizer,
  corpus-wide stats that must survive new segments).
- **Shortcuts with a ceiling** (`# ponytail:`): `storage/segment.py::_check_relpath` misses
  case-only folder clashes, Windows-reserved names and file/folder clashes; upgrade to
  per-component checks if users ever choose array paths. `scripts/specs.py` checks shell
  commands by text match; upgrade to a PostToolUse diff of spec statuses if it proves too weak.
- **Surprises**: the CRLF qrels file needed `.gitattributes` to stay hash-stable. The earlier
  hook gap let a Bash script approve a spec without the owner's prompt (owner had typed the
  approval, so no harm).
- **Next**: start the `researcher` agent on: how PyTerrier (`>>` pipelines), PyLate, RAGatouille,
  bm25s and sentence-transformers represent a first stage vs a reranker, and how a pipeline is
  written down so it can be saved in a manifest. Then `/spec M1-1` (Index + pipeline, dense
  first) from its report, the design doc §3–6, and the `golden-values` skill.

### 2026-10-11 (golden harness)
- **Done**: M0-4 approved with option (a) (commit the CC BY-SA 4.0 qrels with a README) and built:
  `tests/golden/{conftest.py,test_scifact.py,data/}`. With smve-lab's embeddings, SciFact exhaustive
  0.69912 / 0.92033 and dense 0.65418 / 0.89700 (nDCG@10 / Recall@100) match smve-lab to 5 decimals.
  `pytest -m slow` takes 27 s (exhaustive MaxSim 26.4 s in numpy vs 27.6 s in smve-lab's torch).
- **Then**: reviewer `merge`; committed `dbba2bb` (`Spec: M0-4`) and `0f3ba0d` (hook now asks
  before shell commands set a spec status; `.gitattributes` pins the qrels bytes). CI green on
  both (199 passed, 2 golden skipped, on 3 OS × 2 Pythons). M0-4 marked done.
- **Next**: plan M1. Use the `researcher` agent before designing the first stage, then `/spec`.

### 2026-10-11 (storage review fixes)
- **Done**: fixed all 5 review defects plus the 'also proposed' items (case-colliding array
  paths, storage may import only kernels/storage/errors/_version, `DocStore` docstring,
  hypothesis round trip for docs, decisions logged under spec Changes). Item 1 was already
  fixed in the tree. Ticked AT2.
- **Review rounds 2 and 3**: fixed Windows `PermissionError` on a folder manifest (`is_file()`
  check), token count before the segment rename, falsy metadata, `TypeError`→`ValueError` in
  `write_manifest`, non-dict doc lines, unclosed `NpzFile`, `RecursionError` and 4300-digit ints
  in JSON. Deleted `Segment.__repr__`, `_fsync_tree`, one trivial test. Deferred (ponytail note in
  `segment.py::_check_relpath`): case-only folder clashes, Windows-reserved names, file/folder
  path clashes.
- **Then**: committed (`0a07721` ponytail, `841e9fc` storage), owner pushed, CI green on all 6
  jobs (197 passed, no skips). M0-3 marked done. Drafted `docs/specs/M0-4-golden-harness.md`.
- **Next**: owner reviews M0-4 (open question: committing the CC BY-SA 4.0 SciFact qrels
  fixture), then `/spec M0-4 approve`.

### 2026-10-05 (storage build and review)
- **Done**: built M0-3 tasks 1–4: `errors.py`, `_version.py`, `storage/{arrays,manifest,docs,
  segment,_fsync}.py`, `tests/storage/test_{arrays,manifest,docs,segment}.py`, a stdlib-only
  check for leaf modules in `tests/test_layers.py`. Includes a hard-kill subprocess test for the
  manifest and an end-to-end test (store tokens → reopen memory-mapped → MaxSim).
- **Decisions**: `storage/_fsync.py` holds `fsync_folder` for both writers (spec Changes, 10-04).
- **Review (fix first), confirmed by hand.** Proposed fixes, waiting for the owner's OK:
  1. `segment.py::_check_relpath` accepts `"C:/x.npy"`, `"C:x.npy"`, `"x.npy:ads.npy"`, which
     escape the segment folder on Windows → reject any `:` and Windows-absolute paths; add to the
     bad-paths test.
  2. `write_manifest` writes a manifest it can't read back (stages mutated after construction,
     tuples, int keys) → parse its own output and require equality before touching disk.
  3. `load_array(mmap=False)` lets `MemoryError` escape (header claims a huge shape);
     `load_ragged` crashes with `TypeError` on 0-d `flat` → `IndexFormatError` for both.
  4. Wrong-`format` message has no action (fails AT2) → add one.
  5. Raw errors: `DocStore.ids()` (`KeyError`/`JSONDecodeError`), `write_docs` (`TypeError`,
     partial file, NaN, int keys → strict read-back check, delete partial file), `read_manifest`
     on a folder (`IsADirectoryError` → wrap; leave `PermissionError` raw).
  Also proposed: reject array keys that collide under `casefold()`; restrict storage's
  `laterank.*` imports in `test_layers.py` to kernels/storage/errors/_version; reword the
  `DocStore` "can always be renamed" docstring (memmaps hold files open on Windows); a hypothesis
  round-trip test for docs; record small decisions (segment cap 999,999, docs reject unknown
  keys/duplicate ids, constructors raise `ValueError`) under spec Changes.
  Skipped: fsync failing after a successful rename (document only), `Manifest` hashability,
  numpy error-text matching (misses still become `IndexFormatError`).
- **Next**: owner agrees to (or edits) the fix list above; then fix item 1 first
  (`_check_relpath` + test), then 2–5, re-run the reviewer, tick AT2/AT10, commit `Spec: M0-3`.

### 2026-10-05
- **Done**: installed the `ponytail` plugin (DietrichGebert/ponytail v4.12.0) at project scope
  (`.claude/settings.json`). Wired it into `CLAUDE.md`, the implementer and reviewer agents
  (the reviewer now returns a **Delete-list**), the `/spec`, `port-module` and `/wrap-up` skills,
  the spec template (`Skipped: X, add when Y`) and `START_HERE.md` §5.
- **Decisions**: specs, checkpoint stops and teaching explanations override ponytail's
  "ship it, explain in 3 lines". There is a "never cut" list in `CLAUDE.md`.
- **Ponytail audit applied**: commit `30318d9` (`Spec: none`) removed `kernels/smve.py::smve_single`
  (the tests' `_dense_reference` now takes `reps` and is the only reference), and `smve_encode` now
  reuses `maxsim._batches`. In M0-3 (uncommitted): `DocStore` reuses `arrays._ragged_problem`;
  `segment.py` reuses `manifest._SEGMENT_ID`; `Manifest.format_version` field removed (spec
  change logged). Kept `show_progress`/tqdm: owner wants progress bars. 177 tests pass.
- **Next**: continue M0-3. The storage code and its tests exist, uncommitted; tick the
  acceptance tests, run the reviewer, then `/wrap-up` to commit.

### 2026-10-03 (session 2)
- **Done**: committed the spec-workflow tooling on its own. Reviewer agent on the kernels: "fix
  first". Fixed SMVE crashing on empty documents (int64 `bincount`), `top_k` with unsigned/bool/
  large-int scores (it no longer negates or converts), MaxSim offsets not checked against the token
  array, and the docstring that tripped AT4's grep. Gave each RNG purpose its own stream
  (`sample_indices` now derives its seed). Added a frozen hash of 10**6 float64 normals and an
  odd-d anchor test. Re-review: "merge". Ticked AT1–AT6.
- **Open**: AT3's cross-platform claim is proven only on macOS until CI runs on ubuntu/windows
  (needs the first push).
- **Next**: owner runs `/spec M0-2 done`; draft `/spec M0-3` (storage) for the owner to approve.

### 2026-10-03
- **Done**: scaffolded the repo (checkpoint 1). Added project skills (`/start-session`, `/wrap-up`,
  `/split-work`, `/explain`, `port-module`, `golden-values`), agents (`implementer`, `reviewer`,
  `researcher`), and a `commit-msg` git hook that rejects Claude attribution.
- **Decisions**: name `laterank`; Apache-2.0; Python 3.12+; add/delete deferred to 0.2; default
  model decided by M1 measurements.
- **Decisions (checkpoint 2)**: the RNG computes log/sin/cos with its own polynomials (only
  + − × ÷ √, which IEEE-754 makes bit-identical everywhere) instead of `np.log`/`np.cos`, whose
  last bit varies by platform. Anchors are generated column by column, so a wider matrix starts
  with the narrower one. `anchor_sha256` hashes the whole float32 matrix (the design doc said
  "first anchor block"). Metrics live in `kernels/metrics.py` until an `eval/` layer exists.
  Building the largest anchor matrix (1024 × 131,072) takes ~4.5 s, so M1 should avoid
  regenerating it on every `open`.
- **Workflow**: switched to spec-driven development (`docs/specs/README.md`). Added a `/spec`
  command, `scripts/specs.py` (with tests in `tests/test_specs.py`), a Claude Code hook that
  blocks `src/` and `tests/` edits without an approved spec, a versioned git hook
  (`core.hooksPath = scripts/git-hooks`) requiring a `Spec: <id>` trailer, and a CI spec check.
  Wrote M0-1 (done) and M0-2 (approved) after the fact from the scaffold prompt.
- **Next**: the owner reviews `docs/specs/M0-2-kernels.md` (note that `anchor_sha256` differs
  from the design doc), the reviewer agent checks the kernels, then commit with `Spec: M0-2`,
  run `/spec M0-2 done`, and draft `/spec M0-3` (storage).
