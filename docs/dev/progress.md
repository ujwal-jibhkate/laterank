# laterank progress log

This file is the project's memory between Claude Code sessions. `/start-session` reads it, and
`/wrap-up` updates it. Newest entries go first.

## Current state

- Milestone **M1: methods + measurements**. M1-1..M1-4 done (`198728b`, CI green on
  3 OS × 2 Pythons). No spec is approved; next is `/spec M1-5` (MUVERA).
- Works: first stages `dense`, `lexical`, `hybrid`, `bm25`, `smve`, each alone or `> maxsim@N`.
  All 11 SciFact golden tests pass (~6.5 min; `-k "not smve"` ~1 min).
- Nothing is broken. 292 passed, 11 golden skip without `LATERANK_GOLDEN_DIR`.

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
- [ ] 5. MUVERA
- [ ] 6. Encoders + presets

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

## Log

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
