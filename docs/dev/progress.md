# laterank progress log

This file is the project's memory between Claude Code sessions. `/start-session` reads it, and
`/wrap-up` updates it. Newest entries go first.

## Current state

- Milestone **M1: methods + measurements**. Checkpoint **M1-1** (Index + pipeline, dense first)
  is built and reviewed (`merge`), uncommitted at the time of writing; 8/9 ATs ticked, AT8 waits
  for CI on 3 OS after the owner pushes.
- Works: `Index.create/open/search` on precomputed vectors, pipeline strings like
  `dense > maxsim@100`, golden SciFact `dense > maxsim@100` = 0.69658 / 0.897 (smve-lab 0.6966 / 0.8970).
- Nothing is broken. 243 passed, 3 golden skip without `LATERANK_GOLDEN_DIR` (3/3 pass with it, 35 s).

## M0 checklist

- [x] 1. Skeleton: uv + hatchling/hatch-vcs, Apache-2.0, extras declared, CI (3 OS × py3.12/3.13)
- [x] 2. Kernels: MaxSim, top-k, CSC scoring, SMVE encode, portable SplitMix64 anchors + SHA-256, metrics
- [x] 3. Storage: manifest.json (atomic write), segments, `.npy` with mmap and no pickle, docs.jsonl + offsets
- [x] 4. Golden harness: `LATERANK_GOLDEN_DIR`, SciFact qrels fixture, exhaustive 0.6991 and dense 0.6542

## M1 checklist

- [ ] 1. Index + pipeline, dense first (`dense > maxsim@100` = 0.6966): built, waiting for CI + `done`
- [ ] 2. Lexical + hybrid (`hybrid > maxsim@100` = 0.6977)
- [ ] 3. BM25 (0.6896; `bm25 > maxsim@100` = 0.7086)
- [ ] 4. SMVE (band [0.537, 0.572])
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
| `scripts/compare_first_stages.py` (rerank the top depth with MaxSim) | `components/maxsim.py`, `orchestration/pipeline.py` | a `Pipeline` object + string grammar; ties go to the lower global row (smve-lab kept candidate order) |

## Log

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
