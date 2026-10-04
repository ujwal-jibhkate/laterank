# laterank progress log

This file is the project's memory between Claude Code sessions. `/start-session` reads it, and
`/wrap-up` updates it. Newest entries go first.

## Current state

- Milestone **M0: foundations** (planned Oct 5–9, 2026).
- Checkpoint 1 (skeleton) is committed: pyproject, src layout, `kernels/` and `storage/`
  packages, CI workflow, `tests/test_layers.py`.
- Checkpoint 2 (kernels) is done: committed (`9659cf9`), reviewed (`merge`), spec M0-2 marked
  done by the owner. 85 tests pass.
- Checkpoint 3 (storage): spec M0-3 approved 2026-10-03; building starts with task 1. Checkpoint 4
  (golden harness) not started.

## M0 checklist

- [x] 1. Skeleton: uv + hatchling/hatch-vcs, Apache-2.0, extras declared, CI (3 OS × py3.12/3.13)
- [x] 2. Kernels: MaxSim, top-k, CSC scoring, SMVE encode, portable SplitMix64 anchors + SHA-256, metrics
- [ ] 3. Storage: manifest.json (atomic write), segments, `.npy` with mmap and no pickle, docs.jsonl + offsets
- [ ] 4. Golden harness: `LATERANK_GOLDEN_DIR`, SciFact qrels fixture, exhaustive 0.6991 and dense 0.6542

## Ported from smve-lab

| smve-lab file | laterank file | what changed |
|---|---|---|
| `maxsim.py` (`maxsim_scores`, `subset_ragged`) | `kernels/maxsim.py` | asserts → `ValueError`; progress bar off by default; empty query/doc sets allowed |
| `maxsim.py` + `inverted_index.py` (`top_k`) | `kernels/topk.py` | returns indices, not id dicts; ties broken by lower row (argpartition alone was arbitrary at the k-th place); 1-D and 2-D |
| `inverted_index.py` | `kernels/inverted.py` | class → pure functions over `(indptr, rows, weights)`; scores returned as float32 |
| `smve.py` | `kernels/smve.py` | torch → numpy (`argpartition`, `bincount`, `np.unique`); anchors from `kernels/rng.py`; `token_mean` samples with the portable RNG and accumulates in float64 |
| `smve.make_anchors` (torch.Generator) | `kernels/rng.py` | new: SplitMix64 + Box–Muller with polynomial log/sin/cos, bit-identical on every platform; `anchor_sha256` |
| `metrics.py` (nDCG, Recall) | `kernels/metrics.py` | pandas dropped; `mean_metric` replaces `evaluate_run`; P/MRR/MAP wait for M2 |

## Log

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
