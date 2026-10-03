# laterank progress log

This file is the project's memory between Claude Code sessions. `/start-session` reads it, and
`/wrap-up` updates it. Newest entries go first.

## Current state

- Milestone **M0: foundations** (planned Oct 5–9, 2026).
- Checkpoint 1 (skeleton) is committed: pyproject, src layout, `kernels/` and `storage/`
  packages, CI workflow, `tests/test_layers.py`.
- Checkpoint 2 (kernels) is in progress. Checkpoints 3 (storage) and 4 (golden harness) are not
  started.

## M0 checklist

- [x] 1. Skeleton: uv + hatchling/hatch-vcs, Apache-2.0, extras declared, CI (3 OS × py3.12/3.13)
- [ ] 2. Kernels: MaxSim, top-k, CSC scoring, SMVE encode, portable SplitMix64 anchors + SHA-256, metrics
- [ ] 3. Storage: manifest.json (atomic write), segments, `.npy` with mmap and no pickle, docs.jsonl + offsets
- [ ] 4. Golden harness: `LATERANK_GOLDEN_DIR`, SciFact qrels fixture, exhaustive 0.6991 and dense 0.6542

## Ported from smve-lab

| smve-lab file | laterank file | what changed |
|---|---|---|

## Log

### 2026-10-03
- **Done**: scaffolded the repo (checkpoint 1). Added project skills (`/start-session`, `/wrap-up`,
  `/split-work`, `/explain`, `port-module`, `golden-values`), agents (`implementer`, `reviewer`,
  `researcher`), and a `commit-msg` git hook that rejects Claude attribution.
- **Decisions**: name `laterank`; Apache-2.0; Python 3.12+; add/delete deferred to 0.2; default
  model decided by M1 measurements.
- **Next**: the owner OKs checkpoint 1 and makes the first commit, then start checkpoint 2 with the
  portable anchor generator and its frozen test vector.
