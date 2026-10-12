---
id: M0-4
title: Golden harness
status: approved
approved: 2026-10-11
---

# M0-4: Golden harness

## Goal

A test harness that re-runs smve-lab's two simplest SciFact pipelines on smve-lab's stored BGE-M3
embeddings, using only laterank's kernels and storage, and checks that the numbers match
smve-lab's results. It proves the ported numpy code gives the same answers as the torch
research code, on real data. M1 then adds one golden test per stage on top of the same harness.

## Out of scope

- BM25, lexical, hybrid, `> maxsim@100`, SMVE and MUVERA golden tests: M1, with the stages that
  compute them. These tests call the kernels directly; there is no `Index` yet.
- NFCorpus and ArguAna: M1. Skipped: their qrels fixtures, add when a test needs them.
- Converting the embeddings into a laterank index folder: M1 builds indexes from them.
- Running golden tests in CI: the embeddings are about 14 GB. Golden tests skip in CI. The design's
  nightly job waits for M1.
- Skipped: a script that regenerates the qrels fixture. Add one when a second dataset needs
  qrels. Until then, the source and SHA-256 recorded below are enough to check the file.
- Nothing in `src/` changes.

## Interfaces

```text
LATERANK_GOLDEN_DIR=~/Projects/smve-lab/artifacts/embeddings   # smve-lab's folder, unchanged
  scifact_bgem3/
    {docs,queries}_colbert_flat.npy      (T, 1024) float16, ragged with ..._colbert_offsets.npy
    {docs,queries}_colbert_offsets.npy   (n + 1,) int64
    {docs,queries}_dense.npy             (n, 1024) unit vectors
    {docs,queries}_ids.json              list[str], row order
tests/golden/data/scifact-test-qrels.tsv   # byte copy of BEIR's test.tsv, header + 339 rows
tests/golden/data/README.md                # source URL, revision, SHA-256, CC BY-SA 4.0 credit
tests/golden/conftest.py
tests/golden/test_scifact.py
```

```python
# tests/golden/conftest.py
@pytest.fixture(scope="session")
def golden_dir() -> Path:
    # $LATERANK_GOLDEN_DIR. Unset -> pytest.skip. Set, but no such folder -> pytest.fail
    # naming the path (a typo must not turn into a silent skip).
@pytest.fixture(scope="session")
def scifact(golden_dir) -> SimpleNamespace:
    # qrels: {query_id: {doc_id: score}}, score > 0 only (as smve-lab's load_qrels)
    # query_ids: list(qrels), the 300 judged queries, in qrels order
    # doc_ids: list[str]
    # q_flat, q_offsets: ColBERT vectors of the judged queries (kernels.maxsim.subset_ragged)
    # d_flat, d_offsets: doc ColBERT vectors, d_flat memory-mapped
    # q_dense, d_dense: dense vectors, judged queries only for q_dense
    # Every array is read with laterank.storage.arrays.load_array (allow_pickle=False).
    # Fails, naming the file, if one is missing.

# tests/golden/test_scifact.py  (both marked @pytest.mark.slow)
def test_exhaustive_maxsim(scifact): ...   # maxsim_scores -> top_k(…, 100) -> mean_metric
def test_dense(scifact): ...               # q_dense @ d_dense.T -> top_k(…, 100) -> mean_metric
```

## Decisions

| decision | why | differs from design doc? |
|---|---|---|
| `LATERANK_GOLDEN_DIR` points at smve-lab's embeddings folder as it is, with no conversion | there's no `Index` until M1, and a converted copy would double 3.8 GB on disk | no |
| arrays are read with laterank's `load_array`, not `np.load` | it tests our storage code on a real 3.8 GB memory map, and keeps the no-pickle rule | no |
| the qrels fixture is a byte copy of `BeIR/scifact-qrels` `test.tsv`, revision `2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029`, SHA-256 `0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c` | 5.4 KB. The test needs no network and no `datasets` dependency, and the hash shows it's the exact file smve-lab used | no |
| the qrels file is committed, with a `README.md` giving its source and its CC BY-SA 4.0 license (owner's choice (a)) | `tests/` isn't shipped in the wheel, and the licence asks for credit, which the README gives | no |
| the loader keeps only `score > 0` | that's what smve-lab's `load_qrels` does (SciFact has no zero scores, but M1's datasets might) | no |
| if `LATERANK_GOLDEN_DIR` is set but wrong, the test fails instead of skipping | a typo would otherwise make the golden tests look green | no |
| tolerance is ±0.002 absolute, for nDCG@10 and Recall@100 | from the `golden-values` skill. laterank's `top_k` breaks ties by lower row and torch's is arbitrary, so ranks can differ slightly | no |
| Recall@100 is checked too | it costs nothing extra, and it catches errors deeper in the ranking that nDCG@10 can miss | no (it adds a check) |
| the expected numbers are written as constants in the test, each with a comment naming the source file | the tests must not depend on smve-lab's results folder | no |

## Acceptance tests

- [ ] AT1: with `LATERANK_GOLDEN_DIR` unset, `uv run pytest -q` passes and both golden tests are
  reported as skipped. CI does the same on all three operating systems.
- [x] AT2: with `LATERANK_GOLDEN_DIR` set to a folder that doesn't exist, the golden tests fail,
  and the message names that path.
- [x] AT3: `tests/golden/data/scifact-test-qrels.tsv` has the SHA-256 given above. It loads as 300
  queries and 339 judgments.
- [x] AT4: SciFact exhaustive MaxSim gives nDCG@10 = 0.6991 ± 0.002 and Recall@100 = 0.9203 ± 0.002
  (`results/cross_dataset/summary.md`).
- [x] AT5: SciFact dense gives nDCG@10 = 0.6542 ± 0.002 and Recall@100 = 0.8970 ± 0.002 (same file).
- [x] AT6: `LATERANK_GOLDEN_DIR=... uv run pytest -m slow` passes locally. Record how long it takes in
  `docs/dev/progress.md`; smve-lab's exhaustive scoring took 27.6 s.
- [x] AT7: the reviewer agent's verdict is `merge`.

## Tasks

| # | task | files it owns | depends on | parallel? |
|---|---|---|---|---|
| 1 | qrels fixture + `conftest.py` | `tests/golden/data/{scifact-test-qrels.tsv,README.md}`, `tests/golden/conftest.py` | M0-3 | no |
| 2 | the two golden tests | `tests/golden/test_scifact.py` | 1 | no |

## Changes

- 2026-10-11: added a session fixture `scifact_qrels` in `conftest.py`. The fast AT3 check (which runs in CI) needs the qrels without `LATERANK_GOLDEN_DIR`, and with `--import-mode=importlib` tests can't import helpers from `conftest`. `scifact` uses this fixture. - found while building
- 2026-10-11: approved with option (a): commit the qrels fixture with a source/license README.
- 2026-10-11: drafted.
