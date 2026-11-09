---
id: M4-1
title: Install guide, quick-start and top-level imports
status: approved
approved: 2026-11-08
---

# M4-1: Install guide, quick-start and top-level imports

## Goal

After this checkpoint someone who has never seen the repo can install laterank and search their own
documents by following only the README, from Python or from the terminal. The main classes import
from `laterank` directly, so the paths people copy from the quick-start are the ones we keep.

## Out of scope

- Chunking: cut from 0.1 (owner, 2026-11-08). The README tells users to split long documents.
- Publishing to PyPI, the `0.1.0b1` tag, the `Pre-Alpha` classifier and the empty `rerank` and
  `pandas` extras: M4-2.
- `--encoder colbert` accepting non-ColBERT models: its own spec.
- Skipped: a docs site, add when the README gets too long to read comfortably.
- Skipped: `Index.build`'s default pipeline following the encoder as the CLI's does (today it fails
  after encoding on ColBERT, which has no `hybrid`), add when encoders declare their heads. The
  quick-start passes `bm25=True` and a pipeline.
- Skipped: exporting `Hit`, `Pipeline`, `load_beir`, `download_beir`, add when the README shows them.
- Skipped: lazy imports. `Index` costs 0.2 s to import (measured in M3-2).
- Skipped: running the install on BGE-M3, add when a user reports a problem (same command, other extra).

## Interfaces

`src/laterank/__init__.py`, the only file in `src/` that changes. The deep paths keep working.

```python
from laterank._version import __version__
from laterank.components.bgem3 import BGEM3Encoder
from laterank.components.colbert import ColBERTEncoder
from laterank.components.encoder import load_encoder
from laterank.orchestration.index import Index
from laterank.orchestration.tune import Budget

__all__ = ["BGEM3Encoder", "Budget", "ColBERTEncoder", "Index", "__version__", "load_encoder"]
```

`README.md`, in this order:

1. **Title and status line.** "Nothing here is usable yet" becomes: "Pre-release. It works end to
   end, but the API and on-disk format may change before 0.1.0."
2. **Intro paragraph**, unchanged.
3. **`## Install`**: Python 3.12+; `pip install "laterank[colbert] @ git+https://github.com/ujwal-jibhkate/laterank"`,
   or `[bge-m3]`; a note that either extra brings in torch, a large download.
4. **`## Quick start`**: the block below, then reopening an index, then the two notes.
5. **`## Choosing a pipeline`**: as today, but importing `from laterank import Budget`, and the
   example defines `queries` and `qrels` so it runs after the quick-start.
6. **`## Command line`**: the `uv sync` line goes; the examples assume the install above. The rest stays.
7. **`## Working on laterank`** and **`## License`**, unchanged.

```python
from laterank import ColBERTEncoder, Index

docs = [
    {"id": "a", "text": "Statins lower cholesterol but can cause muscle pain."},
    {"id": "b", "text": "BM25 is a classic ranking function for search."},
    {"id": "c", "text": "Tea is made from the leaves of a plant."},
]
encoder = ColBERTEncoder("mixedbread-ai/mxbai-edge-colbert-v0-32m")
index = Index.build("./my_index", docs, encoder, bm25=True, pipeline="bm25 > maxsim@100")

for hit in index.search(encoder.encode_query("do statins hurt muscles?"), k=2):
    print(hit.rank, hit.id, round(hit.score, 3), hit.text)
```

Reopening, in a later session:

```python
from laterank import Index, load_encoder

index = Index.open("./my_index")
encoder = load_encoder(index.manifest.encoder)  # the exact model commit the index was built with
```

The two notes: documents need a unique string `id` and a `text`, can have a `title` and a
`metadata` dict, and other keys are refused (the CLI's `--docs` file uses BEIR's `_id`), and **long documents are cut off**: BGE-M3 and mxbai keep the first 512 tokens
(`max_length` and `document_length`), so split long documents into smaller ones before `build`.

## Decisions

| decision | why | differs from design doc? |
|---|---|---|
| Install from the git URL for now | Not on PyPI yet (the name was free on 2026-11-08); M4-2 changes this one line | yes: PyPI |
| Five names at the top level, plus `__version__` | What the quick-start and tune snippet need. `load_encoder` is the only way to get the pinned model back | no |
| Plain imports, not lazy | No model library is imported until an encoder is created, and the import costs 0.2 s | no |
| Quick-start uses mxbai | The recommended model (M1-7), in the README only; no code default, as `CLAUDE.md` requires | no |
| No new test for "import doesn't need torch" | `test_every_module_imports_with_the_model_libraries_blocked` already does `import laterank` with torch blocked | no |
| Fresh-install check uses the working tree | The URL only sees pushed code, and only the owner pushes; AT8 checks the URL afterward | no |
| README only, no docs site | One page is enough for a beta | yes: M4 lists a docs site |
| Chunking cut from 0.1 | Owner's call; golden datasets reach their numbers without it | yes: M3 lists chunking |

## Acceptance tests

Each line is one check that can be verified. Tick it (`- [x]`) only when the test exists and passes.

- [x] AT1: `uv run python -c "from laterank import Index, Budget, ColBERTEncoder, BGEM3Encoder, load_encoder, __version__"` exits 0.
- [x] AT2: `tests/test_layers.py` passes with the new `__init__.py`, including the model-libraries-blocked test.
- [x] AT3: In a new venv outside the repo, install the working tree with `[colbert]`, then run the README's quick-start block copied unchanged: doc `a` ranks first.
- [x] AT4: In the same venv, a new Python process runs the README's reopening block, and searching that index ranks doc `a` first.
- [x] AT5: In the same venv with the venv activated, `laterank build ./cli_idx --docs docs.jsonl --encoder colbert --hf-id <mxbai>`, then `laterank search ./cli_idx "statin side effects" -k 2`, both exit 0 and list hits.
- [x] AT6: The README matches the Interfaces order, has no `uv sync`, no `laterank.orchestration` import, and states the 512-token cut-off.
- [x] AT7: `uv run pytest -q`, `uv run ruff check .` and `uv run ruff format --check .` all pass.
- [ ] AT8: After the owner pushes: CI is green on all 6 jobs, and the README's git URL installs in a new venv and AT1's import works there.
- [x] AT9: `reviewer` agent verdict is `merge`.

## Tasks

| # | task | files it owns | depends on | parallel? |
|---|---|---|---|---|
| 1 | Top-level exports | `src/laterank/__init__.py` | - | no |
| 2 | Rewrite the README sections | `README.md` | 1 | no |
| 3 | Fresh-venv checks (AT3-AT5), about 10 minutes with the torch download | none (temporary folder) | 2 | no |

## Changes

- 2026-11-08: the README's document note states the real rules (unique string `id`, `metadata`, other keys refused) - reviewer: a wrong key otherwise fails only after the whole corpus is encoded.
- 2026-11-08: the tune example defines `queries` and `qrels` (owner OK) - reviewer: it could not be pasted and run.
