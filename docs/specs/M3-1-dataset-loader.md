---
id: M3-1
title: BEIR dataset loader
status: approved
approved: 2026-11-06
---

# M3-1: BEIR dataset loader

## Goal

After this checkpoint a user can get a BEIR dataset into the shapes laterank already takes:
`data = load_beir(download_beir("nfcorpus"))`, then `Index.build(path, data.corpus, encoder)` and
`Index.evaluate(...)` with `data.qrels`. It also creates the `api/` layer, which M3-2's CLI builds
on. Only the standard library is used, so `pip install laterank` gains no dependency.

## Out of scope

- The CLI, encode-time reporting and `tune --apply`: M3-2.
- Encoding queries: `data.queries` holds text, and `evaluate` still takes `Encoded` queries.
- Top-level exports (`from laterank import load_beir`): `__init__` exports `__version__` only.
- Skipped: a general `load("./notes/")` for folders and `.jsonl` (design §4), add when M3-2 or a
  user needs documents that aren't BEIR-shaped.
- Skipped: datasets beyond the 7 below, add a table line once its zip is downloaded and hashed.
  `cqadupstack` needs more than that: its 12 subfolders don't fit this layout.
- Skipped: a mirror for when UKP's host is down, add when it goes down.
- Skipped: a download progress bar, add when someone complains (largest zip: 142.5 MB).
- Skipped: two processes downloading one dataset at once. The loser gets an `OSError`, and running
  it again returns the folder. Add a lock when something runs downloads in parallel.

## Interfaces

```python
# api/__init__.py: a docstring only.  api/datasets.py: standard library only.
@dataclass(frozen=True)
class BeirData:
    corpus: list[dict[str, str]]  # {"id", "title", "text"}, file order; goes to Index.build
    queries: dict[str, str]  # query id -> text, only the queries judged in this split
    qrels: dict[str, dict[str, int]]  # query id -> {doc id: relevance}, relevance > 0 only
    ignore_identical_ids: bool  # pass to Index.evaluate / Index.tune


def load_beir(folder: str | Path, split: str = "test") -> BeirData:
    """Read a BEIR folder: corpus.jsonl, queries.jsonl, qrels/<split>.tsv.

    Ids become strings. `title` defaults to "". `metadata` is dropped. Title and text stay
    separate (`Index.build` joins them). Documents with no text are kept. Judged documents
    missing from the corpus stay in qrels and count as misses. `ignore_identical_ids` is True
    when the folder is named "arguana".

    Raises:
        FileNotFoundError: one of the three files is missing (says which).
        ValueError: a JSON line is invalid or lacks `_id` or a string `text`; a qrels row isn't
            `query-id, corpus-id, integer score`; the first qrels row isn't that header; a judged
            query isn't in queries.jsonl. Names the file, and the line where there is one.
    """


def download_beir(name: str, folder: str | Path | None = None) -> Path:
    """Fetch one BEIR dataset on request and return its folder, `<folder>/<name>`.

    `folder` defaults to ~/.cache/laterank/beir. If `<folder>/<name>` exists it is returned with no
    network call: it only ever appears through the final rename, so it is complete. Otherwise:
    urlopen (timeout 60 s) into a temp folder inside `folder`, check the md5, extract, `os.replace`.

    Raises:
        ValueError: `name` isn't in the table (lists the names; before any network call), or the
            md5 doesn't match (says so; nothing is left behind, so a retry starts clean).
        OSError: the network or disk failed (`URLError` is an `OSError`). Nothing is left behind.
    """
```

URL: `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip`. Each zip holds
one folder named `{name}`. The md5 table, hashed from the downloaded zips and matching BEIR's README:

| name | MB | md5 |
|---|---|---|
| `scifact` | 2.8 | `5f7d1de60b170fc8027bb7898e2efca1` |
| `nfcorpus` | 2.4 | `a89dba18a62ef92f7d323ec890a0d38d` |
| `arguana` | 3.8 | `8ad3e3c2a5867cdced806d6503f29b99` |
| `fiqa` | 17.9 | `17918ed23cd04fb15047f73e6c3bd9d9` |
| `quora` | 15.9 | `18fb154900ba42a600f84b839c173167` |
| `scidocs` | 142.5 | `38121350fc3a4d2f48850f6aff52e4a9` |
| `trec-covid` | 73.9 | `ce62140cb23feb9becf6270d0d1fe6d1` |

## Decisions

| decision | why | differs from design doc? |
|---|---|---|
| two functions, `load_beir(folder)` and `download_beir(name)` | owner's pick: download only on explicit request (BEIR leaves license checks to the user), and offline users never touch the network | yes: design has one `mvr.datasets.beir("scifact")` |
| `BeirData` holds plain dicts, not passed whole to `evaluate` | `evaluate` needs `Encoded` queries, so a dataset object can't go straight in | yes: design has `index.evaluate(ds, ...)` |
| keep only queries judged in the split | `queries.jsonl` holds every split (nfcorpus 3,237 vs 323), so encoding all of it costs up to 10x more | no |
| drop relevance ≤ 0 | smve-lab does; `ndcg_at_k` would count trec-covid's `-1` rows as negative gain | no |
| check the qrels header, don't skip it blindly | a file with no header would silently lose its first judgment | no |
| keep empty documents | dropping them changes the corpus compared with other BEIR results. Measured: BGE-M3 and mxbai encode them, and build and search work. They can outrank a real document (BGE-M3: 4.58 vs 3.37 on one query, 5 docs). FiQA has 38, TREC-COVID 1 | no |
| `ignore_identical_ids` from the folder name | ArguAna is the only dataset where queries are corpus documents (1,298 of 1,406 ids). `# ponytail: a renamed ArguAna folder loses the rule, add an argument if that bites` | no |
| md5, with no other check | the only published checksum; it catches error pages and cut-off files, which BEIR and bm25s miss. Not a security measure | no |
| golden checks use NFCorpus and ArguAna texts, and qrels on all three | owner's pick: smve-lab's stored SciFact texts differ (all 1,109 queries start with `". "`, 18 docs by one space). Existing SciFact golden tests are untouched | no |
| errors are `ValueError` / `FileNotFoundError`, no new class | same as `Index.create`; M3-2's CLI catches both | no |

## Acceptance tests

- [ ] AT1: on a tiny BEIR folder, `load_beir` handles a CRLF header, integer `_id`s (become strings), scores 0 and -1 (dropped), a query judged only in another split (left out), `metadata` (dropped), a missing `title` (`""`), an empty document (kept), and a judged document missing from the corpus (kept in qrels).
- [ ] AT2: `split="dev"` reads `qrels/dev.tsv`, and `ignore_identical_ids` is True for a folder named `arguana` (also when passed as `"."` from inside it), False otherwise.
- [ ] AT3: each error in `load_beir`'s Raises list fires, and its message names the file.
- [ ] AT4: `Index.build` then `Index.evaluate` run on `load_beir`'s output (FakeEncoder).
- [ ] AT5: with a `file://` URL and a test zip (no network in CI), `download_beir` returns a folder `load_beir` reads; a second call returns it without opening the URL; a wrong md5 raises and leaves `folder` empty, and the next call succeeds; an unknown name raises before opening the URL.
- [ ] AT6 (golden, slow, real network): after `download_beir` of scifact, nfcorpus and arguana, qrels equal `qrels_test.json` on all three. On nfcorpus and arguana, `_encoder_text` of each document and each query text equal the stored texts by id. `ignore_identical_ids` is True only for arguana.
- [ ] AT7: `uv run pytest -q`, `ruff check`, `ruff format --check` and `tests/test_layers.py` pass, and `api/datasets.py` imports only the standard library.
- [ ] AT8: reviewer agent verdict `merge`.

## Tasks

| # | task | files it owns | depends on | parallel? |
|---|---|---|---|---|
| 1 | `BeirData`, `load_beir`; AT1-AT4 | `api/__init__.py`, `api/datasets.py`, `tests/api/test_datasets.py` | - | no |
| 2 | `download_beir`; AT5 | same files | 1 | no |
| 3 | golden test; AT6; note in `CLAUDE.md` that `-m slow` now downloads about 9 MB | `tests/golden/test_datasets.py`, `CLAUDE.md` | 2 | no |

## Changes

Dated lines for anything changed after approval: `- 2026-10-06: <what> - <why>`.

- 2026-11-06: `tests/test_layers.py` joins task 1's files (`api/datasets.py` added to
  `LEAF_MODULES`) - AT7's stdlib-only check reuses the existing test instead of a new one.
- 2026-11-06 (reviewer round 1): qrels read as bytes, not `csv` - `csv` let bad encoding and huge
  fields escape as raw errors; the reviewer's bytes parser matched `csv` on all 13 real files. `http.client` errors
  become `OSError`, as the docstring promises. `ignore_identical_ids` uses `abspath`, not
  `resolve` - a symlink named `arguana` lost the rule. Marked the download race `# ponytail:`.
  `_rows` catches `RecursionError` too. Added tests for the rename, a cut-off response, a symlink,
  deep JSON and bad qrels encoding. Not changed: `_id` and `title` values beyond the spec's checks
  (`Index.build` refuses a non-string title).
- 2026-11-06 (reviewer round 2): verdict `merge`. The cut-off test now fails during the read, where
  a real cut-off happens. Known limit: a symlink with another name pointing at an `arguana` folder
  counts as a renamed folder, so it loses the rule (same `# ponytail:` ceiling).
