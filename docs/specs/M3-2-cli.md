---
id: M3-2
title: Command line
status: done
approved: 2026-11-08
---

# M3-2: Command line

## Goal

After this checkpoint a user can do everything from a terminal, with no Python: `laterank build`,
`search`, `evaluate` and `tune`. The commands wrap `Index`, and they add the one thing `Index`
leaves to the caller: turning query text into `Encoded` vectors with the index's own encoder. It
adds no dependency.

## Out of scope

- Chunking and add/delete (design's other M3 items): add/delete is fixed for 0.2; chunking gets its
  own spec if wanted.
- Skipped: `--json`, add when a script needs it (the Python API already serves scripts).
- Skipped: `search --queries FILE` and a daemon, add when the measured model load (AT9) hurts.
- Skipped: `--docs -` (stdin), a folder of files, and `id` as a key, add when a user asks.
- Skipped: flags for `repeats`, `n_queries`, `seed`, device and batch size, add when someone needs one.
- Skipped: checking that `--dataset` matches the index, add when it bites (a wrong one shows zeros).
- Skipped: refusing `--encoder colbert --pipeline hybrid` before encoding. It still fails after
  encoding and writes nothing (`Index.build`'s known limit), add when encoders declare their heads.
- Skipped: lazy imports. `Index` costs 0.2 s to import (measured), so `--help` stays fast enough.
- Skipped: `--version`, quiet handling of broken pipes, add when asked.

## Interfaces

```text
laterank build INDEX (--docs FILE | --dataset D) --encoder {bge-m3,colbert}
               [--hf-id ID] [--revision SHA]
               [--bm25] [--smve] [--muvera] [--pipeline P]
laterank search INDEX QUERY [-k 10] [--pipeline P]
laterank evaluate INDEX --dataset D [--split test] [--pipeline P]...
laterank tune INDEX --budget-ms MS [--dataset D] [--split test] [--apply]
```

`D`: a folder holding `corpus.jsonl` wins; anything else is a BEIR name, passed to `download_beir`
(a notice goes to stderr first). `--docs FILE`: rows shaped like BEIR's `corpus.jsonl`: `_id`,
`text`, optional `title`.

| command | stdout | notes |
|---|---|---|
| `build` | `built <INDEX>: <n> documents, stages [...], default pipeline '<p>'` | `--hf-id` is required for `colbert`, optional for `bge-m3` (`BAAI/bge-m3`). `--revision` None means today's main; the sha is recorded either way. Default `--pipeline`: `hybrid > maxsim@100` (bge-m3), `bm25 > maxsim@100` (colbert). A pipeline starting with `bm25`, `smve` or `muvera` builds that stage without its flag |
| `search` | one line per hit: rank, score (4 decimals), id, text with whitespace collapsed and cut to 80 chars | `--pipeline` defaults to the index's default |
| `evaluate` | `Evaluation.table()`, then `query encoding: p50 <x> ms over <n> queries` | `--pipeline` repeats; none means the default. Passes the dataset's `ignore_identical_ids` |
| `tune` | `TuneReport.table()`, then `best: <p>` or `best: none (nothing fits <MS> ms)` | with `--dataset`: labels. Without: label-free, and stderr warns that it measures agreement with MaxSim, not relevance. `--apply` calls `set_default(best)` and prints `default pipeline: <p>`; with no best it exits 1 |

Exit codes: 0 ok; 1 for `LaterankError`, `ValueError`, `OSError`, `ImportError`, as one line
`laterank: error: <message>` on stderr; 2 for usage errors (argparse). Other exceptions keep their
traceback, because they are bugs. Results go to stdout; notices and progress bars go to stderr.

```python
# api/cli.py; pyproject.toml: [project.scripts] laterank = "laterank.api.cli:main"
def main(argv: Sequence[str] | None = None) -> int:
    """Parse argv (default sys.argv[1:]), run one command, return the exit code."""


def _make_encoder(name: str, hf_id: str | None, revision: str | None) -> Encoder:
    """Build a BGEM3Encoder or ColBERTEncoder (imported here). Tests patch this."""


def _folder(value: str) -> Path:
    """The rule for D above: the dataset's folder."""


def _dataset(value: str, split: str) -> BeirData:
    """load_beir(_folder(value), split)."""


def _encode_queries(encoder: Encoder, texts: Mapping[str, str]) -> tuple[dict[str, Encoded], float]:
    """Encode each text with a tqdm bar. Returns the queries and p50 ms per query."""


# api/datasets.py: split out of load_beir, which calls it
def load_corpus(path: str | Path) -> list[dict[str, str]]:
    """Read a corpus.jsonl into [{"id", "title", "text"}]. Raises as load_beir does for it."""
```

## Decisions

| decision | why | differs from design doc? |
|---|---|---|
| argparse, one file `api/cli.py` | adds no dependency; bm25s, Pyserini and ir_datasets use it too | no |
| `--encoder` required on `build`; the other commands read the manifest | CLAUDE.md: no hard-coded default encoder. `load_encoder` already rebuilds it. llm stores the model the same way | yes: design builds with a default model |
| default pipeline follows the encoder | `Index.build`'s `hybrid` default fails on ColBERT (no dense or lexical heads), and only after encoding. Both values are design §3's | no |
| the pipeline's first stage turns on its stage | `--pipeline "bm25 > maxsim@100"` without `--bm25` would also fail after encoding | no |
| `--dataset` takes a name or a folder; no `download` command | typing a name is the explicit request M3-1 asked for, and a cached or local dataset never touches the network. ir_measures and Pyserini do the same | no |
| `build --dataset` reads only `corpus.jsonl`, and has no `--split` | the corpus is the same for every split, so the flag could only make a build fail | no |
| a folder counts only if it holds `corpus.jsonl` | the researcher's "a path wins" breaks `laterank evaluate scifact --dataset scifact`: the index folder would shadow the name | no |
| `--docs` reads BEIR's row shape, `_id` included | `load_beir` already parses and checks it, so own documents need no second reader | yes: design has `load("./notes/")` |
| query encoding is one line, not a column | the encoder runs once per query, so a column repeats one number on every row | yes: M2-1 planned a column |
| `evaluate`, not `eval` | matches `Index.evaluate` and the M3 checklist | yes: design has `mvr eval` |
| every `search` call loads the model | accepted: colgrep and llm do too. Load time gets measured in AT9 | no |
| exit 1 for our errors, 2 for usage, bugs keep tracebacks | the convention in tantivy, llm, ir_datasets and argparse | no |

## Acceptance tests

Tests call `main([...])` with `FakeEncoder` (patching `_make_encoder` and `ENCODERS`), and check
exit code, stdout and stderr.

- [x] AT1: `build` from `--docs` and from a `--dataset` folder (holding only `corpus.jsonl`, or fetched by name) gives an index that opens. Defaults: `hybrid > maxsim@100` for bge-m3; `bm25 > maxsim@100` with the `bm25` stage for colbert. `--muvera` builds its stage, and `--pipeline "bm25 > maxsim@5"` builds `bm25` with no flag.
- [x] AT2: `--encoder colbert` without `--hf-id` and a pipeline that doesn't parse both exit 1 before `_make_encoder` runs. A non-empty INDEX folder exits 1.
- [x] AT3: `_dataset`: a folder with `corpus.jsonl` is read with no download; a folder without it, named like a BEIR dataset, goes to `download_beir`; an unknown name exits 1 and lists the names.
- [x] AT4: `search` prints k lines in the agreed format, a long text is cut to 80 chars, and `--pipeline` overrides the default.
- [x] AT5: `evaluate` prints one row per `--pipeline` (default: the index's) and the encoding line; for a folder named `arguana` its numbers equal `Index.evaluate(..., ignore_identical_ids=True)`.
- [x] AT6: `tune --dataset` shows `nDCG@10`; without it, `agreement` and the stderr warning. `--apply` changes the manifest's default and the next `search` uses it; without `--apply` the manifest is unchanged. A budget nothing fits prints `best: none`, and with `--apply` exits 1.
- [x] AT7: a missing index, a missing extra (`ImportError`, the message keeps its `pip install` hint) and `--budget-ms 0` each exit 1 with one stderr line and no traceback; a missing argument exits 2; a `RuntimeError` is not caught.
- [x] AT8: `uv run laterank --help` lists the four commands, and the `laterank` console script loads `main`.
- [x] AT9 (golden, slow, `tests/golden/test_cli.py`, needs `--extra colbert`): `build --dataset scifact --encoder colbert` with mxbai at its pinned sha, then `evaluate`, gives nDCG@10 within 0.01 of 0.7453 and Recall@100 within 0.01 of 0.9249. `search` returns 10 hits. This is also the first real ColBERT build-then-`Index.open`. Wall time of one `search` is recorded under Changes. If the band misses, stop and report: BEIR's SciFact texts differ slightly from the stored ones that 0.7453 used.
- [x] AT10: `uv run pytest -q`, `ruff check`, `ruff format --check` pass, including `tests/test_layers.py`.
- [x] AT11: reviewer agent verdict `merge`.

## Tasks

| # | task | files it owns | depends on | parallel? |
|---|---|---|---|---|
| 1 | `load_corpus`; parser, error handling, `_dataset`, `_make_encoder`, `build`, `search`; AT1-AT4, AT7, AT8 | `api/datasets.py`, `api/cli.py`, `pyproject.toml`, `tests/api/test_cli.py` | - | no |
| 2 | `_encode_queries`, `evaluate`, `tune`; AT5, AT6 | same files | 1 | no |
| 3 | golden test; AT9; README "Command line" section (say to quote pipelines: an unquoted `>` redirects output); command in `CLAUDE.md` | `tests/golden/test_cli.py`, `README.md`, `CLAUDE.md` | 2 | no |

## Changes

Dated lines for anything changed after approval: `- 2026-10-06: <what> - <why>`.

- 2026-11-08: `search` and `evaluate` check `--pipeline` against the index (`Index._pipeline`)
  before the model loads - a typo otherwise costs a model load and, in `evaluate`, encoding every
  query. No interface or acceptance test changes; one extra test covers it.
- 2026-11-08: AT9 measured: nDCG@10 0.7459, Recall@100 0.9249 (BEIR texts; 0.7453 used the stored
  ones), query encoding p50 7.1 ms, one `laterank search` 6.8 s wall time with the model load
  (cached model, Apple-silicon Mac). `START_HERE.md` also got the golden command (project rule).
- 2026-11-08 (reviewer round 1): error messages are collapsed to one line (Hugging Face's run to
  7). Empty `--dataset`, `--docs` and `--pipeline` are refused, not read as absent (`tune --dataset
  "$UNSET"` silently went label-free). `build` checks the INDEX folder before the model loads.
  stdout uses `errors="replace"` (one Greek letter aborted `search` on a cp1252 stream). Tests added
  for `--split`, the stage choices, `_make_encoder`, `tune`'s identical-ids rule, `OSError`, and the
  notice staying on stderr. Not changed: the download notice printing before an unknown name fails.
- 2026-11-08 (reviewer round 2): verdict `merge`, with 3 small notes, all fixed: an empty `--dataset`
  no longer means the current folder, `main` works when stdout can't be reconfigured, and a test
  checks that `--hf-id` and `--revision` reach `_make_encoder` (the model pin).
- 2026-11-08: owner's decision: `build --dataset` reads only `corpus.jsonl`, `--split` removed from
  `build`, `_folder` added (`_dataset` uses it), AT1 extended - the flag only chose which qrels file
  had to exist, and a folder with just a corpus failed.
