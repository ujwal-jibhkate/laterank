---
id: M1-6
title: BGE-M3 encoder and Index.build
status: done
approved: 2026-11-03
---

# M1-6: BGE-M3 encoder and Index.build

## Goal

Today `Index.create` takes vectors somebody else made. After this checkpoint a user hands laterank
**texts**: an `Encoder` turns them into token, dense and lexical vectors (BGE-M3 through
FlagEmbedding), `Index.build` writes the index, and the manifest records the exact model
revision and settings so a reopened index encodes queries the same way. The torch code lives in
one component, behind the `bge-m3` extra. Everything else stays torch-free and is tested with a
fake encoder.

## Out of scope

- **The old M1-6 was "encoders + presets". This spec is its first half.** Proposed split: M1-6
  BGE-M3 encoder (this), M1-7 ColBERT encoder (`MultiVectorEncoder`) and the default-model
  measurement. The progress log's checklist would change to match once you agree.
- Skipped: presets (`PRESETS` dict, `smve(preset=...)` in the pipeline grammar). No library we
  read has them, and `SMVE_BALANCED` / `MUVERA_DEFAULT` already are the only settings with a
  measured band. Add when a second measured setting exists (M2 `tune`).
- Skipped: a ColBERT encoder, query prefixes, query expansion, skiplists. Add in M1-7, when a
  checkpoint is chosen; no ColBERT weights are cached here.
- Skipped: `Index.search_text(text)`. Reopen with `load_encoder(index.manifest.encoder)` and pass
  `encode_query(text)` to `search`. Add with the API layer.
- Skipped: batching progress bars beyond tqdm per batch, multi-GPU (`devices` list).

## Interfaces

```python
# storage/manifest.py: EncoderInfo gains one field (format_version stays 1: nothing is released)
settings: dict  # JSON-only, default {}; BGE-M3 writes {"max_length": 512, "fp16": True}
# _ENCODER_KEYS gains "settings". Old manifests without it become IndexFormatError (rebuild).


# components/encoder.py (new; no torch)
@dataclass(frozen=True)
class EncodedDocs:
    tokens: np.ndarray  # (T, dim) float16, every document's token vectors
    offsets: np.ndarray  # (n + 1,) int64
    dense: np.ndarray | None  # (n, d) float32
    lexical: sp.csr_matrix | None  # (n, vocab) float32


class Encoder(Protocol):
    info: EncoderInfo

    def encode_docs(self, texts: Sequence[str]) -> EncodedDocs: ...
    def encode_query(self, text: str) -> Encoded: ...  # tokens, dense, lexical and text set


def load_encoder(info: EncoderInfo) -> Encoder:
    """Rebuild the encoder an index was made with. Raises ValueError for an unknown
    `info.name`, and when the model on disk isn't `info.revision`."""


# components/bgem3.py (new; FlagEmbedding and huggingface_hub imported inside __init__)
class BGEM3Encoder:
    def __init__(
        self,
        hf_id: str = "BAAI/bge-m3",
        revision: str | None = None,
        *,
        max_length: int = 512,
        use_fp16: bool = True,
        device: str = "auto",
        batch_size: int = 32,
    ): ...

    # revision=None: resolve today's `main` to its 40-hex sha, download that sha, record it.
    # The model is always loaded from the downloaded folder (FlagEmbedding has no `revision`).


# orchestration/index.py
@classmethod
def build(
    cls,
    path: Path,
    docs: Sequence[Mapping],
    encoder: Encoder,
    *,
    bm25: bool = False,
    smve: bool = False,
    muvera: bool = False,
    pipeline: str = "hybrid > maxsim@100",
) -> Index:
    """Encode each doc's text, then Index.create(...) with the encoder's heads
    (dense, lexical, hybrid). Errors as in create."""
```

pyproject: `bge-m3 = ["FlagEmbedding>=1.4.2"]` (the version smve-lab ran).

## Decisions

| decision | why | differs from design doc? |
|---|---|---|
| FlagEmbedding for BGE-M3, not sentence-transformers | only it returns token, dense and lexical vectors in one call; `MultiVectorEncoder` can't read BGE-M3's ColBERT head | no (doc names ST for ColBERT only) |
| Pin by `snapshot_download(repo_id, revision=sha)`, load from that folder | FlagEmbedding takes `main` otherwise; the sha goes in the manifest | no |
| `EncoderInfo.settings` dict | `max_length` and fp16 change vectors; queries must match the documents | yes: the doc's manifest lists model and revision only |
| `max_length=512` for docs and queries, fp16 on | what smve-lab's golden embeddings used (`EncodeConfig`) | no |
| Token vectors stored float16, dense float32 | same as smve-lab (`encode_texts`) | no |
| Text sent to the encoder is `"{title}. {text}"`, or `text` when the title is empty | smve-lab's `_join` (note: bm25 joins with a newline; the two are separate on purpose) | no |
| Torch imports only inside `BGEM3Encoder.__init__` | layer rule; `import laterank` must work without the extra | no |
| Default pipeline `hybrid > maxsim@100`, not tied to a model | the doc's BGE-M3 default; `create` refuses it clearly if a stage is missing | no |

## Acceptance tests

- [x] AT1: `EncoderInfo(settings=...)` round-trips through `write_manifest`/`read_manifest`; a
  non-JSON value and a manifest without `settings` raise (`ValueError`, `IndexFormatError`).
- [x] AT2: with a fake `Encoder` (hash-seeded numpy vectors, in `tests/`), `Index.build` makes an
  index whose stages are `dense`, `lexical`, `hybrid`; `search(encoder.encode_query(t))` runs
  `hybrid > maxsim@100`; a fake with no lexical head and the default pipeline raises `ValueError`
  and leaves no folder.
- [x] AT3: `load_encoder` raises `ValueError` for an unknown name and for a revision mismatch
  (fake `snapshot_download`, no network).
- [x] AT4: `import laterank` and every module outside `components/bgem3.py` import with no torch
  installed (`tests/test_layers.py` extended); `BGEM3Encoder()` without the extra raises an
  `ImportError` that says `pip install "laterank[bge-m3]"`.
- [x] AT5 (golden, `-m slow`): `BGEM3Encoder` on SciFact's first 20 documents: dense rows have
  cosine >= 0.999 with smve-lab's stored `dense`; token counts per document equal smve-lab's
  offsets; lexical term ids match. The 0.999 is my guess for fp16/device noise: tell me if you
  want it tighter or looser.
- [x] AT6: `uv run pytest -q` passes (321+ tests) and `uv run ruff check . && uv run ruff format
  --check .` is clean.
- [x] AT7: the `reviewer` agent returns `merge`.

## Tasks

| # | task | files it owns | depends on | parallel? |
|---|---|---|---|---|
| 1 | `EncoderInfo.settings` + manifest tests | `storage/manifest.py`, `tests/storage/test_manifest.py` | none | no |
| 2 | `Encoder`, `EncodedDocs`, `load_encoder`, fake encoder | `components/encoder.py`, `tests/components/test_encoder.py` | 1 | no |
| 3 | `BGEM3Encoder` + the extra | `components/bgem3.py`, `pyproject.toml`, `tests/test_layers.py` | 2 | yes, with 4 |
| 4 | `Index.build` | `orchestration/index.py`, `tests/orchestration/test_build.py` | 2 | yes, with 3 |
| 5 | golden test AT5 | `tests/golden/test_encoder.py` | 3, 4 | no |

## Changes

- 2026-11-03: task 2 also adds `tests/conftest.py` (a shared `FakeEncoder` and a
  `fake_encoder_cls` fixture) - task 4's tests need the same fake, and pytest's importlib mode
  can't import a sibling test file.
- 2026-11-03: `Index.build` checks the output folder before encoding (a `_check_free` helper
  shared with `create`) - encoding SciFact takes minutes, so a taken folder should fail first. A
  pipeline that needs a head the encoder lacks is still found only after encoding, when `create`
  checks it. Skipped: making encoders declare their heads, add when encoding time makes that hurt.
- 2026-11-03: reviewer round 1 ("merge", low findings), all fixed on the owner's "fix all":
  `BGEM3Encoder` refuses a revision that isn't a full sha (or None) *before* downloading, instead
  of comparing the folder name after; `load_encoder` refuses a `bge-m3` entry with no `hf_id`;
  `Index.build` refuses a non-string title or text with a `ValueError`; new
  `tests/components/test_bgem3.py` stubs FlagEmbedding and huggingface_hub, so the pinning rules
  (AT3's "fake snapshot_download") are tested without a model.
- 2026-11-03: AT5 widened, with the owner's "fix all": the golden test also encodes 5 queries and
  compares token vectors, dense and lexical weights with smve-lab's. Lexical is compared as
  weights within 0.002 (first 0.01; tightened after reviewer round 2: observed noise is 0.0004, and 0.01 let a 2-3% scaling bug pass), not as term ids: a term with weight 0.00014 ("to") was present in
  smve-lab's output and absent in ours (fp16 noise across the relu), so exact ids would fail by
  chance. Documents use the same rule.
- 2026-11-03: not fixed, noted: `EncoderInfo` is frozen but holds a dict, so `hash(info)` raises and
  `settings` can be mutated in place. Nothing hashes it. Fix when something needs to.
