---
id: M0-3
title: Storage
status: approved
approved: 2026-10-03
---

# M0-3: Storage

## Goal

The on-disk index format, so that M1's Index can save everything it builds and reopen it later
without re-encoding. An index is a plain folder: a `manifest.json` "recipe card", plus segment
folders of `.npy` arrays (memory-mapped, never pickled) and a `docs.jsonl` with byte offsets.
Writes are crash-safe: a crash leaves the previous index intact. Storage knows file formats,
not methods.

## Out of scope

- Postings triplets, `stats/`, per-stage folders (`dense/`, `bm25/`, `smve-*/`): M1, with the
  stages that write them. Storage only gives them `save_array`/`load_array`.
- `LOCK` files, add/delete, tombstones, `compact`: 0.2 (single writer until then).
- Checking `anchor_sha256` at open: the SMVE stage does it in M1 (storage treats stages as opaque).
- Creating or overwriting an index folder (`IndexExistsError`), chunking, `passages.npy`: M1/M3.
- Migrating older format versions: only version 1 exists.

## Interfaces

```python
# laterank/errors.py  (leaf module: imports only the stdlib; every layer may import it)
class LaterankError(Exception): ...           # base of every laterank error
class IndexFormatError(LaterankError): ...    # missing/corrupt files, pickled data, bad manifest

# laterank/_version.py  (leaf) __version__: str; laterank/__init__.py re-exports it

# storage/manifest.py
FORMAT = "laterank-index"; FORMAT_VERSION = 1; MANIFEST = "manifest.json"
@dataclass(frozen=True)
class EncoderInfo:
    name: str               # alias, e.g. "bge-m3"
    hf_id: str | None       # e.g. "BAAI/bge-m3"; None for precomputed vectors
    revision: str | None    # 40-hex commit sha; required exactly when hf_id is set
    dim: int                # token vector dimension
@dataclass(frozen=True)
class SegmentInfo:
    id: str                 # "seg-000001"
    documents: int
    tokens: int
@dataclass(frozen=True)
class Manifest:
    encoder: EncoderInfo
    stages: dict[str, dict]         # name -> {"type": str, ...}; opaque to storage
    default_pipeline: str           # e.g. "hybrid > maxsim@100"; parsed in M1
    segments: tuple[SegmentInfo, ...]
    format_version: int = FORMAT_VERSION
    created_with: str = "laterank <__version__>"
    created_at: str = "<UTC now, ISO-8601, e.g. 2026-10-06T12:00:00Z>"
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "Manifest": ...   # validates; IndexFormatError
def write_manifest(index_dir: Path, manifest: Manifest) -> None   # atomic
def read_manifest(index_dir: Path) -> Manifest                     # IndexFormatError

# storage/arrays.py
def save_array(path: Path, arr: np.ndarray) -> None    # ValueError for object dtype
def load_array(path: Path, mmap: bool = True) -> np.ndarray
    # always allow_pickle=False; mmap=True -> read-only np.memmap; IndexFormatError naming the
    # file when it's missing, pickled, object-dtype or not a valid .npy
def save_ragged(folder: Path, flat: np.ndarray, offsets: np.ndarray) -> None  # flat.npy, offsets.npy
def load_ragged(folder: Path, mmap: bool = True) -> tuple[np.ndarray, np.ndarray]
    # offsets: 1-D int64, starts at 0, non-decreasing, ends at len(flat); else IndexFormatError

# storage/docs.py
def write_docs(folder: Path, docs: Iterable[Mapping]) -> int  # docs.jsonl + docs.offsets.npy; returns count
class DocStore:                       # opened per segment
    def __init__(self, folder: Path): ...
    def __len__(self) -> int: ...
    def __getitem__(self, row: int) -> dict: ...   # seeks to the row's byte offset
    def ids(self) -> list[str]: ...
# one JSON object per line: {"id": str, "title": str | None, "text": str | None, "metadata": dict}
# docs.offsets.npy: (n + 1,) int64 byte offsets into docs.jsonl

# storage/segment.py
def segment_name(n: int) -> str     # 1 -> "seg-000001"
def write_segment(index_dir: Path, n: int, arrays: Mapping[str, np.ndarray],
                  docs: Sequence[Mapping]) -> SegmentInfo
    # array keys are relative paths like "tokens/flat.npy"; written into
    # seg-000001.tmp-<random>/, fsynced, then renamed to seg-000001/. FileExistsError if it exists.
    # SegmentInfo.tokens = len(arrays["tokens/flat.npy"]) when present, else 0.
class Segment:
    name: str; docs: DocStore
    def array(self, relpath: str, mmap: bool = True) -> np.ndarray: ...
def open_segment(index_dir: Path, name: str) -> Segment
def open_index(index_dir: Path) -> tuple[Manifest, list[Segment]]
    # opens ONLY the segments the manifest lists; IndexFormatError if one is missing or its
    # document count disagrees with the manifest
```

## Decisions

| decision | why | differs from design doc? |
|---|---|---|
| `errors.py` and `_version.py` are leaf modules at the package root | every layer needs the errors; storage needs the version without importing `laterank/__init__` (which will import the api layer). `MvrError` is renamed `LaterankError` | no (design has `errors.py` at the root) |
| manifest holds the four requested fields plus `format`, `created_with`, `created_at`, `segments` | `segments` is the commit point: readers see only listed segments. The design's `chunking`, `documents`, `fingerprint` arrive with the features that fill them | partly: fewer fields for now |
| format version 1 is frozen at the first PyPI release, not before | M1 will add fields; until a release, nobody has v1 indexes to break | no |
| strict validation written by hand; unknown top-level keys are rejected | no `jsonschema` dependency; `format_version` is the only way the format grows | **yes**: the design ships a JSON Schema file |
| `revision` must be a full 40-hex commit sha whenever `hf_id` is set | "main" isn't pinned; the design's point is that queries use the exact model revision | no |
| storage checks only that each stage has a `"type"` string | storage knows formats, not methods | no |
| manifest: temp file in the same folder → flush → fsync → `os.replace`, then fsync the folder (POSIX only; Windows can't open folders) | `os.replace` is atomic on POSIX and Windows; the manifest is written last, so a crash before it leaves the old index | no |
| segments are written to `seg-XXXXXX.tmp-<random>/`, then renamed | a reader can never see a half-written segment | no |
| `docs.jsonl` is written in binary mode, UTF-8, `ensure_ascii=False`, `\n` endings | Windows text mode would turn `\n` into `\r\n` and break the byte offsets | no |
| no `LOCK` file yet | only one writer exists until add/delete in 0.2 | **yes**: deferred |
| a newer `format_version` → `IndexFormatError` quoting the manifest's `created_with` | we can't know the future release that wrote it, but the manifest says which one did | no (refines it) |

## Acceptance tests

- [ ] AT1: `write_manifest` → `read_manifest` returns an equal `Manifest`; the JSON on disk has
  exactly the documented keys.
- [ ] AT2: `read_manifest` raises `IndexFormatError` with a message saying what to do for each of:
  a newer `format_version` (names `created_with`, says to upgrade), a missing or non-integer
  version, a wrong `format`, truncated/invalid JSON, a missing or unknown key, and `hf_id`
  without a 40-hex `revision`.
- [ ] AT3: a half-written manifest never replaces a good one: a failure during serialization, and
  one just before `os.replace`, leave the old file byte-identical with no temp file left; a
  subprocess hard-killed (`os._exit`) between writing the temp file and the swap leaves the old
  manifest readable.
- [ ] AT4: arrays round-trip exactly for float16, float32, int32 and int64, including empty
  arrays; `load_array(mmap=True)` returns a read-only `np.memmap`.
- [ ] AT5: pickled data is rejected: an object-dtype `.npy` and a raw pickle saved as `.npy` both
  raise `IndexFormatError`, with mmap on and off; `save_array` refuses object dtype;
  `grep -rn "allow_pickle=True" src/` finds nothing.
- [ ] AT6: docs round-trip with non-ASCII text, newlines inside text, `None` text and nested
  metadata; reading row i via its byte offset returns exactly document i.
- [ ] AT7: `load_ragged` rejects offsets that don't start at 0, decrease, or don't end at
  `len(flat)`.
- [ ] AT8: `write_segment` → `open_segment` round-trips; a failure mid-write leaves no
  `seg-XXXXXX/` folder; `open_index` ignores unlisted segment folders and raises
  `IndexFormatError` for a listed one that is missing.
- [ ] AT9: `tests/test_layers.py` also checks that `errors.py` and `_version.py` import only the
  stdlib; `grep -rn "import torch" src/laterank/storage` finds nothing; `uv run pytest -q` and
  `uv run ruff check .` pass on all three CI operating systems.
- [ ] AT10: the reviewer agent's verdict is `merge`.

## Tasks

| # | task | files it owns | depends on | parallel? |
|---|---|---|---|---|
| 1 | errors, version, arrays + ragged | `src/laterank/{errors,_version,__init__}.py`, `storage/arrays.py`, `tests/storage/test_arrays.py`, `tests/test_layers.py` | M0-2 | no (others need it) |
| 2 | manifest | `storage/manifest.py`, `tests/storage/test_manifest.py` | 1 | yes, with 3 |
| 3 | docs store | `storage/docs.py`, `tests/storage/test_docs.py` | 1 | yes, with 2 |
| 4 | segments + `open_index` | `storage/segment.py`, `tests/storage/test_segment.py` | 2, 3 | no |

## Changes

- 2026-10-03: approved as drafted: unknown manifest keys are rejected, one checkpoint (not
  split), no Windows retry loop for `os.replace` unless CI shows a need.
- 2026-10-03: drafted.
