"""manifest.json: the index's "recipe card", written atomically and read strictly.

The manifest says everything about how an index was built: the encoder and its exact Hugging
Face revision (so queries are encoded by the same model as the documents), the stages that were
built, the default pipeline, and the list of segments. Opening an index starts here.

Crash safety. The manifest is written LAST, and atomically:

  1. serialize the whole manifest to bytes in memory (a bug here never touches the disk)
  2. write them to a temp file next to it, flush, and fsync (the bytes are really on disk)
  3. os.replace(temp, "manifest.json"): an atomic swap on POSIX and Windows, so a reader sees
     either the old file or the new one, never half of one
  4. fsync the folder (POSIX), so the swap itself survives a power cut

Readers only open the segments the manifest lists, so a crash at any point leaves the previous,
complete index. A crash between steps 2 and 3 leaves a stray `manifest.json.tmp-*` file, which
readers ignore.

Strictness. `read_manifest` refuses anything it doesn't fully understand: a newer
`format_version`, missing or unknown keys, wrong types. `format_version` is the only way the
format grows. Format version 1 is frozen at the first PyPI release.
"""

from __future__ import annotations

import json
import os
import re
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from laterank._version import __version__
from laterank.errors import IndexFormatError
from laterank.storage._fsync import fsync_folder

FORMAT = "laterank-index"
FORMAT_VERSION = 1
MANIFEST = "manifest.json"

_TOP_KEYS = (
    "format",
    "format_version",
    "created_with",
    "created_at",
    "encoder",
    "stages",
    "default_pipeline",
    "segments",
)
_ENCODER_KEYS = ("name", "hf_id", "revision", "dim")
_SEGMENT_KEYS = ("id", "documents", "tokens")
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_SEGMENT_ID = re.compile(r"seg-\d{6}")
_FIX = "The manifest is corrupt or was edited by hand: restore it or rebuild the index."


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)  # True is an int in Python


@dataclass(frozen=True)
class EncoderInfo:
    """Which model encoded the documents."""

    name: str  # alias, e.g. "bge-m3"
    hf_id: str | None  # e.g. "BAAI/bge-m3"; None for precomputed vectors
    revision: str | None  # 40-hex commit sha; required exactly when hf_id is set
    dim: int  # token vector dimension

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("encoder name must be a non-empty string")
        if self.hf_id is not None and (not isinstance(self.hf_id, str) or not self.hf_id):
            raise ValueError("encoder hf_id must be a non-empty string or None")
        if self.hf_id is None and self.revision is not None:
            raise ValueError("encoder revision is set but hf_id is not")
        if self.hf_id is not None and not (
            isinstance(self.revision, str) and _COMMIT_SHA.fullmatch(self.revision)
        ):
            raise ValueError(
                f"encoder revision must be a full 40-character commit sha for {self.hf_id}, "
                f"got {self.revision!r} (a branch like 'main' isn't pinned: it moves)"
            )
        if not _is_int(self.dim) or self.dim <= 0:
            raise ValueError(f"encoder dim must be a positive integer, got {self.dim!r}")


@dataclass(frozen=True)
class SegmentInfo:
    """One segment folder the index includes."""

    id: str  # "seg-000001"
    documents: int
    tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _SEGMENT_ID.fullmatch(self.id):
            raise ValueError(f"segment id must look like 'seg-000001', got {self.id!r}")
        for name in ("documents", "tokens"):
            value = getattr(self, name)
            if not _is_int(value) or value < 0:
                raise ValueError(f"segment {name} must be a non-negative integer, got {value!r}")


@dataclass(frozen=True)
class Manifest:
    """Everything needed to reopen an index. See the module docstring for the rules."""

    encoder: EncoderInfo
    stages: dict[str, dict]  # name -> {"type": str, ...}; opaque to storage
    default_pipeline: str  # e.g. "hybrid > maxsim@100"; parsed in M1
    segments: tuple[SegmentInfo, ...]
    created_with: str = field(default_factory=lambda: f"laterank {__version__}")
    created_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", tuple(self.segments))  # accept any sequence
        if not isinstance(self.encoder, EncoderInfo):
            raise ValueError("encoder must be an EncoderInfo")
        if not isinstance(self.stages, dict):
            raise ValueError("stages must be a dict of stage name -> settings")
        for name, stage in self.stages.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"stage names must be non-empty strings, got {name!r}")
            if not isinstance(stage, dict) or not isinstance(stage.get("type"), str):
                raise ValueError(f"stage {name!r} needs a 'type' string")
        if not isinstance(self.default_pipeline, str) or not self.default_pipeline.strip():
            raise ValueError("default_pipeline must be a non-empty string")
        if not all(isinstance(s, SegmentInfo) for s in self.segments):
            raise ValueError("segments must be SegmentInfo objects")
        ids = [s.id for s in self.segments]
        if len(set(ids)) != len(ids):
            raise ValueError(f"segment ids must be unique, got {ids}")
        for name in ("created_with", "created_at"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")

    def to_dict(self) -> dict:
        """The manifest as JSON-ready data, keys in a fixed order."""
        return {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "created_with": self.created_with,
            "created_at": self.created_at,
            "encoder": {k: getattr(self.encoder, k) for k in _ENCODER_KEYS},
            "stages": self.stages,
            "default_pipeline": self.default_pipeline,
            "segments": [{k: getattr(s, k) for k in _SEGMENT_KEYS} for s in self.segments],
        }

    @classmethod
    def from_dict(cls, d: Any) -> Manifest:
        """Validate parsed manifest JSON and build a Manifest.

        Raises:
            IndexFormatError: anything this release doesn't fully understand.
        """
        if not isinstance(d, dict) or d.get("format") != FORMAT:
            raise IndexFormatError(
                f'this is not a laterank index (expected "format": "{FORMAT}"). Check that the '
                "path points at a laterank index folder, or rebuild the index."
            )
        writer = d.get("created_with", "an unknown laterank version")
        version = d.get("format_version")
        if not _is_int(version) or version < 1:
            raise IndexFormatError(
                f"format_version must be a positive integer, got {version!r}. {_FIX}"
            )
        if version > FORMAT_VERSION:
            raise IndexFormatError(
                f"this index was written by {writer} in format {version}, but this laterank "
                f"({__version__}) reads format {FORMAT_VERSION} only. "
                "Upgrade laterank: pip install -U laterank"
            )
        _exact_keys(d, _TOP_KEYS, "the manifest", writer)
        encoder, segments = d["encoder"], d["segments"]
        if not isinstance(encoder, dict) or not isinstance(segments, list):
            raise IndexFormatError(f"'encoder' must be an object and 'segments' a list. {_FIX}")
        _exact_keys(encoder, _ENCODER_KEYS, "'encoder'", writer)
        for s in segments:
            if not isinstance(s, dict):
                raise IndexFormatError(f"every entry of 'segments' must be an object. {_FIX}")
            _exact_keys(s, _SEGMENT_KEYS, "a segment entry", writer)
        try:
            return cls(
                encoder=EncoderInfo(**encoder),
                stages=d["stages"],
                default_pipeline=d["default_pipeline"],
                segments=tuple(SegmentInfo(**s) for s in segments),
                created_with=d["created_with"],
                created_at=d["created_at"],
            )
        except ValueError as e:
            raise IndexFormatError(f"{e}. {_FIX}") from None


def _exact_keys(d: dict, expected: tuple[str, ...], what: str, writer: str) -> None:
    missing = [k for k in expected if k not in d]
    unknown = sorted(k for k in d if k not in expected)
    if missing:
        raise IndexFormatError(f"{what} is missing {missing}. {_FIX}")
    if unknown:
        raise IndexFormatError(
            f"{what} has keys this laterank ({__version__}) doesn't know: {unknown}. It was "
            f"written by {writer}; upgrade laterank (pip install -U laterank) or rebuild the index."
        )


def write_manifest(index_dir: Path, manifest: Manifest) -> None:
    """Atomically write `index_dir/manifest.json` (see the module docstring).

    Either the new manifest is in place when this returns, or the old one is untouched.

    Raises:
        ValueError: the manifest wouldn't read back as written (nothing is written).
    """
    index_dir = Path(index_dir)
    try:
        text = json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False, allow_nan=False)
    except TypeError as e:  # e.g. np.int64 or an object inside stages
        raise ValueError(f"stages may hold only JSON values ({e})") from None
    try:  # never write a manifest that read_manifest would refuse or read differently
        same = Manifest.from_dict(json.loads(text)) == manifest
    except IndexFormatError:
        same = False
    if not same:
        raise ValueError(
            "this manifest wouldn't read back as written: stages may hold only JSON values "
            "(string keys, lists not tuples) and must not be changed after the Manifest is built"
        )
    data = (text + "\n").encode("utf-8")
    index_dir.mkdir(parents=True, exist_ok=True)
    tmp = index_dir / f"{MANIFEST}.tmp-{secrets.token_hex(4)}"
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, index_dir / MANIFEST)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    fsync_folder(index_dir)


def read_manifest(index_dir: Path) -> Manifest:
    """Read and validate `index_dir/manifest.json`.

    Raises:
        IndexFormatError: the manifest is missing, not valid JSON, or not fully understood.
    """
    path = Path(index_dir) / MANIFEST
    if not path.is_file():  # missing, a folder, or index_dir is a file (errors differ by OS)
        raise IndexFormatError(
            f"{index_dir} has no {MANIFEST} file: it isn't a laterank index, or its build never "
            "finished. Build the index again."
        )
    raw = path.read_bytes()
    try:
        d = json.loads(raw.decode("utf-8"))
    except (ValueError, RecursionError) as e:  # bad UTF-8/JSON, ints over 4300 digits
        raise IndexFormatError(
            f"{path} is not valid JSON ({e}); it may be truncated. {_FIX}"
        ) from None
    try:
        return Manifest.from_dict(d)
    except IndexFormatError as e:
        raise IndexFormatError(f"{path}: {e}") from None
