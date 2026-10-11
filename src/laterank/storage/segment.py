"""Segments: one folder per batch of documents, written invisibly and then renamed into place.

    my_index/
      manifest.json
      seg-000001/
        docs.jsonl, docs.offsets.npy     the documents (storage/docs.py)
        tokens/flat.npy, offsets.npy     token vectors, a ragged pair (storage/arrays.py)
        dense/vectors.npy ...            whatever the stages store, as relative .npy paths

Crash safety. A segment is written into `seg-000001.tmp-<random>/`, every file and folder is
fsynced, and only then is the folder renamed to `seg-000001/`. A rename within one folder is
atomic, so nobody ever sees a half-written `seg-000001/`. The manifest is written after that and
lists the segment, and readers open ONLY listed segments (`open_index`). So a crash leaves at
worst a stray `.tmp-` folder, or a complete segment the manifest doesn't list yet; both are
ignored.

Storage knows the layout, not the methods: it stores whatever arrays the stages hand it.
"""

from __future__ import annotations

import os
import secrets
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

import numpy as np

from laterank.errors import IndexFormatError
from laterank.storage._fsync import fsync_folder
from laterank.storage.arrays import load_array, save_array
from laterank.storage.docs import DOCS, OFFSETS, DocStore, write_docs
from laterank.storage.manifest import _SEGMENT_ID, Manifest, SegmentInfo, read_manifest

TOKENS_FLAT = "tokens/flat.npy"


def segment_name(n: int) -> str:
    """1 -> "seg-000001"."""
    if not 1 <= n <= 999_999:
        raise ValueError(f"segment numbers run from 1 to 999999, got {n}")
    return f"seg-{n:06d}"


def _check_relpath(relpath: str) -> PurePosixPath:
    """An array's path inside a segment: relative, forward slashes, no '..', ending in .npy."""
    # ponytail: folders differing only by case ("Dense/" vs "dense/") and Windows-unsafe names
    # ("CON.npy", "a./x.npy") pass; fine while paths come from laterank's own stages (M1).
    # A path that is also another path's folder ("a.npy", "a.npy/b.npy") fails mid-write with
    # OSError (cleaned up). Check per path component if users ever choose array paths.
    p = PurePosixPath(relpath) if isinstance(relpath, str) else None
    if (
        p is None
        or "\\" in relpath
        or ":" in relpath  # Windows drives ("C:/x", "C:x") and NTFS streams ("x.npy:s")
        or p.is_absolute()
        or ".." in p.parts
        or p.suffix != ".npy"
        or str(p) != relpath  # rejects "", "./x.npy", "a//b.npy"
    ):
        raise ValueError(
            f"array paths must be relative 'folder/name.npy' paths without '..', got {relpath!r}"
        )
    if relpath.casefold() in (DOCS, OFFSETS):  # macOS and Windows ignore case
        raise ValueError(f"{relpath!r} is reserved for the documents")
    return p


def write_segment(
    index_dir: Path, n: int, arrays: Mapping[str, np.ndarray], docs: Sequence[Mapping]
) -> SegmentInfo:
    """Write segment number `n` crash-safely (see the module docstring).

    Args:
        arrays: relative path (e.g. "tokens/flat.npy") -> array.
        docs: documents for `write_docs`.

    Returns:
        SegmentInfo with the document count, and tokens = len(arrays["tokens/flat.npy"]) when
        that array is present, else 0. Add it to the manifest to make the segment visible.

    Raises:
        FileExistsError: segment `n` already exists.
        ValueError: a bad array path or document (nothing is left on disk).
    """
    index_dir = Path(index_dir)
    name = segment_name(n)
    final = index_dir / name
    for relpath in arrays:  # check everything before writing anything
        _check_relpath(relpath)
    if len({r.casefold() for r in arrays}) != len(arrays):  # one file on macOS and Windows
        raise ValueError(f"array paths must differ by more than case, got {sorted(arrays)}")
    flat = arrays.get(TOKENS_FLAT)
    if flat is not None and np.ndim(flat) == 0:
        raise ValueError(f"{TOKENS_FLAT} needs one row per token, got a single number")
    tokens = 0 if flat is None else len(flat)
    if final.exists():
        raise FileExistsError(f"{final} already exists; segments are never overwritten")

    index_dir.mkdir(parents=True, exist_ok=True)
    tmp = index_dir / f"{name}.tmp-{secrets.token_hex(4)}"
    tmp.mkdir()
    try:
        for relpath, arr in arrays.items():
            save_array(tmp / relpath, arr)
        count = write_docs(tmp, docs)
        for dirpath, _, _ in os.walk(tmp, topdown=False):  # files were fsynced when written
            fsync_folder(Path(dirpath))
        os.rename(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    fsync_folder(index_dir)
    return SegmentInfo(id=name, documents=count, tokens=tokens)


class Segment:
    """One segment folder, opened read-only. Arrays are loaded on demand."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.name = self.folder.name
        self.docs = DocStore(self.folder)

    def array(self, relpath: str, mmap: bool = True) -> np.ndarray:
        """Load one of the segment's arrays (memory-mapped by default)."""
        return load_array(self.folder / _check_relpath(relpath), mmap=mmap)


def open_segment(index_dir: Path, name: str) -> Segment:
    """Open the segment folder `name` (e.g. "seg-000001") inside `index_dir`.

    Raises:
        ValueError: `name` isn't a segment name.
        IndexFormatError: the folder or its documents are missing or corrupt.
    """
    if not isinstance(name, str) or not _SEGMENT_ID.fullmatch(name):
        raise ValueError(f"segment names look like 'seg-000001', got {name!r}")
    folder = Path(index_dir) / name
    if not folder.is_dir():
        raise IndexFormatError(f"{folder} is missing. The index is damaged; rebuild it.")
    return Segment(folder)


def open_index(index_dir: Path) -> tuple[Manifest, list[Segment]]:
    """Read the manifest and open exactly the segments it lists, in its order.

    Folders the manifest doesn't list (leftovers from a crash) are ignored.

    Raises:
        IndexFormatError: a bad manifest, a listed segment that is missing, or one whose
            document count disagrees with the manifest.
    """
    manifest = read_manifest(index_dir)
    segments = []
    for info in manifest.segments:
        seg = open_segment(index_dir, info.id)
        if len(seg.docs) != info.documents:
            raise IndexFormatError(
                f"{seg.folder} holds {len(seg.docs)} documents, but the manifest says "
                f"{info.documents}. The index is damaged; rebuild it."
            )
        segments.append(seg)
    return manifest, segments
