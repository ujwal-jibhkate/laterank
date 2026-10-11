"""docs.jsonl: the documents of one segment, with byte offsets for random access.

Each document is one JSON object on one line:

    {"id": "doc-1", "title": "Statins", "text": "Statins lower ...", "metadata": {"year": 2019}}

JSON escapes newlines inside strings, so a document never spans two lines. Next to it,
`docs.offsets.npy` holds (n + 1,) int64 byte offsets: document i is the bytes
offsets[i]:offsets[i+1]. Fetching hit #4,000 is one seek and one small read, not a scan.

The file is written in binary mode as UTF-8 with "\\n" line endings. In text mode, Windows would
turn every "\\n" into "\\r\\n" and shift every offset after the first line.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from laterank.errors import IndexFormatError
from laterank.storage.arrays import _ragged_problem, load_array, save_array

DOCS = "docs.jsonl"
OFFSETS = "docs.offsets.npy"
_KEYS = ("id", "title", "text", "metadata")


def _record(doc: Mapping) -> dict:
    """Check one document and return it with every key, in a fixed order."""
    unknown = sorted(set(doc) - set(_KEYS))
    if unknown:
        raise ValueError(f"documents may only have the keys {list(_KEYS)}, got extra {unknown}")
    doc_id = doc.get("id")
    if not isinstance(doc_id, str) or not doc_id:
        raise ValueError(f"every document needs a non-empty string 'id', got {doc_id!r}")
    for key in ("title", "text"):
        if doc.get(key) is not None and not isinstance(doc[key], str):
            raise ValueError(f"document {doc_id!r}: '{key}' must be a string or None")
    metadata = doc.get("metadata")
    metadata = {} if metadata is None else metadata
    if not isinstance(metadata, Mapping):
        raise ValueError(f"document {doc_id!r}: 'metadata' must be a dict")
    return {
        "id": doc_id,
        "title": doc.get("title"),
        "text": doc.get("text"),
        "metadata": dict(metadata),
    }


def _line(rec: dict) -> bytes:
    """One document as a JSON line, refusing anything that wouldn't read back equal."""
    try:
        text = json.dumps(rec, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as e:  # an object, or NaN/inf
        raise ValueError(f"document {rec['id']!r}: metadata must be plain JSON ({e})") from None
    if json.loads(text) != rec:  # int keys become strings, tuples become lists
        raise ValueError(
            f"document {rec['id']!r}: metadata must be plain JSON (string keys, lists not tuples)"
        )
    return text.encode("utf-8")


def write_docs(folder: Path, docs: Iterable[Mapping]) -> int:
    """Write `folder/docs.jsonl` and `folder/docs.offsets.npy`.

    Args:
        docs: mappings with a unique string "id" and optional "title", "text" (str or None) and
            "metadata" (a JSON-serializable dict).

    Returns:
        The number of documents written.

    Raises:
        ValueError: a missing or duplicate id, an unknown key, a value of the wrong type, or
            metadata that wouldn't read back as written. No docs.jsonl is left behind.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    offsets, seen = [0], set()
    try:
        with open(folder / DOCS, "wb") as f:
            for doc in docs:
                rec = _record(doc)
                if rec["id"] in seen:
                    raise ValueError(f"duplicate document id {rec['id']!r}")
                seen.add(rec["id"])
                line = _line(rec)
                f.write(line + b"\n")
                offsets.append(offsets[-1] + len(line) + 1)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        (folder / DOCS).unlink(missing_ok=True)
        raise
    save_array(folder / OFFSETS, np.array(offsets, dtype=np.int64))
    return len(offsets) - 1


class DocStore:
    """Read-only, random access to one segment's documents.

    docs.jsonl isn't held open between reads. (Arrays memory-mapped from the same segment
    still hold their files open, and Windows refuses to rename or delete an open file.)
    """

    def __init__(self, folder: Path):
        self.path = Path(folder) / DOCS
        self._offsets = load_array(Path(folder) / OFFSETS, mmap=False)
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            raise IndexFormatError(f"{self.path} is missing. Rebuild the index.") from None
        if _ragged_problem(size, self._offsets):  # byte offsets follow the ragged-offsets rules
            raise IndexFormatError(
                f"{folder}/{OFFSETS} doesn't match {DOCS} ({size} bytes). Rebuild the index."
            )

    def __len__(self) -> int:
        return len(self._offsets) - 1

    def __getitem__(self, row: int) -> dict:
        n = len(self)
        if not -n <= row < n:
            raise IndexError(f"document row {row} out of range for {n} documents")
        row %= n
        start, end = int(self._offsets[row]), int(self._offsets[row + 1])
        with open(self.path, "rb") as f:
            f.seek(start)
            data = f.read(end - start)
        try:
            doc = json.loads(data.decode("utf-8"))
        except (ValueError, RecursionError):  # bad UTF-8/JSON, ints over 4300 digits
            doc = None
        if not isinstance(doc, dict):
            raise IndexFormatError(
                f"{self.path}: document row {row} is corrupt. Rebuild the index."
            )
        return doc

    def ids(self) -> list[str]:
        """Every document id, in row order."""
        with open(self.path, "rb") as f:
            try:
                return [json.loads(line)["id"] for line in f]
            except (ValueError, KeyError, TypeError, RecursionError) as e:
                raise IndexFormatError(
                    f"{self.path} is corrupt ({e!r}). Rebuild the index."
                ) from None
