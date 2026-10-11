"""Saving and loading `.npy` arrays safely, and ragged (flat, offsets) pairs.

Every array in an index is a plain `.npy` file: a small header (dtype, shape) followed by the raw
bytes. Any tool that reads numpy can open it, and it can be memory-mapped: `np.load(...,
mmap_mode="r")` maps the file into memory without reading it, so opening a 3.8 GB token store
costs almost no RAM and MaxSim pages in only the rows it touches.

No pickles, ever. A `.npy` file can also hold "object" arrays, which numpy stores with Python's
pickle, and unpickling a file can run arbitrary code. So `load_array` always passes
`allow_pickle=False` and turns numpy's refusal into an `IndexFormatError` naming the file, and
`save_array` refuses to write object arrays in the first place.

Ragged arrays. Texts have different numbers of tokens, so token vectors are stored as one tall
`flat` array plus `offsets`, where item i owns rows offsets[i]:offsets[i+1] (see
kernels/maxsim.py). They live in one folder as `flat.npy` and `offsets.npy`.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from laterank.errors import IndexFormatError


def save_array(path: Path, arr: np.ndarray) -> None:
    """Write `arr` to `path` as a `.npy` file and flush it to disk (creates parent folders).

    Raises:
        ValueError: `arr` has an object dtype (it could only be stored as a pickle).
    """
    arr = np.asarray(arr)
    if arr.dtype.hasobject:
        raise ValueError(
            f"refusing to save an object-dtype array to {path}: it would be stored as a pickle. "
            "Convert it to a numeric dtype first."
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.save(f, arr, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())  # on disk before a segment folder is renamed into place


def load_array(path: Path, mmap: bool = True) -> np.ndarray:
    """Open a `.npy` file, never unpickling anything.

    Args:
        path: the `.npy` file.
        mmap: return a read-only `np.memmap` (default) instead of reading the file into memory.

    Raises:
        IndexFormatError: the file is missing, holds pickled data or an object array, or is not
            a valid `.npy` file.
    """
    path = Path(path)
    try:
        arr = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
    except FileNotFoundError:
        raise IndexFormatError(
            f"{path} is missing, so the index is incomplete. Rebuild the index."
        ) from None
    except MemoryError as e:  # mmap=False and the header claims a huge shape
        raise IndexFormatError(
            f"{path} is too large to read into memory ({e}). Open it with mmap=True; if the file "
            "itself is small, its header is corrupt: rebuild the index."
        ) from None
    except (ValueError, EOFError, OSError) as e:
        if "pickle" in str(e) or "Python objects" in str(e) or "Object arrays" in str(e):
            raise IndexFormatError(
                f"{path} contains pickled Python objects. laterank never loads pickles, because "
                "opening one can run arbitrary code. Rebuild the index with laterank."
            ) from None
        raise IndexFormatError(
            f"{path} is not a valid .npy file ({e}). It may be truncated; rebuild the index."
        ) from None
    if not isinstance(arr, np.ndarray):  # np.load returns an NpzFile for zip archives
        arr.close()  # Windows locks the file while it is open
        raise IndexFormatError(f"{path} is not a .npy array (it looks like a .npz archive).")
    return arr


def _ragged_problem(n_rows: int, offsets: np.ndarray) -> str | None:
    """Why (flat with n_rows rows, offsets) isn't a valid ragged pair, or None if it is."""
    if offsets.ndim != 1 or offsets.dtype != np.int64:
        return f"offsets must be a 1-D int64 array, got {offsets.dtype} with shape {offsets.shape}"
    if len(offsets) == 0 or offsets[0] != 0:
        return "offsets must start at 0"
    if np.any(np.diff(offsets) < 0):
        return "offsets must never decrease"
    if offsets[-1] != n_rows:
        return f"offsets end at {offsets[-1]}, but flat has {n_rows} rows"
    return None


def save_ragged(folder: Path, flat: np.ndarray, offsets: np.ndarray) -> None:
    """Write a ragged pair as `folder/flat.npy` and `folder/offsets.npy` (offsets as int64).

    Raises:
        ValueError: the offsets don't describe `flat` (see `load_ragged` for the rules).
    """
    offsets = np.asarray(offsets)
    if np.issubdtype(offsets.dtype, np.integer):
        offsets = offsets.astype(np.int64)
    problem = _ragged_problem(len(flat), offsets)
    if problem:
        raise ValueError(f"not a valid ragged array: {problem}")
    folder = Path(folder)
    save_array(folder / "flat.npy", flat)
    save_array(folder / "offsets.npy", offsets)


def load_ragged(folder: Path, mmap: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Open a ragged pair saved by `save_ragged`.

    `flat` is memory-mapped when `mmap` is true; `offsets` (one int64 per item) is always read
    into memory, since it is small and has to be checked anyway.

    Returns:
        (flat, offsets): offsets is 1-D int64, starts at 0, never decreases, and ends at len(flat).

    Raises:
        IndexFormatError: a file is missing or unreadable, or the offsets break those rules.
    """
    folder = Path(folder)
    flat = load_array(folder / "flat.npy", mmap=mmap)
    offsets = load_array(folder / "offsets.npy", mmap=False)
    if flat.ndim == 0:
        raise IndexFormatError(f"{folder}/flat.npy holds a single number, not rows. Rebuild it.")
    problem = _ragged_problem(len(flat), offsets)
    if problem:
        raise IndexFormatError(f"{folder} holds a corrupt ragged array: {problem}. Rebuild it.")
    return flat, offsets
