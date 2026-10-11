"""Safe .npy I/O: exact round trips, memory maps, and no pickles under any circumstances."""

import pickle

import numpy as np
import pytest

from laterank.errors import IndexFormatError
from laterank.storage.arrays import load_array, load_ragged, save_array, save_ragged


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.int32, np.int64])
@pytest.mark.parametrize("shape", [(7,), (5, 3), (0,), (0, 4)])
def test_round_trip_is_exact(tmp_path, dtype, shape):
    rng = np.random.default_rng(0)
    arr = (rng.standard_normal(shape) * 100).astype(dtype)
    save_array(tmp_path / "sub" / "a.npy", arr)  # parent folders are created
    for mmap in (True, False):
        got = load_array(tmp_path / "sub" / "a.npy", mmap=mmap)
        assert got.dtype == arr.dtype and got.shape == arr.shape
        assert np.array_equal(got, arr)


def test_mmap_is_read_only_and_optional(tmp_path):
    save_array(tmp_path / "a.npy", np.arange(10, dtype=np.float32))
    mm = load_array(tmp_path / "a.npy")
    assert isinstance(mm, np.memmap)
    with pytest.raises(ValueError):  # read-only: an index is never modified in place
        mm[0] = 1
    plain = load_array(tmp_path / "a.npy", mmap=False)
    assert not isinstance(plain, np.memmap)


@pytest.mark.parametrize("mmap", [True, False])
def test_object_array_is_rejected(tmp_path, mmap):
    path = tmp_path / "obj.npy"
    np.save(path, np.array([{"a": 1}], dtype=object), allow_pickle=True)  # a hostile file
    with pytest.raises(IndexFormatError, match="pickled"):
        load_array(path, mmap=mmap)


@pytest.mark.parametrize("mmap", [True, False])
def test_raw_pickle_named_npy_is_rejected(tmp_path, mmap):
    path = tmp_path / "evil.npy"
    path.write_bytes(pickle.dumps([1, 2, 3]))
    with pytest.raises(IndexFormatError, match="pickled"):
        load_array(path, mmap=mmap)


def test_save_refuses_object_arrays(tmp_path):
    with pytest.raises(ValueError, match="pickle"):
        save_array(tmp_path / "obj.npy", np.array(["a", None], dtype=object))
    assert not (tmp_path / "obj.npy").exists()


def test_missing_truncated_and_npz_files(tmp_path):
    with pytest.raises(IndexFormatError, match="missing"):
        load_array(tmp_path / "nope.npy")

    save_array(tmp_path / "a.npy", np.arange(1000, dtype=np.float32))
    data = (tmp_path / "a.npy").read_bytes()
    (tmp_path / "cut.npy").write_bytes(data[: len(data) // 2])
    for mmap in (True, False):
        with pytest.raises(IndexFormatError, match=r"not a valid \.npy"):
            load_array(tmp_path / "cut.npy", mmap=mmap)

    np.savez(tmp_path / "z.npz", a=np.arange(3))
    (tmp_path / "z.npz").rename(tmp_path / "z.npy")
    with pytest.raises(IndexFormatError, match="npz"):
        load_array(tmp_path / "z.npy", mmap=False)


def test_header_claiming_a_huge_shape_is_an_index_error(tmp_path):
    path = tmp_path / "huge.npy"
    with open(path, "wb") as f:  # 16 bytes of data, but the header claims ~73 TiB
        np.lib.format.write_array_header_1_0(
            f, {"descr": "<f8", "fortran_order": False, "shape": (10**13,)}
        )
        f.write(bytes(16))
    for mmap in (True, False):
        with pytest.raises(IndexFormatError):
            load_array(path, mmap=mmap)


# --- Ragged pairs -----------------------------------------------------------------------------


def test_ragged_round_trip(tmp_path):
    flat = np.random.default_rng(1).standard_normal((9, 4)).astype(np.float16)
    offsets = np.array([0, 2, 2, 9], dtype=np.int32)  # empty items are allowed; stored as int64
    save_ragged(tmp_path / "tokens", flat, offsets)
    got_flat, got_off = load_ragged(tmp_path / "tokens")
    assert isinstance(got_flat, np.memmap) and np.array_equal(got_flat, flat)
    assert got_off.dtype == np.int64 and got_off.tolist() == [0, 2, 2, 9]

    save_ragged(tmp_path / "empty", flat[:0], np.array([0]))
    got_flat, got_off = load_ragged(tmp_path / "empty")
    assert got_flat.shape == (0, 4) and got_off.tolist() == [0]


BAD_OFFSETS = {
    "doesn't start at 0": [1, 3, 9],
    "decreases": [0, 5, 3, 9],
    "ends short": [0, 2, 8],
    "ends past": [0, 2, 10],
    "empty": [],
}


@pytest.mark.parametrize("offsets", BAD_OFFSETS.values(), ids=BAD_OFFSETS.keys())
def test_bad_offsets_are_rejected_on_load_and_save(tmp_path, offsets):
    flat = np.zeros((9, 2), dtype=np.float32)
    save_array(tmp_path / "flat.npy", flat)  # write a corrupt pair by hand
    save_array(tmp_path / "offsets.npy", np.array(offsets, dtype=np.int64))
    with pytest.raises(IndexFormatError, match="corrupt ragged"):
        load_ragged(tmp_path)
    with pytest.raises(ValueError, match="not a valid ragged"):
        save_ragged(tmp_path / "other", flat, np.array(offsets, dtype=np.int64))


def test_offsets_must_be_int64_on_disk(tmp_path):
    save_array(tmp_path / "flat.npy", np.zeros((3, 2), dtype=np.float32))
    save_array(tmp_path / "offsets.npy", np.array([0, 3], dtype=np.int32))
    with pytest.raises(IndexFormatError, match="int64"):
        load_ragged(tmp_path)


def test_zero_dimensional_flat_is_rejected(tmp_path):
    save_array(tmp_path / "flat.npy", np.float32(1))
    save_array(tmp_path / "offsets.npy", np.array([0], dtype=np.int64))
    with pytest.raises(IndexFormatError, match="single number"):
        load_ragged(tmp_path)
