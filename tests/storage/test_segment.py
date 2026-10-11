"""Segments: crash-safe writes, round trips, and open_index trusting only the manifest."""

import numpy as np
import pytest

import laterank.storage.segment as segment_mod
from laterank.errors import IndexFormatError
from laterank.kernels.maxsim import maxsim_scores
from laterank.storage.manifest import EncoderInfo, Manifest, SegmentInfo, write_manifest
from laterank.storage.segment import (
    open_index,
    open_segment,
    segment_name,
    write_segment,
)


def _contents(seed=0, lengths=(3, 1, 4)):
    rng = np.random.default_rng(seed)
    flat = rng.standard_normal((sum(lengths), 8)).astype(np.float16)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    arrays = {
        "tokens/flat.npy": flat,
        "tokens/offsets.npy": offsets,
        "dense/vectors.npy": rng.standard_normal((len(lengths), 4)).astype(np.float32),
    }
    docs = [{"id": f"d{seed}-{i}", "text": f"document {i}"} for i in range(len(lengths))]
    return arrays, docs


def _manifest(*infos):
    return Manifest(
        encoder=EncoderInfo(name="precomputed", hf_id=None, revision=None, dim=8),
        stages={},
        default_pipeline="exhaustive",
        segments=infos,
    )


def _names(folder):
    return sorted(p.name for p in folder.iterdir())


def test_segment_name():
    assert segment_name(1) == "seg-000001" and segment_name(999_999) == "seg-999999"
    for bad in (0, 1_000_000):
        with pytest.raises(ValueError):
            segment_name(bad)


def test_round_trip(tmp_path):
    arrays, docs = _contents()
    info = write_segment(tmp_path, 1, arrays, docs)
    assert (info.id, info.documents, info.tokens) == ("seg-000001", 3, 8)
    assert _names(tmp_path) == ["seg-000001"]  # no temp folder left behind

    seg = open_segment(tmp_path, "seg-000001")
    assert seg.name == "seg-000001" and len(seg.docs) == 3
    assert seg.docs[1]["text"] == "document 1"
    for relpath, arr in arrays.items():
        got = seg.array(relpath)
        assert isinstance(got, np.memmap) and got.dtype == arr.dtype
        assert np.array_equal(got, arr)


def test_tokens_count_is_zero_without_a_token_store(tmp_path):
    info = write_segment(tmp_path, 2, {}, [{"id": "a"}])
    assert (info.documents, info.tokens) == (1, 0)


def test_existing_segments_are_never_overwritten(tmp_path):
    arrays, docs = _contents()
    write_segment(tmp_path, 1, arrays, docs)
    before = (tmp_path / "seg-000001" / "docs.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        write_segment(tmp_path, 1, *_contents(seed=1))
    assert (tmp_path / "seg-000001" / "docs.jsonl").read_bytes() == before
    assert _names(tmp_path) == ["seg-000001"]


@pytest.mark.parametrize(
    "relpath",
    [
        "../evil.npy",
        "/abs.npy",
        "a\\b.npy",
        "x.txt",
        "",
        "./x.npy",
        "a//b.npy",
        "docs.offsets.npy",
        "DOCS.OFFSETS.npy",
        "C:/x.npy",
        "C:x.npy",
        "x.npy:ads.npy",
        "//server/share/x.npy",
    ],
)
def test_bad_array_paths_are_refused_before_writing(tmp_path, relpath):
    with pytest.raises(ValueError):
        write_segment(tmp_path, 1, {relpath: np.zeros(2)}, [{"id": "a"}])
    assert not tmp_path.exists() or _names(tmp_path) == []


def test_array_paths_differing_only_by_case_are_refused(tmp_path):
    arrays = {"dense/v.npy": np.zeros(2), "Dense/V.npy": np.ones(2)}  # one file on macOS
    with pytest.raises(ValueError, match="case"):
        write_segment(tmp_path, 1, arrays, [{"id": "a"}])
    assert not tmp_path.exists() or _names(tmp_path) == []


def test_scalar_token_array_is_refused_before_writing(tmp_path):
    with pytest.raises(ValueError, match="one row per token"):
        write_segment(tmp_path, 1, {"tokens/flat.npy": np.float32(1)}, [{"id": "a"}])
    assert not tmp_path.exists() or _names(tmp_path) == []


@pytest.mark.parametrize("fail_in", ["arrays", "docs", "rename"])
def test_failure_mid_write_leaves_no_segment(tmp_path, monkeypatch, fail_in):
    def crash(*args, **kwargs):
        raise OSError("disk full")

    target = {"arrays": "save_array", "docs": "write_docs", "rename": "os.rename"}[fail_in]
    if target == "os.rename":
        monkeypatch.setattr(segment_mod.os, "rename", crash)
    else:
        monkeypatch.setattr(segment_mod, target, crash)
    with pytest.raises(OSError, match="disk full"):
        write_segment(tmp_path, 1, *_contents())
    assert _names(tmp_path) == []  # neither seg-000001/ nor a temp folder


def test_bad_documents_leave_nothing(tmp_path):
    arrays, _ = _contents()
    with pytest.raises(ValueError, match="duplicate"):
        write_segment(tmp_path, 1, arrays, [{"id": "a"}, {"id": "a"}])
    assert _names(tmp_path) == []


def test_open_index_opens_only_listed_segments(tmp_path):
    s1 = write_segment(tmp_path, 1, *_contents(seed=1))
    s2 = write_segment(tmp_path, 2, *_contents(seed=2, lengths=(2, 2)))
    write_segment(tmp_path, 3, *_contents(seed=3))  # complete, but never added to the manifest
    (tmp_path / "seg-000004.tmp-dead").mkdir()  # what a crash mid-write leaves behind
    write_manifest(tmp_path, _manifest(s1, s2))

    manifest, segments = open_index(tmp_path)
    assert [s.name for s in segments] == ["seg-000001", "seg-000002"]
    assert segments[1].docs.ids() == ["d2-0", "d2-1"]
    assert manifest.segments == (s1, s2)


def test_open_index_detects_missing_or_mismatched_segments(tmp_path):
    s1 = write_segment(tmp_path, 1, *_contents())
    write_manifest(tmp_path, _manifest(s1, SegmentInfo(id="seg-000002", documents=1, tokens=0)))
    with pytest.raises(IndexFormatError, match="seg-000002 is missing"):
        open_index(tmp_path)

    wrong = SegmentInfo(id=s1.id, documents=99, tokens=s1.tokens)
    write_manifest(tmp_path, _manifest(wrong))
    with pytest.raises(IndexFormatError, match="holds 3 documents, but the manifest says 99"):
        open_index(tmp_path)


def test_open_segment_errors(tmp_path):
    write_segment(tmp_path, 1, *_contents())
    with pytest.raises(ValueError):
        open_segment(tmp_path, "seg-1")
    with pytest.raises(IndexFormatError, match="missing"):
        open_segment(tmp_path, "seg-000009")
    seg = open_segment(tmp_path, "seg-000001")
    with pytest.raises(ValueError):
        seg.array("../manifest.npy")
    with pytest.raises(IndexFormatError, match="missing"):
        seg.array("lexical/weights.npy")


def test_kernels_run_on_a_reopened_index(tmp_path):
    """End to end: store token vectors, reopen them memory-mapped, and score with MaxSim."""
    arrays, docs = _contents(lengths=(5, 2, 7, 1))
    write_manifest(tmp_path, _manifest(write_segment(tmp_path, 1, arrays, docs)))
    _, [seg] = open_index(tmp_path)

    q = np.random.default_rng(9).standard_normal((3, 8)).astype(np.float32)
    q_off = np.array([0, 3])
    from_disk = maxsim_scores(
        q, q_off, seg.array("tokens/flat.npy"), seg.array("tokens/offsets.npy")
    )
    in_memory = maxsim_scores(q, q_off, arrays["tokens/flat.npy"], arrays["tokens/offsets.npy"])
    assert np.array_equal(from_disk, in_memory)
