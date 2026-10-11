"""manifest.json: exact round trips, strict validation, and atomic writes that survive crashes."""

import json
import os
import subprocess
import sys
import textwrap

import pytest

import laterank.storage.manifest as manifest_mod
from laterank.errors import IndexFormatError
from laterank.storage.manifest import (
    MANIFEST,
    EncoderInfo,
    Manifest,
    SegmentInfo,
    read_manifest,
    write_manifest,
)

SHA = "5617a9f61b028005a4858fdac845db406aefb181"


def _manifest(pipeline="hybrid > maxsim@100", docs=5183):
    return Manifest(
        encoder=EncoderInfo(name="bge-m3", hf_id="BAAI/bge-m3", revision=SHA, dim=1024),
        stages={
            "bm25": {"type": "bm25", "params": {"k1": 1.2, "b": 0.75}},
            "smve-balanced": {"type": "smve", "params": {"w": 65536}, "anchor_sha256": "ab" * 32},
        },
        default_pipeline=pipeline,
        segments=[SegmentInfo(id="seg-000001", documents=docs, tokens=1_861_234)],
    )


def _write_json(tmp_path, d):
    (tmp_path / MANIFEST).write_text(json.dumps(d), encoding="utf-8")


# --- AT1: round trip -------------------------------------------------------------------------


def test_round_trip(tmp_path):
    m = _manifest()
    write_manifest(tmp_path, m)
    assert read_manifest(tmp_path) == m
    on_disk = json.loads((tmp_path / MANIFEST).read_text(encoding="utf-8"))
    assert list(on_disk) == [
        "format",
        "format_version",
        "created_with",
        "created_at",
        "encoder",
        "stages",
        "default_pipeline",
        "segments",
    ]
    assert on_disk["format"] == "laterank-index" and on_disk["format_version"] == 1
    assert on_disk["created_with"].startswith("laterank ")
    assert on_disk["encoder"] == {
        "name": "bge-m3",
        "hf_id": "BAAI/bge-m3",
        "revision": SHA,
        "dim": 1024,
    }
    assert on_disk["segments"] == [{"id": "seg-000001", "documents": 5183, "tokens": 1861234}]


def test_precomputed_encoder_needs_no_revision(tmp_path):
    m = Manifest(
        encoder=EncoderInfo(name="precomputed", hf_id=None, revision=None, dim=128),
        stages={},
        default_pipeline="exhaustive",
        segments=(),
    )
    write_manifest(tmp_path, m)
    assert read_manifest(tmp_path) == m


def test_overwrite_replaces_and_leaves_no_temp_files(tmp_path):
    write_manifest(tmp_path, _manifest(docs=1))
    write_manifest(tmp_path, _manifest(docs=2))
    assert read_manifest(tmp_path).segments[0].documents == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == [MANIFEST]


# --- AT2: clear refusals ---------------------------------------------------------------------


def _good(tmp_path):
    return _manifest().to_dict()


def _mutate(d, key, value):
    d = json.loads(json.dumps(d))
    if value is ...:
        del d[key]
    else:
        d[key] = value
    return d


BAD = {
    "newer version": (
        lambda d: _mutate(_mutate(d, "format_version", 2), "created_with", "laterank 0.3.0"),
        r"laterank 0\.3\.0 in format 2.*pip install -U laterank",
    ),
    "missing version": (lambda d: _mutate(d, "format_version", ...), "format_version"),
    "string version": (lambda d: _mutate(d, "format_version", "1"), "format_version"),
    "bool version": (lambda d: _mutate(d, "format_version", True), "format_version"),
    "wrong format": (
        lambda d: _mutate(d, "format", "mvr-index"),
        "not a laterank index.*rebuild the index",
    ),
    "missing key": (lambda d: _mutate(d, "stages", ...), r"missing \['stages'\]"),
    "unknown key": (lambda d: _mutate(d, "chunking", {}), r"doesn't know: \['chunking'\].*upgrade"),
    "branch revision": (
        lambda d: {**d, "encoder": {**d["encoder"], "revision": "main"}},
        "commit sha",
    ),
    "no revision": (lambda d: {**d, "encoder": {**d["encoder"], "revision": None}}, "commit sha"),
    "unknown encoder key": (
        lambda d: {**d, "encoder": {**d["encoder"], "extra": 1}},
        "'encoder' has keys",
    ),
    "stage without type": (lambda d: {**d, "stages": {"bm25": {"params": {}}}}, "'type'"),
    "bad segment id": (
        lambda d: {**d, "segments": [{"id": "seg-1", "documents": 1, "tokens": 0}]},
        "seg-000001",
    ),
    "negative count": (
        lambda d: {**d, "segments": [{"id": "seg-000001", "documents": -1, "tokens": 0}]},
        "non-negative",
    ),
    "empty pipeline": (lambda d: _mutate(d, "default_pipeline", " "), "default_pipeline"),
}


@pytest.mark.parametrize(("make", "pattern"), BAD.values(), ids=BAD.keys())
def test_refuses_what_it_does_not_understand(tmp_path, make, pattern):
    _write_json(tmp_path, make(_good(tmp_path)))
    with pytest.raises(IndexFormatError, match=pattern) as info:
        read_manifest(tmp_path)
    assert MANIFEST in str(info.value)  # every message names the file


def test_missing_and_truncated_manifest(tmp_path):
    with pytest.raises(IndexFormatError, match=r"has no manifest\.json"):
        read_manifest(tmp_path)
    write_manifest(tmp_path, _manifest())
    data = (tmp_path / MANIFEST).read_bytes()
    (tmp_path / MANIFEST).write_bytes(data[: len(data) // 2])
    with pytest.raises(IndexFormatError, match=r"not valid JSON.*truncated"):
        read_manifest(tmp_path)
    for bad in (b"[" * 200_000, b'{"format_version":' + b"9" * 5000 + b"}"):  # too deep, too long
        (tmp_path / MANIFEST).write_bytes(bad)
        with pytest.raises(IndexFormatError, match="not valid JSON"):
            read_manifest(tmp_path)


def test_manifest_that_is_not_a_file(tmp_path):
    (tmp_path / MANIFEST).mkdir()  # Windows raises PermissionError here, POSIX IsADirectoryError
    (tmp_path / "afile").write_bytes(b"")  # NotADirectoryError everywhere
    for index_dir in (tmp_path, tmp_path / "afile"):
        with pytest.raises(IndexFormatError, match=r"has no manifest\.json file"):
            read_manifest(index_dir)


def _int_key_stage(m):
    m.stages["bm25"]["params"] = {1: 2}


def _tuple_stage(m):
    m.stages["bm25"]["params"] = (1.2, 0.75)


def _untyped_stage(m):
    del m.stages["bm25"]["type"]  # mutated after the constructor checked it


def _nan_stage(m):
    m.stages["bm25"]["params"]["k1"] = float("nan")


@pytest.mark.parametrize("spoil", [_int_key_stage, _tuple_stage, _untyped_stage, _nan_stage])
def test_write_refuses_what_would_not_read_back(tmp_path, spoil):
    m = _manifest()
    spoil(m)
    with pytest.raises(ValueError):
        write_manifest(tmp_path, m)
    assert list(tmp_path.iterdir()) == []


def test_constructor_rejects_bad_values():
    with pytest.raises(ValueError, match="commit sha"):
        EncoderInfo(name="bge-m3", hf_id="BAAI/bge-m3", revision="main", dim=1024)
    with pytest.raises(ValueError, match="hf_id is not"):
        EncoderInfo(name="x", hf_id=None, revision=SHA, dim=8)
    seg = SegmentInfo(id="seg-000001", documents=1, tokens=1)
    with pytest.raises(ValueError, match="unique"):
        Manifest(
            encoder=_manifest().encoder, stages={}, default_pipeline="bm25", segments=[seg, seg]
        )


# --- AT3: a half-written manifest never replaces a good one ----------------------------------


def _assert_old_manifest_intact(tmp_path, old_bytes):
    assert (tmp_path / MANIFEST).read_bytes() == old_bytes
    assert sorted(p.name for p in tmp_path.iterdir()) == [MANIFEST]  # no temp file left
    assert read_manifest(tmp_path).segments[0].documents == 1


def test_failure_during_serialization_touches_nothing(tmp_path):
    write_manifest(tmp_path, _manifest(docs=1))
    old = (tmp_path / MANIFEST).read_bytes()
    bad = _manifest(docs=2)
    bad.stages["bm25"]["params"]["k1"] = object()  # not JSON-serializable
    with pytest.raises(ValueError, match="only JSON values"):
        write_manifest(tmp_path, bad)
    _assert_old_manifest_intact(tmp_path, old)


@pytest.mark.parametrize("step", ["fsync", "replace"])
def test_failure_before_the_swap_keeps_the_old_manifest(tmp_path, monkeypatch, step):
    write_manifest(tmp_path, _manifest(docs=1))
    old = (tmp_path / MANIFEST).read_bytes()

    def crash(*args):
        raise OSError("disk full")

    monkeypatch.setattr(manifest_mod.os, step, crash)
    with pytest.raises(OSError, match="disk full"):
        write_manifest(tmp_path, _manifest(docs=2))
    monkeypatch.undo()
    _assert_old_manifest_intact(tmp_path, old)


def test_writer_killed_before_the_swap_leaves_a_readable_index(tmp_path):
    write_manifest(tmp_path, _manifest(docs=1))
    old = (tmp_path / MANIFEST).read_bytes()
    # A separate process writes a new manifest but dies (no cleanup, no exception handlers)
    # right after the temp file is on disk, just before os.replace would swap it in.
    child = textwrap.dedent(
        f"""
        import os, sys
        sys.path.insert(0, {os.path.dirname(__file__)!r})
        import laterank.storage.manifest as m
        from test_manifest import _manifest
        m.os.replace = lambda *args: os._exit(3)
        m.write_manifest({str(tmp_path)!r}, _manifest(docs=2))
        """
    )
    result = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True)
    assert result.returncode == 3, result.stderr
    assert (tmp_path / MANIFEST).read_bytes() == old
    assert read_manifest(tmp_path).segments[0].documents == 1
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != MANIFEST]
    assert len(leftovers) == 1 and leftovers[0].startswith("manifest.json.tmp-")  # ignored
