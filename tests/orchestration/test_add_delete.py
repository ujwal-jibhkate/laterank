"""Index.add and Index.delete: the rewritten index must equal a fresh build (spec M5-2)."""

import json

import pytest

import laterank.components.encoder as encoder_mod
import laterank.orchestration.index as index_module
from laterank.orchestration.index import Index

WORDS = "apple pear plum fig kiwi lime date grape melon peach cherry mango".split()
DOCS = [{"id": f"d{i}", "title": f"T{i}", "text": " ".join(WORDS[i : i + 4])} for i in range(9)]
STAGES = {"bm25": True, "smve": True, "muvera": True}


def build(path, docs, enc):
    return Index.build(path, docs, enc, **STAGES)


def snapshot(path):
    """Every file of the one segment as bytes, plus the manifest without segment id and time."""
    (seg,) = [p for p in path.iterdir() if p.is_dir()]
    files = {str(p.relative_to(seg)): p.read_bytes() for p in seg.rglob("*") if p.is_file()}
    manifest = json.loads((path / "manifest.json").read_text())
    del manifest["created_at"]
    for s in manifest["segments"]:
        del s["id"]
    return files, manifest


def test_add_equals_a_fresh_build(tmp_path, fake_encoder_cls):  # AT1, AT3
    enc = fake_encoder_cls()
    index = build(tmp_path / "ix", DOCS[:6], enc)
    index.add(DOCS[6:], enc)
    assert snapshot(tmp_path / "ix") == snapshot(build(tmp_path / "fresh", DOCS, enc).path)
    # this object reads the new segment, and so does a reopen; the old folder is gone
    query = enc.encode_query("T7. " + DOCS[7]["text"])
    assert index.search(query, k=1)[0].id == "d7"
    assert Index.open(tmp_path / "ix").search(query, k=1)[0].id == "d7"
    assert [p.name for p in (tmp_path / "ix").iterdir() if p.is_dir()] == ["seg-000002"]


def test_add_without_an_encoder_checks_before_loading_the_model(
    tmp_path, fake_encoder_cls, monkeypatch
):  # M6-1 AT2
    enc = fake_encoder_cls()
    monkeypatch.setitem(encoder_mod.ENCODERS, "fake", lambda info: pytest.fail("loaded"))
    index = build(tmp_path / "ix", DOCS[:6], enc)
    with pytest.raises(ValueError, match="already in the index"):
        index.add([{"id": "d0", "text": "x"}])
    monkeypatch.setitem(encoder_mod.ENCODERS, "fake", lambda info: enc)
    index.add(DOCS[6:])
    with_enc = build(tmp_path / "ix2", DOCS[:6], enc)
    with_enc.add(DOCS[6:], enc)
    assert snapshot(tmp_path / "ix") == snapshot(tmp_path / "ix2")


@pytest.mark.parametrize("gone", [["d6", "d7", "d8"], ["d1", "d4"]])  # end and middle
def test_delete_equals_a_fresh_build(tmp_path, fake_encoder_cls, gone):  # AT2, AT3
    enc = fake_encoder_cls()
    index = build(tmp_path / "ix", DOCS, enc)
    index.delete(gone)
    kept = [d for d in DOCS if d["id"] not in gone]
    assert snapshot(tmp_path / "ix") == snapshot(build(tmp_path / "fresh", kept, enc).path)
    assert [p.name for p in (tmp_path / "ix").iterdir() if p.is_dir()] == ["seg-000002"]


def test_refusals_leave_the_index_unchanged_and_add_never_encodes(tmp_path, fake_encoder_cls):
    enc = fake_encoder_cls()  # AT4
    index = build(tmp_path / "ix", DOCS[:6], enc)
    before = snapshot(tmp_path / "ix")

    class Boom(fake_encoder_cls):
        def encode_docs(self, texts):
            raise AssertionError("encoded before the checks")

    boom = Boom()
    other = Boom(vocab=499)  # another settings dict, so another EncoderInfo
    for docs, encoder, match in [
        ([], boom, "one or more"),
        (["d9"], boom, "one or more"),
        ([{"id": "d0", "text": "x"}], boom, "already in the index"),
        ([{"id": "n", "text": "x"}, {"id": "n", "text": "y"}], boom, "repeat"),
        ([{"text": "x"}], boom, "string 'id'"),
        ([{"id": "n", "text": "x", "extra": 1}], boom, "only have the keys"),
        ([{"id": "n", "text": "x", "metadata": {1: "a"}}], boom, "plain JSON"),
        ([{"id": "n", "text": "x"}], other, "isn't the one"),
    ]:
        with pytest.raises(ValueError, match=match):
            index.add(docs, encoder)
    with pytest.raises(ValueError, match="heads"):  # same EncoderInfo, no dense head
        index.add(DOCS[6:7], fake_encoder_cls(dense=False))
    for ids, match in [
        ("d0", "list of ids"),
        (b"d0", "list of ids"),
        ([], "one or more"),
        (["d0", "zz"], "zz"),
    ]:
        with pytest.raises(ValueError, match=match):
            index.delete(ids)
    with pytest.raises(ValueError, match="every document"):
        index.delete([d["id"] for d in DOCS[:6]])
    assert snapshot(tmp_path / "ix") == before


def test_bm25_failure_writes_nothing(tmp_path, fake_encoder_cls):  # AT4, a _stage_arrays refusal
    enc = fake_encoder_cls()
    index = build(tmp_path / "ix", [{"id": "a", "text": "apple"}, {"id": "b", "text": "the"}], enc)
    before = snapshot(tmp_path / "ix")
    with pytest.raises(ValueError, match="no document has a word"):
        index.delete(["a"])  # leaves only "the", a stop word
    assert snapshot(tmp_path / "ix") == before
    assert [p.name for p in (tmp_path / "ix").iterdir() if p.is_dir()] == ["seg-000001"]


def test_a_crash_before_the_manifest_keeps_the_old_index(tmp_path, fake_encoder_cls, monkeypatch):
    enc = fake_encoder_cls()  # AT5
    build(tmp_path / "ix", DOCS[:6], enc)
    index = Index.open(tmp_path / "ix")

    def crash(*args):
        raise OSError("power cut")

    monkeypatch.setattr(index_module, "write_manifest", crash)
    with pytest.raises(OSError, match="power cut"):
        index.add(DOCS[6:7], enc)
    monkeypatch.undo()
    del index  # its memmaps of seg-000001 would block the cleanup on Windows
    reopened = Index.open(tmp_path / "ix")
    assert reopened.manifest.segments[0].id == "seg-000001"
    assert len(reopened._docs) == 6
    assert (tmp_path / "ix" / "seg-000002").is_dir()  # the stray, complete segment
    reopened.add(DOCS[6:], enc)  # takes seg-000003 and removes both older folders
    assert [p.name for p in (tmp_path / "ix").iterdir() if p.is_dir()] == ["seg-000003"]
    assert snapshot(tmp_path / "ix") == snapshot(build(tmp_path / "fresh", DOCS, enc).path)
