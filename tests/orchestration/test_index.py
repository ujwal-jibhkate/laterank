"""Index: create -> open -> search gives what the pipeline gives on the same in-memory arrays."""

import json
from dataclasses import replace

import numpy as np
import pytest

from laterank.components.base import Encoded
from laterank.components.dense import DenseStage
from laterank.components.maxsim import MaxSimReranker
from laterank.errors import IndexExistsError, IndexFormatError
from laterank.orchestration.index import Index
from laterank.orchestration.pipeline import parse_pipeline
from laterank.storage.manifest import EncoderInfo, read_manifest, write_manifest
from laterank.storage.segment import write_segment

DIM = 8
ENCODER = EncoderInfo(name="precomputed", hf_id=None, revision=None, dim=DIM)


def _data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, 6, size=n)
    flat = rng.standard_normal((lengths.sum(), DIM)).astype(np.float16)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    dense = rng.standard_normal((n, 4)).astype(np.float32)
    docs = [{"id": f"d{i}", "text": f"text {i}", "metadata": {"i": i}} for i in range(n)]
    q = Encoded(rng.standard_normal((3, DIM)).astype(np.float32), rng.standard_normal(4))
    return docs, (flat, offsets), dense, q


def test_search_matches_the_pipeline_on_in_memory_arrays(tmp_path):
    docs, (flat, offsets), dense, q = _data()
    Index.create(tmp_path / "ix", docs, (flat, offsets), encoder=ENCODER, dense=dense)
    index = Index.open(tmp_path / "ix")
    stages = {"dense": DenseStage(dense), "maxsim": MaxSimReranker(flat, offsets)}

    for spec, k in (("dense > maxsim@20", 10), ("dense", 5)):
        want = parse_pipeline(spec).run(stages, q, k)
        hits = index.search(q, k, pipeline=spec)
        assert [h.id for h in hits] == [f"d{r}" for r in want.rows]
        assert [h.score for h in hits] == want.scores.tolist()
        assert [h.rank for h in hits] == list(range(1, k + 1))
        r = int(want.rows[0])
        assert (hits[0].text, hits[0].metadata) == (f"text {r}", {"i": r})

    assert index.search(q) == index.search(q, pipeline="dense > maxsim@100")  # the default
    on_disk = json.loads((tmp_path / "ix" / "manifest.json").read_text())
    assert on_disk["stages"] == {"dense": {"type": "dense", "dim": 4}}
    assert on_disk["default_pipeline"] == "dense > maxsim@100"


def test_create_refuses_bad_inputs_and_writes_nothing(tmp_path):
    docs, (flat, offsets), dense, _ = _data()
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "x").write_text("")
    with pytest.raises(IndexExistsError):
        Index.create(tmp_path / "taken", docs, (flat, offsets), encoder=ENCODER, dense=dense)

    bad = {
        "documents": dict(docs=docs[:-1]),
        "dense rows": dict(dense=dense[:-1]),
        "token dim": dict(tokens=(flat[:, :4], offsets)),
        "offsets": dict(tokens=(flat, offsets[:-1])),
        "no dense stage": dict(dense=None),
    }
    for case in bad.values():
        args = dict(docs=docs, tokens=(flat, offsets), dense=dense) | case
        out = tmp_path / "new"
        with pytest.raises(ValueError):
            Index.create(out, args["docs"], args["tokens"], encoder=ENCODER, dense=args["dense"])
        assert not out.exists() or not any(out.iterdir())


def test_search_names_the_stages_an_index_has(tmp_path):
    docs, tokens, dense, q = _data()
    index = Index.create(tmp_path / "ix", docs, tokens, encoder=ENCODER, dense=dense)
    with pytest.raises(ValueError, match=r"first stages \['dense'\]"):
        index.search(q, pipeline="bm25 > maxsim@100")
    with pytest.raises(ValueError, match="unknown rerankers"):
        index.search(q, pipeline="dense > colbert@100")


def test_open_refuses_a_bad_default_pipeline_and_extra_segments(tmp_path):
    docs, tokens, dense, _ = _data()
    path = tmp_path / "ix"
    index = Index.create(path, docs, tokens, encoder=ENCODER, dense=dense)
    manifest = read_manifest(path)

    for broken in ("dense > maxsim", "bm25 > maxsim@100"):
        write_manifest(path, replace(manifest, default_pipeline=broken))
        with pytest.raises(IndexFormatError, match="default pipeline"):
            Index.open(path)

    arrays = {"tokens/flat.npy": tokens[0], "tokens/offsets.npy": tokens[1]}
    info = write_segment(path, 2, arrays | DenseStage.arrays(dense), docs)
    segments = (*index.manifest.segments, info)
    write_manifest(path, replace(manifest, segments=segments))
    with pytest.raises(IndexFormatError, match="2 segments"):
        Index.open(path)


def test_a_query_given_as_lists_works(tmp_path):
    docs, tokens, dense, q = _data()
    index = Index.create(tmp_path / "ix", docs, tokens, encoder=ENCODER, dense=dense)
    as_lists = Encoded(q.tokens.tolist(), q.dense.tolist())
    assert index.search(as_lists, 3) == index.search(q, 3)


def test_open_refuses_arrays_that_disagree_with_the_documents(tmp_path):
    docs, tokens, dense, _ = _data()
    path = tmp_path / "ix"
    Index.create(path, docs, tokens, encoder=ENCODER, dense=dense)
    vectors = path / "seg-000001" / "dense" / "vectors.npy"
    np.save(vectors, np.vstack([dense, dense[:5]]), allow_pickle=False)
    with pytest.raises(IndexFormatError, match="dense vectors"):
        Index.open(path)
    np.save(vectors, dense, allow_pickle=False)
    manifest = read_manifest(path)
    write_manifest(path, replace(manifest, stages={"dense": {"type": "dense", "dim": 99}}))
    with pytest.raises(IndexFormatError, match="dense vectors"):
        Index.open(path)


def test_a_failed_create_leaves_the_folder_as_it_was(tmp_path, monkeypatch):
    docs, tokens, dense, _ = _data()
    bad_docs = [{"text": "no id"}] * len(docs)
    with pytest.raises(ValueError):
        Index.create(tmp_path / "new", bad_docs, tokens, encoder=ENCODER, dense=dense)
    assert not (tmp_path / "new").exists()

    (tmp_path / "empty").mkdir()
    monkeypatch.setattr("laterank.orchestration.index.write_manifest", _fail)
    with pytest.raises(OSError):
        Index.create(tmp_path / "empty", docs, tokens, encoder=ENCODER, dense=dense)
    assert list((tmp_path / "empty").iterdir()) == []  # so a retry can succeed
    monkeypatch.undo()
    Index.create(tmp_path / "empty", docs, tokens, encoder=ENCODER, dense=dense)


def _fail(*args):
    raise OSError("disk full")


def test_open_turns_a_malformed_index_into_index_format_error(tmp_path):
    docs, tokens, dense, _ = _data()
    path = tmp_path / "ix"
    Index.create(path, docs, tokens, encoder=ENCODER, dense=dense)
    manifest = read_manifest(path)
    write_manifest(path, replace(manifest, stages={"dense": {"type": "dense"}}))  # no "dim"
    with pytest.raises(IndexFormatError, match="malformed"):
        Index.open(path)
    write_manifest(path, manifest)
    np.save(path / "seg-000001" / "tokens" / "flat.npy", np.zeros(5), allow_pickle=False)
    with pytest.raises(IndexFormatError):
        Index.open(path)
