"""Index.build: texts in, a searchable index out, with a fake encoder (no model, no torch)."""

import pytest

from laterank.errors import IndexExistsError
from laterank.orchestration.index import Index

DOCS = [
    {"id": "a", "title": "Fruit", "text": "red apple pie"},
    {"id": "b", "text": "green pear tart"},
    {"id": "c", "title": "  ", "text": "  yellow banana split  "},
]
TEXTS = ["Fruit. red apple pie", "green pear tart", "yellow banana split"]  # what the encoder sees


def test_build_makes_the_encoders_stages_and_finds_each_doc_by_its_own_text(
    tmp_path, fake_encoder_cls
):
    enc = fake_encoder_cls()
    index = Index.build(tmp_path / "ix", DOCS, enc)
    assert sorted(index.manifest.stages) == ["dense", "hybrid", "lexical"]
    assert index.manifest.encoder == enc.info
    assert str(index.default_pipeline) == "hybrid > maxsim@100"
    for doc, text in zip(DOCS, TEXTS, strict=True):
        assert index.search(enc.encode_query(text), k=1)[0].id == doc["id"]
    assert Index.open(tmp_path / "ix").manifest == index.manifest  # and it reopens


def test_text_sent_to_the_encoder_is_title_dot_text(tmp_path, fake_encoder_cls):
    seen = []

    class Spy(fake_encoder_cls):
        def encode_docs(self, texts):
            seen.extend(texts)
            return super().encode_docs(texts)

    Index.build(tmp_path / "ix", DOCS, Spy())
    assert seen == TEXTS


def test_missing_head_for_the_default_pipeline_writes_nothing(tmp_path, fake_encoder_cls):
    with pytest.raises(ValueError, match="hybrid"):
        Index.build(tmp_path / "ix", DOCS, fake_encoder_cls(lexical=False))
    assert not (tmp_path / "ix").exists()
    # the same encoder works when the pipeline only needs what it has
    index = Index.build(tmp_path / "ix", DOCS, fake_encoder_cls(lexical=False), pipeline="dense")
    assert sorted(index.manifest.stages) == ["dense"]


def test_a_taken_folder_is_refused_before_any_encoding(tmp_path, fake_encoder_cls):
    (tmp_path / "ix").mkdir()
    (tmp_path / "ix" / "keep.txt").write_text("x")

    class Boom(fake_encoder_cls):
        def encode_docs(self, texts):
            raise AssertionError("encoded before checking the folder")

    with pytest.raises(IndexExistsError):
        Index.build(tmp_path / "ix", DOCS, Boom())
    with pytest.raises(ValueError, match="mapping"):
        Index.build(tmp_path / "new", ["not a mapping"], Boom())
    with pytest.raises(ValueError, match=r"document 'x'.*strings"):
        Index.build(tmp_path / "new", [{"id": "x", "title": 5, "text": "t"}], Boom())
