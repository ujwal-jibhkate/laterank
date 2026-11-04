"""ColBERTEncoder's pinning, settings and output layout, with sentence_transformers and
huggingface_hub stubbed (no model, no network). The real models are checked in
tests/golden/test_colbert.py."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from laterank.components.colbert import ColBERTEncoder
from laterank.components.encoder import load_encoder
from laterank.storage.manifest import EncoderInfo

SHA = "d" * 40
EXPANSION = {"strategy": "fixed", "attend": False, "token": None, "length": 32}


class MultiVectorMask:  # found by class name, as in the real library
    def __init__(self):
        self.skiplist_words = [".", ",", "!"]
        self.skiplist_tasks = ["document"]


def _model_class(expansion):
    class Model:
        def __init__(self, path, device):
            self.path = path
            self.prompts = {"query": "[Q] ", "document": "[D] "}

        def __getitem__(self, i):
            return SimpleNamespace(
                document_length=300, query_length=None, query_expansion=expansion
            )

        def children(self):
            return [object(), MultiVectorMask()]

        def get_embedding_dimension(self):
            return 4

        def encode_document(self, texts, **kwargs):  # one (len(text.split()), 4) array per text
            return [np.full((len(t.split()), 4), 0.5, dtype=np.float32) for t in texts]

        def encode_query(self, texts, **kwargs):
            return [np.full((32, 4), 0.5, dtype=np.float32) for _ in texts]

    return Model


@pytest.fixture
def downloads(monkeypatch, tmp_path):
    """Stub the model libraries; returns the list of snapshot_download calls."""
    folder = tmp_path / "snapshots" / SHA  # the cache names a snapshot folder by its commit
    folder.mkdir(parents=True)
    calls = []

    def snapshot_download(hf_id, revision=None, ignore_patterns=None):
        calls.append((hf_id, revision, ignore_patterns))
        return str(folder)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(MultiVectorEncoder=_model_class(EXPANSION)),
    )
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download)
    )
    no_gpu = SimpleNamespace(is_available=lambda: False)  # so device="auto" works without torch
    torch = SimpleNamespace(backends=SimpleNamespace(mps=no_gpu), cuda=no_gpu)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return calls


@pytest.mark.parametrize("revision", [None, SHA])
def test_the_downloaded_commit_and_the_models_own_settings_are_recorded(downloads, revision):
    enc = ColBERTEncoder("x/colbert", revision, device="cpu")
    assert downloads == [("x/colbert", revision, ["*.onnx", "onnx/*"])]  # no ONNX copies
    assert (enc.info.name, enc.info.hf_id, enc.info.revision) == ("colbert", "x/colbert", SHA)
    assert enc.info.dim == 4
    assert enc.info.settings == {
        "document_length": 300,
        "query_length": None,
        "query_expansion": EXPANSION,
        "query_prefix": "[Q] ",
        "document_prefix": "[D] ",
        "skiplist": ["!", ",", "."],
        "skiplist_tasks": ["document"],
        "normalize": True,
        "vector_dtype": "float16",
    }


@pytest.mark.parametrize("revision", ["main", "v1.0", SHA[:7], SHA.upper(), ""])
def test_anything_but_a_full_sha_is_refused_before_downloading(downloads, revision):
    with pytest.raises(ValueError, match="40-character commit sha"):
        ColBERTEncoder("x/colbert", revision, device="cpu")
    assert downloads == []


def test_docs_come_back_as_float16_tokens_with_offsets_and_no_other_heads(downloads):
    enc = ColBERTEncoder("x/colbert", device="cpu", batch_size=2)
    docs = enc.encode_docs(["a b c", "d", "e f"])
    assert docs.tokens.shape == (6, 4) and docs.tokens.dtype == np.float16
    assert docs.offsets.tolist() == [0, 3, 4, 6] and docs.offsets.dtype == np.int64
    assert docs.dense is None and docs.lexical is None
    with pytest.raises(ValueError, match="at least one text"):
        enc.encode_docs([])


def test_a_query_keeps_every_row_the_model_returns(downloads):
    q = ColBERTEncoder("x/colbert", device="cpu").encode_query("what is this")
    assert q.tokens.shape == (32, 4) and q.tokens.dtype == np.float32
    assert q.text == "what is this" and q.dense is None and q.lexical is None


def test_load_encoder_rebuilds_the_recorded_model_and_refuses_changed_settings(downloads):
    recorded = ColBERTEncoder("x/colbert", device="cpu").info
    assert load_encoder(recorded).info == recorded

    drifted = EncoderInfo(
        "colbert", "x/colbert", SHA, 4, {**recorded.settings, "query_prefix": "?"}
    )
    with pytest.raises(ValueError, match="isn't the one the index was built with"):
        load_encoder(drifted)


def test_a_colbert_entry_without_hf_id_is_refused():
    with pytest.raises(ValueError, match="no hf_id"):
        load_encoder(EncoderInfo("colbert", None, None, 4, {}))


def test_without_the_extra_it_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match=r'pip install "laterank\[colbert\]"'):
        ColBERTEncoder("x/colbert")
