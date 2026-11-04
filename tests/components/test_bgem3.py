"""BGEM3Encoder's pinning rules, with FlagEmbedding and huggingface_hub stubbed (no model, no
network). The real model is checked in tests/golden/test_encoder.py."""

import sys
from types import SimpleNamespace

import pytest

from laterank.components.bgem3 import BGEM3Encoder

SHA = "c" * 40


@pytest.fixture
def downloads(monkeypatch, tmp_path):
    """Stub the model libraries; returns the list of snapshot_download calls."""
    folder = tmp_path / "snapshots" / SHA  # the cache names a snapshot folder by its commit
    folder.mkdir(parents=True)
    (folder / "config.json").write_text('{"hidden_size": 16}')
    calls = []

    def snapshot_download(hf_id, revision=None):
        calls.append((hf_id, revision))
        return str(folder)

    class Model:
        def __init__(self, path, use_fp16, devices):
            self.path, self.tokenizer = path, [0] * 7

    monkeypatch.setitem(sys.modules, "FlagEmbedding", SimpleNamespace(BGEM3FlagModel=Model))
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download)
    )
    return calls


@pytest.mark.parametrize("revision", [None, SHA])
def test_the_downloaded_commit_is_what_gets_recorded(downloads, revision):
    enc = BGEM3Encoder(revision=revision, max_length=64, use_fp16=False, device="cpu")
    assert downloads == [("BAAI/bge-m3", revision)]  # None -> today's main
    assert (enc.info.name, enc.info.hf_id, enc.info.revision) == ("bge-m3", "BAAI/bge-m3", SHA)
    assert enc.info.dim == 16 and enc.info.settings == {"max_length": 64, "fp16": False}
    assert enc._vocab == 7 and enc._model.path.endswith(SHA)


@pytest.mark.parametrize("revision", ["main", "v1.0", SHA[:7], SHA.upper(), ""])
def test_anything_but_a_full_sha_is_refused_before_downloading(downloads, revision):
    with pytest.raises(ValueError, match="40-character commit sha"):
        BGEM3Encoder(revision=revision, device="cpu")
    assert downloads == []


def test_an_explicit_device_never_imports_torch(downloads, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # `import torch` now raises ImportError
    BGEM3Encoder(device="cpu")
    with pytest.raises(ImportError):
        BGEM3Encoder(device="auto")  # only "auto" has to ask torch
