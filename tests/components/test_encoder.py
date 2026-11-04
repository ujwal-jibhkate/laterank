"""load_encoder: the registry lookup and the "same model as the index" check."""

import numpy as np
import pytest

from laterank.components import encoder as encoder_mod
from laterank.components.encoder import load_encoder
from laterank.storage.manifest import EncoderInfo

SHA_A = "a" * 40
SHA_B = "b" * 40


def test_fake_encoder_is_deterministic_and_shaped(fake_encoder_cls):
    enc = fake_encoder_cls()
    docs = enc.encode_docs(["red apple", "green", "red apple"])
    assert docs.tokens.dtype == np.float16 and docs.tokens.shape == (5, 8)
    assert docs.offsets.tolist() == [0, 2, 3, 5] and docs.offsets.dtype == np.int64
    assert docs.dense.shape == (3, 6) and docs.lexical.shape == (3, 500)
    assert np.array_equal(docs.dense[0], docs.dense[2])  # same text, same vectors
    q = enc.encode_query("red apple")
    assert q.text == "red apple" and np.allclose(q.dense, docs.dense[0])


def test_unknown_name_is_refused_with_the_known_ones():
    info = EncoderInfo("precomputed", None, None, 8)
    with pytest.raises(ValueError, match=r"no encoder named 'precomputed'.*bge-m3"):
        load_encoder(info)


def test_loads_the_recorded_encoder_and_refuses_another_revision(monkeypatch, fake_encoder_cls):
    recorded = fake_encoder_cls(hf_id="x/y", revision=SHA_A)
    monkeypatch.setitem(
        encoder_mod.ENCODERS, "fake", lambda info: fake_encoder_cls(hf_id="x/y", revision=SHA_A)
    )
    assert load_encoder(recorded.info).info == recorded.info

    monkeypatch.setitem(
        encoder_mod.ENCODERS, "fake", lambda info: fake_encoder_cls(hf_id="x/y", revision=SHA_B)
    )
    with pytest.raises(ValueError, match="isn't the one the index was built with"):
        load_encoder(recorded.info)


def test_bgem3_settings_are_checked_before_any_model_loads(monkeypatch):
    # the settings check runs after the (lazy) import, so stub the module: no torch needed
    import sys
    import types

    monkeypatch.setitem(
        sys.modules, "laterank.components.bgem3", types.SimpleNamespace(BGEM3Encoder=None)
    )
    info = EncoderInfo("bge-m3", "BAAI/bge-m3", SHA_A, 1024, {"max_length": 512})
    with pytest.raises(ValueError, match="max_length and fp16"):
        load_encoder(info)
    no_model = EncoderInfo("bge-m3", None, None, 1024, {"max_length": 512, "fp16": True})
    with pytest.raises(ValueError, match="no hf_id"):
        load_encoder(no_model)
