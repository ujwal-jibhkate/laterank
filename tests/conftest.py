"""Shared test helpers: a fake encoder, so Encoder code runs without a model or torch."""

import hashlib

import numpy as np
import pytest
import scipy.sparse as sp

from laterank.components.base import Encoded
from laterank.components.encoder import EncodedDocs
from laterank.storage.manifest import EncoderInfo


class FakeEncoder:
    """Vectors seeded by the text's hash: the same text always gives the same vectors, so a
    query equal to a document's text finds that document."""

    def __init__(
        self, dim=8, dense_dim=6, vocab=500, *, dense=True, lexical=True, hf_id=None, revision=None
    ):
        self.dim, self.dense_dim, self.vocab = dim, dense_dim, vocab
        self.dense, self.lexical = dense, lexical
        self.info = EncoderInfo("fake", hf_id, revision, dim, {"vocab": vocab})

    def _one(self, text):
        words = text.split() or ["<empty>"]
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(text.encode()).digest()[:8]))
        tokens = rng.standard_normal((len(words), self.dim)).astype(np.float32)
        dense = rng.standard_normal(self.dense_dim).astype(np.float32) if self.dense else None
        terms = sorted(
            {int.from_bytes(hashlib.sha256(w.encode()).digest()[:4]) % self.vocab for w in words}
        )
        lexical = (np.array(terms, dtype=np.int64), np.ones(len(terms), dtype=np.float32))
        return tokens, dense, lexical if self.lexical else None

    def encode_docs(self, texts):
        parts = [self._one(t) for t in texts]
        offsets = np.concatenate([[0], np.cumsum([len(p[0]) for p in parts])]).astype(np.int64)
        tokens = np.concatenate([p[0] for p in parts]).astype(np.float16)
        dense = np.stack([p[1] for p in parts]) if self.dense else None
        lexical = None
        if self.lexical:
            rows = np.repeat(np.arange(len(parts)), [len(p[2][0]) for p in parts])
            cols = np.concatenate([p[2][0] for p in parts])
            vals = np.concatenate([p[2][1] for p in parts])
            lexical = sp.csr_matrix((vals, (rows, cols)), shape=(len(parts), self.vocab))
        return EncodedDocs(tokens, offsets, dense, lexical)

    def encode_query(self, text):
        tokens, dense, lexical = self._one(text)
        return Encoded(tokens, dense=dense, lexical=lexical, text=text)


@pytest.fixture
def fake_encoder_cls():
    return FakeEncoder
