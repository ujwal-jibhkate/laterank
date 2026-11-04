"""BGEM3Encoder reproduces smve-lab's stored BGE-M3 embeddings on SciFact's first documents.

Needs the `bge-m3` extra (`uv sync --extra bge-m3`) and the cached model at the revision below.
The texts are the exact strings smve-lab encoded (`scripts/export_golden_texts.py`).
"""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

SMVE_LAB_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"  # BAAI/bge-m3 when the golden ran
N = 20  # documents compared
NQ = 5  # queries compared
LEXICAL_ATOL = 0.002  # weights are ~0.1-0.3; fp16 noise is ~0.0004
MIN_COSINE = 0.999  # fp16 and device noise only


def _cosine(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def _same_lexical(ids_a, w_a, ids_b, w_b, what):
    """Same weight for every term, up to LEXICAL_ATOL. A term whose weight is ~0 may be present
    on one side only: fp16 noise pushes it across the relu (seen: 0.00014 for "to")."""
    a, b = (
        dict(zip(ids_a.tolist(), w_a.tolist(), strict=True)),
        dict(zip(ids_b.tolist(), w_b.tolist(), strict=True)),
    )
    worst = max(abs(a.get(t, 0.0) - b.get(t, 0.0)) for t in a.keys() | b.keys())
    assert worst <= LEXICAL_ATOL, f"{what}: a lexical weight differs by {worst}"


@pytest.fixture(scope="module")
def encoder(scifact):
    pytest.importorskip("FlagEmbedding", reason="needs the bge-m3 extra")
    if scifact.d_texts is None or scifact.q_texts is None:
        pytest.fail("the *_texts.json files are missing: run scripts/export_golden_texts.py")
    from laterank.components.bgem3 import BGEM3Encoder

    enc = BGEM3Encoder(revision=SMVE_LAB_REVISION)
    assert enc.info.dim == 1024 and enc.info.revision == SMVE_LAB_REVISION
    return enc


def test_first_documents_match_smve_labs_embeddings(scifact, encoder):
    got = encoder.encode_docs(scifact.d_texts[:N])

    assert np.array_equal(got.offsets, scifact.d_offsets[: N + 1])  # same token count per doc
    cosine = _cosine(got.dense, scifact.d_dense[:N])
    assert cosine.min() >= MIN_COSINE, cosine
    for i in range(N):  # the lexical weights of each document
        a, b = got.lexical.indptr[i : i + 2], scifact.d_lexical.indptr[i : i + 2]
        _same_lexical(
            got.lexical.indices[a[0] : a[1]],
            got.lexical.data[a[0] : a[1]],
            scifact.d_lexical.indices[b[0] : b[1]],
            scifact.d_lexical.data[b[0] : b[1]],
            f"document {i}",
        )


def test_first_queries_match_smve_labs_embeddings(scifact, encoder):
    """encode_query: the token, dense and lexical heads (documents above check only some)."""
    for i in range(NQ):
        q = encoder.encode_query(scifact.q_texts[i])
        assert q.text == scifact.q_texts[i] and q.tokens.dtype == np.float32
        want_tokens = scifact.q_flat[scifact.q_offsets[i] : scifact.q_offsets[i + 1]]
        assert q.tokens.shape == want_tokens.shape, f"query {i}"
        assert _cosine(q.tokens, want_tokens).min() >= MIN_COSINE, f"query {i} tokens"
        assert _cosine(q.dense, scifact.q_dense[i]) >= MIN_COSINE, f"query {i} dense"
        _same_lexical(
            q.lexical[0], q.lexical[1], *map(np.asarray, scifact.q_lexical[i]), f"query {i}"
        )
