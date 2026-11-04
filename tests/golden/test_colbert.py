"""Two small ColBERT models on SciFact: do their vectors look right, and how well do they search?

Needs the `colbert` extra (`uv sync --extra colbert`), the models (~134 MB each, downloaded at the
pinned shas), and the exported texts (`scripts/export_golden_texts.py`). There is no smve-lab
reference for these models, so the nDCG floor only catches broken vectors (a wrong prefix or a
missing normalisation craters it). The real numbers are printed: run with `-s` to read them.
"""

import time

import numpy as np
import pytest

from laterank.kernels.metrics import mean_metric, ndcg_at_k, recall_at_k
from laterank.orchestration.index import Index

pytestmark = pytest.mark.slow

MODELS = {  # name -> (hf_id, commit sha: `main` on 2026-11-04)
    "answerai": (
        "answerdotai/answerai-colbert-small-v1",
        "934fa8bb4ce2284f4c2baa232d81aca4d076fa5e",
    ),
    "mxbai": (
        "mixedbread-ai/mxbai-edge-colbert-v0-32m",
        "bb13a29ec9b1e7edd4ba8f7a0776c48b55cbad66",
    ),
}
PIPELINE = "bm25 > maxsim@100"
NDCG_FLOOR = 0.60  # both model cards report about 0.74 for exhaustive search on SciFact


@pytest.fixture(scope="module", params=MODELS)
def encoder(request, scifact):
    pytest.importorskip("sentence_transformers", reason="needs the colbert extra")
    if scifact.d_texts is None or scifact.q_texts is None:
        pytest.fail("the *_texts.json files are missing: run scripts/export_golden_texts.py")
    from laterank.components.colbert import ColBERTEncoder

    hf_id, sha = MODELS[request.param]
    enc = ColBERTEncoder(hf_id, sha)
    assert enc.info.revision == sha
    return enc


def test_vectors_have_the_models_dim_and_unit_length(scifact, encoder):
    docs = encoder.encode_docs(scifact.d_texts[:5])
    assert docs.tokens.shape[1] == encoder.info.dim and len(docs.offsets) == 6
    norms = np.linalg.norm(docs.tokens.astype(np.float32), axis=-1)
    assert np.abs(norms - 1).max() < 0.01, norms  # unit length, up to float16 rounding

    q = encoder.encode_query(scifact.q_texts[0])
    assert q.tokens.shape[1] == encoder.info.dim
    expansion = encoder.info.settings["query_expansion"]
    if expansion is not None:  # the model pads every query to a fixed number of rows
        assert len(q.tokens) == expansion["length"]


def test_bm25_then_maxsim_on_the_full_corpus(scifact, encoder, tmp_path):
    t0 = time.perf_counter()
    docs = encoder.encode_docs(scifact.d_texts)
    encode_s = time.perf_counter() - t0
    records = [{"id": i, "text": t} for i, t in zip(scifact.doc_ids, scifact.d_texts, strict=True)]
    index = Index.create(
        tmp_path / "index",
        records,
        (docs.tokens, docs.offsets),
        encoder=encoder.info,
        bm25=True,
        pipeline=PIPELINE,
    )
    run = {
        qid: [h.id for h in index.search(encoder.encode_query(text), 100)]
        for qid, text in zip(scifact.query_ids, scifact.q_texts, strict=True)
    }
    ndcg = mean_metric(run, scifact.qrels, ndcg_at_k, 10)
    recall = mean_metric(run, scifact.qrels, recall_at_k, 100)
    n = len(scifact.doc_ids)
    print(
        f"\n{encoder.info.hf_id}@{encoder.info.revision[:8]} dim={encoder.info.dim}: "
        f"{PIPELINE} nDCG@10={ndcg:.4f} Recall@100={recall:.4f}; "
        f"{len(docs.tokens) / n:.1f} tokens/doc, {docs.tokens.nbytes / n / 1024:.1f} KB/doc, "
        f"encode {encode_s:.0f}s"
    )
    assert ndcg >= NDCG_FLOOR, f"nDCG@10 {ndcg:.4f} is below {NDCG_FLOOR}: are the vectors broken?"
