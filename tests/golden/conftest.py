"""Golden-test harness: smve-lab's stored BGE-M3 embeddings, read with laterank's storage code.

Set LATERANK_GOLDEN_DIR to smve-lab's `artifacts/embeddings` folder (about 14 GB, never
committed). Unset, every golden test skips; set to a wrong path, they fail, so a typo can't
pass as green. The expected numbers come from smve-lab's `results/` (see the golden-values skill).
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from laterank.kernels.maxsim import subset_ragged
from laterank.storage.arrays import load_array

QRELS = Path(__file__).parent / "data" / "scifact-test-qrels.tsv"


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    value = os.environ.get("LATERANK_GOLDEN_DIR")
    if not value:
        pytest.skip("LATERANK_GOLDEN_DIR is not set")
    path = Path(value).expanduser()
    if not path.is_dir():
        pytest.fail(
            f"LATERANK_GOLDEN_DIR={value} is not a folder; point it at smve-lab's "
            "artifacts/embeddings"
        )
    return path


@pytest.fixture(scope="session")
def scifact_qrels() -> dict[str, dict[str, int]]:
    """The BEIR qrels TSV (header, then query-id, corpus-id, score) -> {qid: {doc_id: score}}."""
    qrels: dict[str, dict[str, int]] = {}
    for line in QRELS.read_text(encoding="utf-8").splitlines()[1:]:
        qid, doc_id, score = line.split("\t")
        if int(score) > 0:  # as smve-lab's load_qrels
            qrels.setdefault(qid, {})[doc_id] = int(score)
    return qrels


def _side(folder: Path, side: str) -> tuple[dict, list[str]]:
    """One side ("docs" or "queries") of smve-lab's embedding layout."""
    ids_path = folder / f"{side}_ids.json"
    if not ids_path.is_file():
        pytest.fail(f"{ids_path} is missing: is LATERANK_GOLDEN_DIR smve-lab's embeddings folder?")
    arrays = {
        key: load_array(folder / f"{side}_{key}.npy")  # memory-mapped, allow_pickle=False
        for key in ("colbert_flat", "colbert_offsets", "dense")
    }
    return arrays, json.loads(ids_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def scifact(golden_dir, scifact_qrels) -> SimpleNamespace:
    folder = golden_dir / "scifact_bgem3"
    qrels = scifact_qrels
    queries, all_query_ids = _side(folder, "queries")
    docs, doc_ids = _side(folder, "docs")
    query_ids = list(qrels)  # only judged queries can be scored
    row = {qid: i for i, qid in enumerate(all_query_ids)}
    rows = [row[q] for q in query_ids]
    q_flat, q_offsets = subset_ragged(queries["colbert_flat"], queries["colbert_offsets"], rows)
    return SimpleNamespace(
        qrels=qrels,
        query_ids=query_ids,
        doc_ids=doc_ids,
        q_flat=q_flat,
        q_offsets=q_offsets,
        d_flat=docs["colbert_flat"],
        d_offsets=docs["colbert_offsets"],
        q_dense=queries["dense"][rows],
        d_dense=docs["dense"],
    )
