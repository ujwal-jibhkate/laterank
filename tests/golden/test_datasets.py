"""The real BEIR zips against what smve-lab stored. Downloads about 9 MB, so it needs the network.

Needs the exported texts and qrels (`scripts/export_golden_texts.py`). SciFact's texts aren't
compared: smve-lab never stripped them (all 1,109 queries start with ". ", 18 docs differ by a
space).
"""

import json

import pytest

from laterank.api.datasets import download_beir, load_beir
from laterank.orchestration.index import _encoder_text

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return tmp_path_factory.mktemp("beir")


def _stored(folder, side):
    ids = json.loads((folder / f"{side}_ids.json").read_text(encoding="utf-8"))
    texts = json.loads((folder / f"{side}_texts.json").read_text(encoding="utf-8"))
    return dict(zip(ids, texts, strict=True))


@pytest.mark.parametrize("name", ["scifact", "nfcorpus", "arguana"])
def test_qrels_match_smve_lab(golden_dir, cache, name):
    data = load_beir(download_beir(name, cache))
    stored = golden_dir / f"{name}_bgem3" / "qrels_test.json"
    assert data.qrels == json.loads(stored.read_text(encoding="utf-8"))
    assert data.ignore_identical_ids is (name == "arguana")


@pytest.mark.parametrize("name", ["nfcorpus", "arguana"])
def test_texts_match_smve_lab(golden_dir, cache, name):
    data = load_beir(download_beir(name, cache))
    folder = golden_dir / f"{name}_bgem3"
    assert {doc["id"]: _encoder_text(doc) for doc in data.corpus} == _stored(folder, "docs")
    stored = _stored(folder, "queries")
    assert data.queries == {query_id: stored[query_id] for query_id in data.queries}
