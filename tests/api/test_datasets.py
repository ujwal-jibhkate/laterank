"""load_beir and download_beir on tiny BEIR folders: no network, no model."""

import hashlib
import http.client
import io
import json
import zipfile
from pathlib import Path

import pytest

from laterank.api import datasets
from laterank.api.datasets import download_beir, load_beir
from laterank.orchestration.index import Index

CORPUS = [
    {"_id": 1, "title": "Fruit", "text": "red apple", "metadata": {"url": "x"}},  # integer id
    {"_id": "d2", "text": "green pear"},  # no title
    {"_id": "d3", "title": "", "text": ""},  # empty document
]
QUERIES = [
    {"_id": "q1", "text": "red fruit", "metadata": {}},
    {"_id": 2, "text": "green fruit"},
    {"_id": "q3", "text": "judged only in dev"},
    {"_id": "q4", "text": "judged only below zero"},
]
HEADER = "query-id\tcorpus-id\tscore"
TEST_QRELS = f"{HEADER}\r\nq1\t1\t2\r\nq1\td2\t0\r\n2\td2\t1\r\n2\tgone\t1\r\nq4\td3\t-1\r\n"
DEV_QRELS = f"{HEADER}\nq3\t1\t1\n"


def _jsonl(rows):
    return "".join(json.dumps(r) + "\n" for r in rows).encode()


def _beir(folder, corpus=None, queries=None, qrels=None):
    """Write a BEIR folder; pass raw bytes to corrupt a file."""
    (folder / "qrels").mkdir(parents=True)
    (folder / "corpus.jsonl").write_bytes(_jsonl(CORPUS) if corpus is None else corpus)
    (folder / "queries.jsonl").write_bytes(_jsonl(QUERIES) if queries is None else queries)
    (folder / "qrels/test.tsv").write_bytes(TEST_QRELS.encode() if qrels is None else qrels)
    (folder / "qrels/dev.tsv").write_bytes(DEV_QRELS.encode())
    return folder


def test_load_beir_cleans_up_the_format(tmp_path):
    data = load_beir(_beir(tmp_path / "tiny"))
    assert data.corpus == [
        {"id": "1", "title": "Fruit", "text": "red apple"},
        {"id": "d2", "title": "", "text": "green pear"},
        {"id": "d3", "title": "", "text": ""},
    ]
    assert data.queries == {"q1": "red fruit", "2": "green fruit"}
    assert data.qrels == {"q1": {"1": 2}, "2": {"d2": 1, "gone": 1}}
    assert data.ignore_identical_ids is False


def test_split_picks_the_qrels_file(tmp_path):
    data = load_beir(_beir(tmp_path / "tiny"), split="dev")
    assert data.queries == {"q3": "judged only in dev"}
    assert data.qrels == {"q3": {"1": 1}}


def test_arguana_folder_ignores_identical_ids_even_from_inside(tmp_path, monkeypatch):
    folder = _beir(tmp_path / "arguana")
    assert load_beir(folder).ignore_identical_ids is True
    monkeypatch.chdir(folder)
    assert load_beir(".").ignore_identical_ids is True


def test_a_symlink_named_arguana_counts_as_arguana(tmp_path):
    link = tmp_path / "arguana"
    try:
        link.symlink_to(_beir(tmp_path / "v1"), target_is_directory=True)
    except OSError:
        pytest.skip("this system refuses symlinks")
    assert load_beir(link).ignore_identical_ids is True


@pytest.mark.parametrize("name", ["corpus.jsonl", "queries.jsonl", "qrels/test.tsv"])
def test_a_missing_file_is_named(tmp_path, name):
    folder = _beir(tmp_path / "tiny")
    (folder / name).unlink()
    with pytest.raises(FileNotFoundError, match=Path(name).name):  # Windows writes qrels\\test.tsv
        load_beir(folder)


@pytest.mark.parametrize(
    ("files", "message"),
    [
        ({"corpus": b"{oops\n"}, r"corpus\.jsonl, line 1: not valid JSON"),
        ({"corpus": b'{"_id": "a", "text": "\xff"}\n'}, r"corpus\.jsonl, line 1: not valid JSON"),
        ({"corpus": b"[" * 200_000 + b"\n"}, r"corpus\.jsonl, line 1: not valid JSON"),
        ({"corpus": b'{"text": "no id"}\n'}, r"corpus\.jsonl, line 1: every row needs"),
        ({"corpus": b'{"_id": "a", "text": 7}\n'}, r"corpus\.jsonl, line 1: every row needs"),
        ({"queries": b"[1]\n"}, r"queries\.jsonl, line 1: every row needs"),
        ({"qrels": b"q1\t1\t2\n"}, r"test\.tsv: the first row must be"),
        ({"qrels": f"{HEADER}\nq1\t1\n".encode()}, r"test\.tsv, line 2: expected"),
        ({"qrels": f"{HEADER}\nq1\t1\thigh\n".encode()}, r"test\.tsv, line 2: expected"),
        ({"qrels": HEADER.encode() + b"\nq1\td\xff\t1\n"}, r"test\.tsv, line 2: expected"),
        ({"qrels": f"{HEADER}\nnope\t1\t1\n".encode()}, r"test\.tsv: 1 judged queries aren't"),
    ],
)
def test_bad_files_raise_and_name_the_file(tmp_path, files, message):
    with pytest.raises(ValueError, match=message):
        load_beir(_beir(tmp_path / "tiny", **files))


def test_build_and_evaluate_run_on_the_output(tmp_path, fake_encoder_cls):
    data = load_beir(_beir(tmp_path / "tiny"))
    encoder = fake_encoder_cls()
    index = Index.build(tmp_path / "index", data.corpus, encoder, pipeline="dense > maxsim@3")
    queries = {query_id: encoder.encode_query(text) for query_id, text in data.queries.items()}
    report = index.evaluate(
        queries,
        data.qrels,
        ["dense > maxsim@3"],
        repeats=1,
        ignore_identical_ids=data.ignore_identical_ids,
    )
    assert report.n_queries == 2
    assert 0 <= report.scores[0].ndcg_at_10 <= 1


@pytest.fixture
def host(tmp_path, monkeypatch):
    """A local stand-in for UKP: "tiny" is one zip behind a file:// URL, with its right md5."""
    site = tmp_path / "site"
    site.mkdir()
    source = _beir(tmp_path / "source" / "tiny")
    with zipfile.ZipFile(site / "tiny.zip", "w") as z:
        for path in source.rglob("*"):
            z.write(path, path.relative_to(source.parent))
    monkeypatch.setattr(datasets, "_URL", site.as_uri() + "/{}.zip")
    md5 = hashlib.md5((site / "tiny.zip").read_bytes()).hexdigest()
    monkeypatch.setitem(datasets._MD5, "tiny", md5)
    return md5


def _no_network(*args, **kwargs):
    raise AssertionError("opened the URL")


def test_download_returns_a_readable_folder_then_stays_offline(tmp_path, host, monkeypatch):
    cache = tmp_path / "cache"
    folder = download_beir("tiny", cache)
    assert folder == cache / "tiny"
    assert [p.name for p in cache.iterdir()] == ["tiny"]  # the temp folder is gone
    assert load_beir(folder).qrels == {"q1": {"1": 2}, "2": {"d2": 1, "gone": 1}}
    monkeypatch.setattr(datasets, "urlopen", _no_network)
    assert download_beir("tiny", cache) == folder


def test_a_damaged_download_leaves_nothing_and_a_retry_works(tmp_path, host, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setitem(datasets._MD5, "tiny", "0" * 32)
    with pytest.raises(ValueError, match="damaged"):
        download_beir("tiny", cache)
    assert list(cache.iterdir()) == []
    monkeypatch.setitem(datasets._MD5, "tiny", host)
    assert download_beir("tiny", cache).is_dir()


def test_the_folder_appears_only_by_a_rename_inside_the_cache(tmp_path, host, monkeypatch):
    cache = tmp_path / "cache"
    sources = []

    def fail(source, target):
        sources.append(Path(source))
        raise OSError("disk full")

    monkeypatch.setattr(datasets.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        download_beir("tiny", cache)
    assert cache in sources[0].parents  # same disk, so the rename is atomic
    assert list(cache.iterdir()) == []


def test_a_chunked_response_cut_off_mid_read_is_an_oserror_naming_the_url(
    tmp_path, host, monkeypatch
):
    class CutOff(io.BytesIO):
        def read(self, *args):
            raise http.client.IncompleteRead(b"")

    monkeypatch.setattr(datasets, "urlopen", lambda *args, **kwargs: CutOff())
    with pytest.raises(OSError, match=r"tiny\.zip"):
        download_beir("tiny", tmp_path / "cache")
    assert list((tmp_path / "cache").iterdir()) == []


def test_an_unknown_name_raises_before_any_network_call(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "urlopen", _no_network)
    with pytest.raises(ValueError, match="scifact"):
        download_beir("nope", tmp_path)
