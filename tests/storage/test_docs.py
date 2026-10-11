"""docs.jsonl + byte offsets: exact round trips and random access, on every platform."""

import json

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from laterank.errors import IndexFormatError
from laterank.storage.docs import DOCS, OFFSETS, DocStore, write_docs

DOCS_IN = [
    {"id": "a", "title": "Statins", "text": "Statins lower LDL.", "metadata": {"year": 2019}},
    {"id": "b", "text": "हिन्दी पाठ और emoji 🧬, café"},  # non-ASCII: offsets must count bytes
    {"id": "c", "text": "line one\nline two\r\nline three\u2028still c"},  # newlines inside text
    {
        "id": "d",
        "title": None,
        "text": None,
        "metadata": {"nested": {"tags": ["x", "y"], "n": 1.5}},
    },
]


def _full(doc):
    return {
        "id": doc["id"],
        "title": doc.get("title"),
        "text": doc.get("text"),
        "metadata": doc.get("metadata", {}),
    }


def test_round_trip_and_random_access(tmp_path):
    assert write_docs(tmp_path, DOCS_IN) == 4
    store = DocStore(tmp_path)
    assert len(store) == 4
    assert store.ids() == ["a", "b", "c", "d"]
    for i in (2, 0, 3, 1):  # out of order: each read seeks to its own offset
        assert store[i] == _full(DOCS_IN[i])
    assert store[-1] == _full(DOCS_IN[3])


def test_offsets_are_byte_offsets_of_lines(tmp_path):
    write_docs(tmp_path, DOCS_IN)
    raw = (tmp_path / DOCS).read_bytes()
    assert b"\r\n" not in raw.replace(b"\\r\\n", b"")  # only escaped \r\n inside text, no CRLF
    lines = raw.split(b"\n")[:-1]
    assert len(lines) == 4  # one line per document despite the newlines inside "c"
    offsets = np.load(tmp_path / OFFSETS, allow_pickle=False)
    assert offsets.dtype == np.int64
    assert offsets.tolist() == np.cumsum([0] + [len(line) + 1 for line in lines]).tolist()
    for i, line in enumerate(lines):
        assert raw[offsets[i] : offsets[i + 1]] == line + b"\n"
        assert json.loads(line)["id"] == DOCS_IN[i]["id"]


def test_empty_segment(tmp_path):
    assert write_docs(tmp_path, []) == 0
    store = DocStore(tmp_path)
    assert len(store) == 0 and store.ids() == []
    with pytest.raises(IndexError):
        store[0]


def test_bad_documents_are_refused(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        write_docs(tmp_path, [{"id": "a"}, {"id": "a"}])
    with pytest.raises(ValueError, match="'id'"):
        write_docs(tmp_path, [{"text": "no id"}])
    with pytest.raises(ValueError, match="extra"):
        write_docs(tmp_path, [{"id": "a", "body": "?"}])
    with pytest.raises(ValueError, match="string or None"):
        write_docs(tmp_path, [{"id": "a", "text": 42}])
    for falsy in ("", 0, [], False):  # only None means "no metadata"
        with pytest.raises(ValueError, match="'metadata' must be a dict"):
            write_docs(tmp_path, [{"id": "a", "metadata": falsy}])


@pytest.mark.parametrize(
    "metadata",
    [{"x": object()}, {1: 2}, {"x": float("nan")}, {"x": float("inf")}, {"x": (1, 2)}],
    ids=["object", "int key", "nan", "inf", "tuple"],
)
def test_metadata_that_would_not_read_back_is_refused(tmp_path, metadata):
    with pytest.raises(ValueError, match="plain JSON"):
        write_docs(tmp_path, [{"id": "ok"}, {"id": "a", "metadata": metadata}])
    assert list(tmp_path.iterdir()) == []  # no partial docs.jsonl


def test_corrupt_lines_are_an_index_error(tmp_path):
    write_docs(tmp_path, [{"id": "a"}])
    size = (tmp_path / DOCS).stat().st_size
    (tmp_path / DOCS).write_bytes(b'{"x":1}'.ljust(size - 1) + b"\n")  # same size, no "id"
    with pytest.raises(IndexFormatError, match="corrupt"):
        DocStore(tmp_path).ids()
    (tmp_path / DOCS).write_bytes(b"[1]".ljust(size - 1) + b"\n")  # valid JSON, not a document
    with pytest.raises(IndexFormatError, match="corrupt"):
        DocStore(tmp_path)[0]


_json = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(),
    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text(), inner, max_size=3),
    max_leaves=10,
)
_doc = st.fixed_dictionaries(
    {
        "title": st.none() | st.text(),
        "text": st.none() | st.text(),
        "metadata": st.dictionaries(st.text(), _json, max_size=3),
    }
)


@settings(max_examples=50, deadline=None)
@given(st.lists(_doc, max_size=5))
def test_any_documents_round_trip(tmp_path_factory, docs):
    folder = tmp_path_factory.mktemp("docs")
    docs = [{"id": f"d{i}", **d} for i, d in enumerate(docs)]
    assert write_docs(folder, docs) == len(docs)
    store = DocStore(folder)
    assert [store[i] for i in range(len(store))] == docs


def test_out_of_range_rows(tmp_path):
    write_docs(tmp_path, DOCS_IN)
    store = DocStore(tmp_path)
    for row in (4, -5):
        with pytest.raises(IndexError):
            store[row]


def test_offsets_that_do_not_match_the_file_are_rejected(tmp_path):
    write_docs(tmp_path, DOCS_IN)
    with open(tmp_path / DOCS, "ab") as f:
        f.write(b'{"id":"sneaky"}\n')  # the file grew but the offsets didn't
    with pytest.raises(IndexFormatError, match="doesn't match"):
        DocStore(tmp_path)


def test_missing_files(tmp_path):
    with pytest.raises(IndexFormatError, match="missing"):
        DocStore(tmp_path)
    write_docs(tmp_path, DOCS_IN)
    (tmp_path / DOCS).unlink()
    with pytest.raises(IndexFormatError, match="missing"):
        DocStore(tmp_path)
