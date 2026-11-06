"""BEIR datasets: read a local folder, or fetch one by name. Standard library only."""

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from urllib.request import urlopen

_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
_MD5 = {  # hashed from the downloaded zips; they match BEIR's README
    "scifact": "5f7d1de60b170fc8027bb7898e2efca1",
    "nfcorpus": "a89dba18a62ef92f7d323ec890a0d38d",
    "arguana": "8ad3e3c2a5867cdced806d6503f29b99",
    "fiqa": "17918ed23cd04fb15047f73e6c3bd9d9",
    "quora": "18fb154900ba42a600f84b839c173167",
    "scidocs": "38121350fc3a4d2f48850f6aff52e4a9",
    "trec-covid": "ce62140cb23feb9becf6270d0d1fe6d1",
}
_HEADER = b"query-id\tcorpus-id\tscore"


@dataclass(frozen=True)
class BeirData:
    corpus: list[dict[str, str]]  # {"id", "title", "text"}, file order; goes to Index.build
    queries: dict[str, str]  # query id -> text, only the queries judged in this split
    qrels: dict[str, dict[str, int]]  # query id -> {doc id: relevance}, relevance > 0 only
    ignore_identical_ids: bool  # pass to Index.evaluate / Index.tune


def _need(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing: point load_beir at an unzipped BEIR folder that has this split"
        )
    return path


def _rows(path: Path) -> Iterator[tuple[str, dict]]:
    """Each line of a BEIR .jsonl file as (id, row)."""
    with path.open("rb") as f:  # bytes, so a badly encoded line fails inside json.loads
        for n, line in enumerate(f, 1):
            try:
                row = json.loads(line)
            except (ValueError, RecursionError) as e:
                raise ValueError(
                    f"{path}, line {n}: not valid JSON ({e}); fix the file or download it again"
                ) from None
            if (
                not isinstance(row, dict)
                or row.get("_id") is None
                or not isinstance(row.get("text"), str)
            ):
                raise ValueError(
                    f"{path}, line {n}: every row needs an `_id` and a string `text`; "
                    "fix the file or download it again"
                )
            yield str(row["_id"]), row


def _read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open("rb") as f:  # bytes, so a badly encoded line fails inside the try
        if f.readline().rstrip(b"\r\n") != _HEADER:
            raise ValueError(
                f"{path}: the first row must be the header query-id<tab>corpus-id<tab>score; "
                "fix the file or download it again"
            )
        for n, line in enumerate(f, 2):
            try:
                query_id, doc_id, score = line.decode().rstrip("\r\n").split("\t")
                score = int(score)
            except ValueError:
                raise ValueError(
                    f"{path}, line {n}: expected query-id, corpus-id and an integer score, "
                    f"tab-separated, got {line[:80]!r}; fix the file or download it again"
                ) from None
            if score > 0:  # as smve-lab; ndcg_at_k would count a -1 as a negative gain
                qrels.setdefault(query_id, {})[doc_id] = score
    return qrels


def load_beir(folder: str | Path, split: str = "test") -> BeirData:
    """Read a BEIR folder: corpus.jsonl, queries.jsonl, qrels/<split>.tsv.

    Ids become strings. `title` defaults to "". `metadata` is dropped. Title and text stay
    separate (`Index.build` joins them). Documents with no text are kept. Judged documents
    missing from the corpus stay in qrels and count as misses. `ignore_identical_ids` is True
    when the folder is named "arguana".

    Raises:
        FileNotFoundError: one of the three files is missing (says which).
        ValueError: a JSON line is invalid or lacks `_id` or a string `text`; a qrels row isn't
            `query-id, corpus-id, integer score`; the first qrels row isn't that header; a judged
            query isn't in queries.jsonl. Names the file, and the line where there is one.
    """
    folder = Path(folder)
    corpus_path, queries_path, qrels_path = (
        _need(folder / name) for name in ("corpus.jsonl", "queries.jsonl", f"qrels/{split}.tsv")
    )
    qrels = _read_qrels(qrels_path)
    texts = {query_id: row["text"] for query_id, row in _rows(queries_path)}
    missing = sorted(set(qrels) - set(texts))
    if missing:
        raise ValueError(
            f"{qrels_path}: {len(missing)} judged queries aren't in {queries_path}, "
            f"first {missing[0]!r}; fix the files or download them again"
        )
    corpus = [
        {"id": doc_id, "title": row.get("title") or "", "text": row["text"]}
        for doc_id, row in _rows(corpus_path)
    ]
    # ponytail: a renamed ArguAna folder loses the rule, add an argument if that bites
    identical = Path(os.path.abspath(folder)).name == "arguana"  # keeps a symlink's own name
    return BeirData(corpus, {query_id: texts[query_id] for query_id in qrels}, qrels, identical)


def download_beir(name: str, folder: str | Path | None = None) -> Path:
    """Fetch one BEIR dataset on request and return its folder, `<folder>/<name>`.

    `folder` defaults to ~/.cache/laterank/beir. If `<folder>/<name>` exists it is returned with no
    network call: it only ever appears through the final rename, so it is complete. Otherwise:
    urlopen (timeout 60 s) into a temp folder inside `folder`, check the md5, extract, `os.replace`.

    Raises:
        ValueError: `name` isn't in the table (lists the names; before any network call), or the
            md5 doesn't match (says so; nothing is left behind, so a retry starts clean).
        OSError: the network or disk failed (`URLError` is an `OSError`). Nothing is left behind.
    """
    if name not in _MD5:
        raise ValueError(f"no BEIR dataset named {name!r} (known: {sorted(_MD5)})")
    folder = Path.home() / ".cache/laterank/beir" if folder is None else Path(folder)
    target = folder / name
    if target.is_dir():
        return target
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder) as tmp:
        archive = Path(tmp) / f"{name}.zip"
        url = _URL.format(name)
        try:
            with urlopen(url, timeout=60) as response, archive.open("wb") as f:
                shutil.copyfileobj(response, f)
        except (OSError, HTTPException) as e:  # a cut-off response isn't an OSError
            raise OSError(f"couldn't download {url} ({e}); fix that, then try again") from e
        with archive.open("rb") as f:
            md5 = hashlib.file_digest(f, lambda: hashlib.md5(usedforsecurity=False)).hexdigest()
        if md5 != _MD5[name]:
            raise ValueError(
                f"the {name} download is damaged (md5 {md5}, expected {_MD5[name]}); try again"
            )
        with zipfile.ZipFile(archive) as z:
            z.extractall(tmp)
        # ponytail: two processes can race here (loser: OSError), lock if downloads go parallel
        os.replace(Path(tmp) / name, target)
    return target
