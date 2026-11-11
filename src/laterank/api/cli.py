"""The `laterank` command: build, add, delete, search, evaluate, tune.

Each command wraps `Index`. The one thing it adds is turning query text into vectors, with the
encoder the index records. Results go to stdout; notices and progress bars go to stderr.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from tqdm import tqdm

from laterank.api.datasets import BeirData, download_beir, load_beir, load_corpus
from laterank.components.base import Encoded
from laterank.components.encoder import Encoder, load_encoder
from laterank.errors import LaterankError
from laterank.orchestration.index import Index, _check_free
from laterank.orchestration.pipeline import parse_pipeline
from laterank.orchestration.tune import Budget

DEFAULT_PIPELINE = {"bge-m3": "hybrid > maxsim@100", "colbert": "bm25 > maxsim@100"}
PIPELINE_HELP = "a pipeline, in quotes (an unquoted > redirects the output to a file)"
DATASET_HELP = "a folder holding corpus.jsonl, or a BEIR name (downloaded on first use)"


def _make_encoder(name: str, hf_id: str | None, revision: str | None) -> Encoder:
    """Build a BGEM3Encoder or ColBERTEncoder (imported here). Tests patch this."""
    if name == "bge-m3":
        from laterank.components.bgem3 import BGEM3Encoder

        return BGEM3Encoder(hf_id or "BAAI/bge-m3", revision)
    from laterank.components.colbert import ColBERTEncoder

    return ColBERTEncoder(hf_id, revision)


def _folder(value: str) -> Path:
    """A folder holding corpus.jsonl wins; anything else is a BEIR name."""
    if value and (Path(value) / "corpus.jsonl").is_file():  # Path("") is the current folder
        return Path(value)
    print(f"laterank: BEIR dataset {value!r}: downloaded if not cached yet", file=sys.stderr)
    return download_beir(value)


def _dataset(value: str, split: str) -> BeirData:
    return load_beir(_folder(value), split)


def _encode_queries(encoder: Encoder, texts: Mapping[str, str]) -> tuple[dict[str, Encoded], float]:
    """Encode each text with a tqdm bar. Returns the queries and p50 ms per query."""
    queries, seconds = {}, []
    for query_id, text in tqdm(texts.items(), desc="encoding queries"):
        t0 = time.perf_counter()
        queries[query_id] = encoder.encode_query(text)
        seconds.append(time.perf_counter() - t0)
    return queries, float(np.median(seconds or [0])) * 1000  # no queries: evaluate refuses them


def _build(args: argparse.Namespace) -> int:
    if args.encoder == "colbert" and not args.hf_id:
        raise ValueError(
            "--encoder colbert needs --hf-id, e.g. mixedbread-ai/mxbai-edge-colbert-v0-32m"
        )
    pipeline = DEFAULT_PIPELINE[args.encoder] if args.pipeline is None else args.pipeline
    first = parse_pipeline(pipeline).first  # before the model loads; a bad string fails fast
    _check_free(Path(args.index))  # before the model loads, which can be a multi-GB download
    # the corpus is the same for every split, so a build needs no queries or qrels
    path = args.docs if args.docs is not None else _folder(args.dataset) / "corpus.jsonl"
    docs = load_corpus(path)
    index = Index.build(
        args.index,
        docs,
        _make_encoder(args.encoder, args.hf_id, args.revision),
        bm25=args.bm25 or first == "bm25",
        smve=args.smve or first == "smve",
        muvera=args.muvera or first == "muvera",
        pipeline=pipeline,
    )
    print(
        f"built {args.index}: {len(docs)} documents, stages {sorted(index.manifest.stages)}, "
        f"default pipeline '{index.default_pipeline}'"
    )
    return 0


def _add(args: argparse.Namespace) -> int:
    index = Index.open(args.index)  # a bad folder fails first
    docs = load_corpus(args.docs)  # a missing file or bad JSON fails before the model loads
    index.add(docs, load_encoder(index.manifest.encoder))
    print(
        f"added {len(docs)} documents to {args.index}: "
        f"{index.manifest.segments[0].documents} in total"
    )
    return 0


def _delete(args: argparse.Namespace) -> int:  # never loads the model
    index = Index.open(args.index)
    index.delete(args.ids)
    print(
        f"deleted {len(set(args.ids))} documents from {args.index}: "
        f"{index.manifest.segments[0].documents} in total"
    )
    return 0


def _search(args: argparse.Namespace) -> int:
    index = Index.open(args.index)
    # checked before the model loads
    pipeline = index._pipeline(index.default_pipeline if args.pipeline is None else args.pipeline)
    query = load_encoder(index.manifest.encoder).encode_query(args.query)
    for hit in index.search(query, args.k, pipeline=pipeline):
        text = " ".join((hit.text or "").split())[:80]
        print(f"{hit.rank:>3}  {hit.score:.4f}  {hit.id}  {text}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    index = Index.open(args.index)
    # checked before the model loads and the queries are encoded
    pipelines = [index._pipeline(p) for p in args.pipeline or [index.default_pipeline]]
    data = _dataset(args.dataset, args.split)
    queries, p50_ms = _encode_queries(load_encoder(index.manifest.encoder), data.queries)
    evaluation = index.evaluate(
        queries, data.qrels, pipelines, ignore_identical_ids=data.ignore_identical_ids
    )
    print(evaluation.table())
    print(f"query encoding: p50 {p50_ms:.1f} ms over {len(queries)} queries")
    return 0


def _tune(args: argparse.Namespace) -> int:
    budget = Budget(args.budget_ms)
    index = Index.open(args.index)
    if args.dataset is not None:
        data = _dataset(args.dataset, args.split)
        queries, _ = _encode_queries(load_encoder(index.manifest.encoder), data.queries)
        report = index.tune(
            budget,
            queries=queries,
            qrels=data.qrels,
            ignore_identical_ids=data.ignore_identical_ids,
        )
    else:
        print(
            "laterank: no --dataset, so this tunes without labels: it measures agreement with "
            "exact MaxSim, not relevance. Pass --dataset for the recommended mode.",
            file=sys.stderr,
        )
        report = index.tune(budget)
    print(report.table())
    print(f"best: {report.best or f'none (nothing fits {args.budget_ms:g} ms)'}")
    if args.apply:
        if report.best is None:
            raise ValueError("no pipeline fits the budget, so the default pipeline is unchanged")
        index.set_default(report.best)
        print(f"default pipeline: {report.best}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laterank", description="Local late-interaction retrieval."
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    build = commands.add_parser("build", help="encode documents and write an index folder")
    build.set_defaults(run=_build)
    build.add_argument("index", metavar="INDEX", help="the folder to create")
    source = build.add_mutually_exclusive_group(required=True)
    source.add_argument("--docs", metavar="FILE", help="JSONL rows: _id, text, optional title")
    source.add_argument("--dataset", metavar="D", help=DATASET_HELP)
    build.add_argument("--encoder", required=True, choices=sorted(DEFAULT_PIPELINE))
    build.add_argument("--hf-id", help="Hugging Face model id; required for colbert")
    build.add_argument("--revision", metavar="SHA", help="commit to pin (default: today's main)")
    for stage in ("bm25", "smve", "muvera"):
        build.add_argument(f"--{stage}", action="store_true", help=f"also build the {stage} stage")
    build.add_argument("--pipeline", metavar="P", help=f"default pipeline: {PIPELINE_HELP}")

    add = commands.add_parser("add", help="encode documents and add them (rewrites the index)")
    add.set_defaults(run=_add)
    add.add_argument("index", metavar="INDEX")
    add.add_argument(
        "--docs", metavar="FILE", required=True, help="JSONL rows: _id, text, optional title"
    )

    delete = commands.add_parser("delete", help="remove documents by id (rewrites the index)")
    delete.set_defaults(run=_delete)
    delete.add_argument("index", metavar="INDEX")
    delete.add_argument("ids", metavar="ID", nargs="+", help='put "--" before an id starting "-"')

    search = commands.add_parser("search", help="search an index with one query")
    search.set_defaults(run=_search)
    search.add_argument("index", metavar="INDEX")
    search.add_argument("query", metavar="QUERY")
    search.add_argument("-k", type=int, default=10, help="how many hits (default: 10)")
    search.add_argument("--pipeline", metavar="P", help=PIPELINE_HELP)

    evaluate = commands.add_parser("evaluate", help="score pipelines against relevance labels")
    evaluate.set_defaults(run=_evaluate)
    evaluate.add_argument("index", metavar="INDEX")
    evaluate.add_argument("--dataset", metavar="D", required=True, help=DATASET_HELP)
    evaluate.add_argument("--split", default="test", help="the BEIR split (default: test)")
    evaluate.add_argument(
        "--pipeline", metavar="P", action="append", help=f"{PIPELINE_HELP}; repeats"
    )

    tune = commands.add_parser("tune", help="pick the best pipeline within a latency budget")
    tune.set_defaults(run=_tune)
    tune.add_argument("index", metavar="INDEX")
    tune.add_argument("--budget-ms", metavar="MS", type=float, required=True)
    tune.add_argument("--dataset", metavar="D", help=f"{DATASET_HELP}; recommended")
    tune.add_argument("--split", default="test", help="the BEIR split (default: test)")
    tune.add_argument("--apply", action="store_true", help="make the best pipeline the default")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse argv (default sys.argv[1:]), run one command, return the exit code."""
    args = _parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # a StringIO has no encoding to fail
        sys.stdout.reconfigure(errors="replace")  # a character it can't show must not abort
    try:
        return args.run(args)
    except (LaterankError, ValueError, OSError, ImportError) as e:
        print(f"laterank: error: {' '.join(str(e).split())}", file=sys.stderr)
        return 1
