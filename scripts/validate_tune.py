"""Does label-free `tune` pick what real labels would pick? Spec M2-3, design doc §8.

    LATERANK_GOLDEN_DIR=~/Projects/smve-lab/artifacts/embeddings HF_HUB_OFFLINE=1 \\
        uv run python scripts/validate_tune.py --dataset scifact

Needs `uv sync --extra bge-m3`, and `scripts/export_golden_texts.py scifact nfcorpus arguana`
run once inside smve-lab (texts and qrels_test.json). Stages are built straight from
smve-lab's memory-mapped BGE-M3 arrays (no index copy), then the same engines `Index.tune`
and `Index.evaluate` use are run on them:

1. evaluate: every candidate on the real test queries (smve-lab's stored query vectors).
2. tune: span queries encoded with BGE-M3 -> the Pareto frontier.
3. For each budget B (every frontier p50, then no limit): tune's pick vs the best nDCG@10 among
   candidates whose real-query p50 <= B. Holds if every gap <= 0.01.

Prints the result and appends it to docs/dev/tune_validation.md.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import scipy.sparse as sp

from laterank.components.base import Encoded
from laterank.components.bgem3 import BGEM3Encoder
from laterank.components.bm25 import BM25Stage
from laterank.components.dense import DenseStage
from laterank.components.hybrid import HYBRID_ALPHA, HybridStage
from laterank.components.lexical import LexicalStage
from laterank.components.maxsim import MaxSimReranker
from laterank.components.muvera import MUVERA_DEFAULT, MuveraStage
from laterank.components.smve import SMVE_BALANCED, SMVEStage
from laterank.kernels.rng import make_anchors
from laterank.orchestration.evaluate import evaluate
from laterank.orchestration.tune import EXHAUSTIVE, Budget, candidates, span_queries, tune
from laterank.storage.arrays import load_array

GOLDEN_EXHAUSTIVE = {"scifact": 0.6991, "nfcorpus": 0.3441, "arguana": 0.4841}  # nDCG@10
TOL, MAX_GAP = 0.002, 0.01
IGNORE_SELF = {"arguana"}  # BEIR's ignore_identical_ids: queries are corpus documents
VOCAB = 250_002  # BGE-M3's XLM-RoBERTa vocabulary
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"  # the BAAI/bge-m3 smve-lab encoded with
OUT = Path(__file__).resolve().parents[1] / "docs" / "dev" / "tune_validation.md"


def _side(folder: Path, side: str) -> tuple[dict, list[str], list[str]]:
    keys = ("colbert_flat", "colbert_offsets", "dense", "sparse_indptr", "sparse_ids")
    arrays = {k: load_array(folder / f"{side}_{k}.npy") for k in (*keys, "sparse_weights")}
    ids = json.loads((folder / f"{side}_ids.json").read_text(encoding="utf-8"))
    texts = json.loads((folder / f"{side}_texts.json").read_text(encoding="utf-8"))
    return arrays, ids, texts


def _first_stages(d: dict, texts: list[str]) -> dict:
    """Every first stage Index.create builds, with its default settings."""
    n = len(texts)
    flat, offsets = d["colbert_flat"], d["colbert_offsets"]
    csr = sp.csr_matrix((d["sparse_weights"], d["sparse_ids"], d["sparse_indptr"]), (n, VOCAB))
    dense = DenseStage(d["dense"])
    lexical = LexicalStage(*LexicalStage.arrays(csr).values(), n_docs=n)
    a = BM25Stage.arrays(texts)
    terms = a["bm25/terms.npy"].tobytes().decode().split("\n")
    s, _ = SMVEStage.arrays(flat, offsets, SMVE_BALANCED)
    m = MuveraStage.arrays(flat, offsets, MUVERA_DEFAULT)
    return {
        "dense": dense,
        "lexical": lexical,
        "hybrid": HybridStage(dense, lexical, HYBRID_ALPHA),
        "bm25": BM25Stage(terms, *list(a.values())[1:], n_docs=n),
        "smve": SMVEStage(
            *(s[f"smve/{x}.npy"] for x in ("indptr", "rows", "weights")),
            n,
            make_anchors(flat.shape[1], SMVE_BALANCED["w"], SMVE_BALANCED["seed"]),
            SMVE_BALANCED["k"],
            SMVE_BALANCED["reps"],
            s["smve/center.npy"],
        ),
        "muvera": MuveraStage(
            *(m[f"muvera/{x}.npy"] for x in ("fdes", "hyperplanes", "projection")),
            MUVERA_DEFAULT["k_sim"],
        ),
    }


def _real_queries(q: dict, ids: list[str], texts: list[str], judged) -> dict[str, Encoded]:
    row = {qid: i for i, qid in enumerate(ids)}
    off, ptr = q["colbert_offsets"], q["sparse_indptr"]
    out = {}
    for qid in judged:
        i = row[qid]
        out[qid] = Encoded(
            q["colbert_flat"][off[i] : off[i + 1]],
            q["dense"][i],
            (q["sparse_ids"][ptr[i] : ptr[i + 1]], q["sparse_weights"][ptr[i] : ptr[i + 1]]),
            texts[i],
        )
    return out


def main(name: str, n_queries: int, seed: int) -> None:
    start = time.perf_counter()
    folder = Path(os.environ["LATERANK_GOLDEN_DIR"]).expanduser() / f"{name}_bgem3"
    docs, doc_ids, doc_texts = _side(folder, "docs")
    queries, query_ids, query_texts = _side(folder, "queries")
    qrels = json.loads((folder / "qrels_test.json").read_text(encoding="utf-8"))
    real = _real_queries(queries, query_ids, query_texts, qrels)
    n, dim = len(doc_ids), docs["colbert_flat"].shape[1]

    print(f"{name}: building first stages over {n} documents ...", flush=True)
    first = _first_stages(docs, doc_texts)
    maxsim = MaxSimReranker(docs["colbert_flat"], docs["colbert_offsets"])
    stages = {**first, "maxsim": maxsim, EXHAUSTIVE: maxsim}
    pipes = candidates(first, n)
    print(f"built in {time.perf_counter() - start:.0f} s; {len(pipes)} candidates", flush=True)

    last = [None]

    def search(p, q, k):
        if p is not last[0]:
            last[0] = p
            print(f"  {time.perf_counter() - start:6.0f} s  {p}", flush=True)
        return p.run(stages, q, k)

    ignore = name in IGNORE_SELF
    ev = evaluate(search, doc_ids.__getitem__, real, qrels, pipes[:1], 1, dim, ignore)
    got = ev.scores[0].ndcg_at_10
    if abs(got - GOLDEN_EXHAUSTIVE[name]) > TOL:  # the data or the ArguAna rule loaded wrong
        sys.exit(f"sanity gate: exhaustive nDCG@10 {got:.4f}, golden {GOLDEN_EXHAUSTIVE[name]}")
    scores = (
        ev.scores
        + evaluate(search, doc_ids.__getitem__, real, qrels, pipes[1:], 1, dim, ignore).scores
    )

    print(f"encoding {n_queries} span queries with BGE-M3 ...", flush=True)
    encoder = BGEM3Encoder(revision=REVISION)
    spans = [(r, encoder.encode_query(s)) for r, s in span_queries(doc_texts, n_queries, seed)]
    report = tune(search, spans, pipes, Budget(1e12))

    by_name = {s.pipeline: s for s in scores}
    rows = []
    for budget in [pt.p50_ms for pt in report.frontier] + [math.inf]:
        fits = [pt for pt in report.frontier if pt.p50_ms <= budget]  # tune's own rule
        pick = fits[-1].pipeline
        within = [s for s in scores if s.p50_ms <= budget]
        label = max(within, key=lambda s: (s.ndcg_at_10, -s.p50_ms)) if within else None
        gap = label.ndcg_at_10 - by_name[pick].ndcg_at_10 if label else None
        rows.append((budget, pick, label, gap))
    bad = [b for b, _, _, g in rows if g is not None and g > MAX_GAP]
    verdict = "holds" if not bad else "fails at B = " + ", ".join(f"{b:.1f}" for b in bad) + " ms"

    md = [
        f"## {name}: {verdict}",
        "",
        f"{len(real)} judged queries, {report.n_queries} span queries (seed {seed}), {n} docs, "
        f"ignore_identical_ids={ignore}, run took {(time.perf_counter() - start) / 60:.1f} min.",
        "",
        "| pipeline | nDCG@10 | R@100 | real p50 ms | agreement | span p50 ms |",
        "|---|---|---|---|---|---|",
    ]
    front = {pt.pipeline: pt for pt in report.frontier}
    for s in scores:
        pt = front.get(s.pipeline)
        tail = f"{pt.score:.4f} | {pt.p50_ms:.1f}" if pt else "- | -"
        head = f"| `{s.pipeline}` | {s.ndcg_at_10:.4f} | {s.recall_at_100:.4f} | {s.p50_ms:.1f}"
        md.append(f"{head} | {tail} |")
    md += ["", "Agreement and span p50 are shown for tune's frontier only.", ""]
    md += [
        "| budget ms | tune pick | its nDCG@10 | label pick | its nDCG@10 | gap |",
        "|---|---|---|---|---|---|",
    ]
    for budget, pick, label, gap in rows:
        b = "no limit" if budget == math.inf else f"{budget:.1f}"
        lab = f"`{label.pipeline}` | {label.ndcg_at_10:.4f}" if label else "none fits | -"
        md.append(
            f"| {b} | `{pick}` | {by_name[pick].ndcg_at_10:.4f} | {lab} | "
            f"{'-' if gap is None else f'{gap:+.4f}'} |"
        )
    text = "\n".join(md) + "\n"
    print("\n" + text)
    if not OUT.exists():
        OUT.write_text("# Tune validation (spec M2-3)\n\nWritten by scripts/validate_tune.py.\n")
    with OUT.open("a", encoding="utf-8") as f:
        f.write("\n" + text)
    print(f"appended to {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, choices=sorted(GOLDEN_EXHAUSTIVE))
    ap.add_argument("--n-queries", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args.dataset, args.n_queries, args.seed)
