# laterank

> **Pre-release.** It works end to end, but the API and on-disk format may change before 0.1.0.

`laterank` is a small Python library for searching your own documents locally with
late-interaction models (BGE-M3 or ColBERT-style). It encodes your documents once into a plain
folder of NumPy arrays and JSON, finds about a hundred promising candidates with a cheap first
stage (BM25, dense, hybrid, SMVE or MUVERA), then re-orders them with exact MaxSim — so you get
close to exhaustive late-interaction quality at a fraction of the cost. Pipelines are short
strings like `hybrid > maxsim@100`, you can measure any of them on your own data, and the defaults
are the settings that won in our measurements on BEIR. It grows out of the research in
[smve-lab](https://github.com/ujwal-jibhkate/smve-lab); SMVE is TopK's method and is credited as such.

## Install

You need Python 3.12 or newer. Pick the extra for the model you want (`colbert` or `bge-m3`):

```bash
pip install "laterank[colbert]"
```

Either extra brings in torch, which is a large download.

## Quick start

```python
from laterank import ColBERTEncoder, Index

docs = [
    {"id": "a", "text": "Statins lower cholesterol but can cause muscle pain."},
    {"id": "b", "text": "BM25 is a classic ranking function for search."},
    {"id": "c", "text": "Tea is made from the leaves of a plant."},
]
encoder = ColBERTEncoder("mixedbread-ai/mxbai-edge-colbert-v0-32m")
index = Index.build("./my_index", docs, encoder, bm25=True, pipeline="bm25 > maxsim@100")

for hit in index.search(encoder.encode_query("do statins hurt muscles?"), k=2):
    print(hit.rank, hit.id, round(hit.score, 3), hit.text)
```

To use the index in a later session, reopen it:

```python
from laterank import Index, load_encoder

index = Index.open("./my_index")
encoder = load_encoder(index.manifest.encoder)  # the exact model commit the index was built with
```

Each document needs a unique string `id` and a `text`, and can have a `title` and a `metadata`
dict; other keys are refused. The command line's `--docs` file uses BEIR's `_id` instead of `id`.

**Long documents are cut off.** BGE-M3 and mxbai keep only the first 512 tokens (`max_length` and
`document_length`), so split long documents into smaller ones before you build the index.

## Choosing a pipeline

`index.tune` tries every pipeline the index can run (exact search, each first stage alone, each
first stage followed by `maxsim@20` ... `maxsim@1000`) and picks the best one that fits a
latency budget. **Give it a few judged queries** (recommended): it keeps the highest nDCG@10.

```python
from laterank import Budget

queries = {"q1": encoder.encode_query("do statins hurt muscles?")}
qrels = {"q1": {"a": 1}}
report = index.tune(Budget(latency_ms=20), queries=queries, qrels=qrels)
print(report.table())
if report.best:  # None means no pipeline fits the budget
    index.set_default(report.best)
```

`queries` maps a query id to `encoder.encode_query(text)`, and `qrels` maps a query id to
`{doc_id: relevance}`, the same as `index.evaluate`.

Without labels, leave both out. Tune then cuts short spans out of your documents and picks the
pipeline that best agrees with exact MaxSim. That measures "agrees with MaxSim", not "finds
relevant documents": it can't pick a pipeline that beats MaxSim, and it can miss `bm25 > maxsim@d`,
which sometimes does (SciFact: 0.7086 vs 0.6991 for exact search). Up to 0.052 nDCG@10 was lost
in our tests (at sub-millisecond budgets; 0.042 above 1 ms); see [docs/dev/tune_validation.md](https://github.com/ujwal-jibhkate/laterank/blob/main/docs/dev/tune_validation.md).

## Command line

```bash
laterank build ./idx --dataset scifact --encoder colbert \
    --hf-id mixedbread-ai/mxbai-edge-colbert-v0-32m
laterank search ./idx "statin side effects" -k 5
laterank evaluate ./idx --dataset scifact --pipeline "bm25 > maxsim@100" --pipeline exhaustive
laterank tune ./idx --budget-ms 20 --dataset scifact --apply
```

**Quote your pipelines.** Without quotes, the shell reads `>` as "write the output to a file".

`--dataset` takes a folder holding `corpus.jsonl`, or a BEIR name, which is downloaded the first
time. To index your own documents, use `--docs FILE` instead: a JSONL file with `_id`, `text` and
an optional `title` on each line. `--encoder` is required, and `search`, `evaluate` and `tune` read
the model from the index. Every `search` loads the model again (about 7 s with mxbai on a Mac), so
for many queries use the Python API. Without `--dataset`, `tune` measures agreement with MaxSim
(see above).

## Contributing

See [CONTRIBUTING.md](https://github.com/ujwal-jibhkate/laterank/blob/main/CONTRIBUTING.md): setup, tests, and the rules the code follows.

## License

Apache-2.0 — see [LICENSE](https://github.com/ujwal-jibhkate/laterank/blob/main/LICENSE).
