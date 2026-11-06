# laterank

> **Work in progress.** Nothing here is usable yet; the API, on-disk format and defaults will
> change before 0.1.0.

`laterank` is a small Python library for searching your own documents locally with
late-interaction models (BGE-M3 or ColBERT-style). It encodes your documents once into a plain
folder of NumPy arrays and JSON, finds about a hundred promising candidates with a cheap first
stage (BM25, dense, hybrid, SMVE or MUVERA), then re-orders them with exact MaxSim — so you get
close to exhaustive late-interaction quality at a fraction of the cost. Pipelines are short
strings like `hybrid > maxsim@100`, you can measure any of them on your own data, and the defaults
are the settings that won in our measurements on BEIR. It grows out of the research in
[smve-lab](https://github.com/ujwal-jibhkate/smve-lab); SMVE is TopK's method and is credited as such.

## Choosing a pipeline

`index.tune` tries every pipeline the index can run (exact search, each first stage alone, each
first stage followed by `maxsim@20` ... `maxsim@1000`) and picks the best one that fits a
latency budget. **Give it a few judged queries** (recommended): it keeps the highest nDCG@10.

```python
from laterank.orchestration.tune import Budget

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
in our tests (at sub-millisecond budgets; 0.042 above 1 ms); see [docs/dev/tune_validation.md](docs/dev/tune_validation.md).

## Working on laterank

Start with [START_HERE.md](START_HERE.md): the daily workflow, the spec rules, and every command.

## License

Apache-2.0 — see [LICENSE](LICENSE).
