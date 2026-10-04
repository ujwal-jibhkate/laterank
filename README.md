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

## Working on laterank

Start with [START_HERE.md](START_HERE.md): the daily workflow, the spec rules, and every command.

## License

Apache-2.0 — see [LICENSE](LICENSE).
