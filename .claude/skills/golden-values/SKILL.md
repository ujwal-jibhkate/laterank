---
name: golden-values
description: The smve-lab reference numbers that laterank must reproduce (nDCG@10 and Recall@100 per pipeline and dataset), their tolerances, and where the source files are. Use when writing or checking golden tests, quoting any result, or judging whether a change broke retrieval quality.
---

# Golden values

**Never quote these from memory.** Every number comes from a file in
`~/Projects/smve-lab/results/`. If a number here disagrees with the file, the file wins. Stop and
tell the owner.

Sources:
- `results/cross_dataset/summary.md`: all three datasets
- `results/scifact_bgem3/first_stage/summary.md`: SciFact, with storage and latency

Embeddings: `~/Projects/smve-lab/artifacts/embeddings/{scifact,nfcorpus,arguana}_bgem3/` (BGE-M3,
about 14 GB in total; never commit them). Tests read the path from `LATERANK_GOLDEN_DIR` and skip
when it's unset.

## nDCG@10 (tolerance ±0.002)

| pipeline | SciFact | NFCorpus | ArguAna |
|---|---|---|---|
| `exhaustive` (MaxSim over everything) | 0.6991 | 0.3441 | 0.4841 |
| `bm25` | 0.6896 | 0.3279 | 0.4919 |
| `dense` | 0.6542 | 0.3165 | 0.5266 |
| `lexical` | 0.6376 | 0.2857 | 0.3405 |
| `hybrid` (dense + lexical) | 0.6842 | 0.3287 | 0.4986 |
| `bm25 > maxsim@100` | 0.7086 | 0.3427 | 0.4884 |
| `dense > maxsim@100` | 0.6966 | 0.3393 | 0.4844 |
| `lexical > maxsim@100` | 0.7048 | 0.3321 | 0.4877 |
| `hybrid > maxsim@100` | 0.6977 | 0.3418 | 0.4849 |

## Recall@100 (tolerance ±0.002)

| pipeline | SciFact | NFCorpus | ArguAna |
|---|---|---|---|
| `exhaustive` | 0.9203 | 0.2914 | 0.9666 |
| `bm25` | 0.9249 | 0.2572 | 0.9730 |
| `dense` | 0.8970 | 0.2825 | 0.9815 |
| `lexical` | 0.8980 | 0.2335 | 0.9303 |
| `hybrid` | 0.9237 | 0.2886 | 0.9879 |

## SMVE and MUVERA: a band, not a point

laterank draws its random anchors with its own portable generator, not torch's, so its SMVE and
MUVERA scores won't match smve-lab digit for digit. smve-lab saw SMVE nDCG@10 move by up to 0.038
across seeds.
- SMVE `balanced` (w=65,536, k=32, centered), SciFact nDCG@10 alone: must land in
  **[0.537, 0.572]** (smve-lab's fixed setting scored 0.5471 with seed 0).
- With `> maxsim@100` on top, smve-lab's fixed setting scored 0.6898 (seed 0). Tolerance
  **±0.01**, i.e. [0.6798, 0.6998]. Confirmed in M1-4 across laterank seeds 0–2 (table below).
  smve-lab's seeds 0–2 alone: 0.5471 / 0.5618 / 0.5599 (`results/scifact_bgem3/smve/
  w65536_k32_center_seed*/summary.json`); only seed 0 has a rerank number.

laterank SMVE `balanced` on SciFact, measured 2026-10-12 (M1-4), BGE-M3 embeddings from smve-lab:

| seed | `smve` nDCG@10 | `smve` Recall@100 | `smve > maxsim@100` nDCG@10 |
|---|---|---|---|
| 0 | 0.5502 | 0.8497 | 0.6913 |
| 1 | 0.5469 | 0.8430 | 0.6864 |
| 2 | 0.5572 | 0.8547 | 0.6960 |

The rerank spread across seeds (0.0096) is about the ±0.01 tolerance, so don't tighten it.
laterank's Recall@100 runs ~0.01 below smve-lab's (0.857–0.862). Building the doc vectors takes
~5.2 min on the owner's Mac (10 cores; 8.3 min before the review fixed the slice size);
smve-lab's torch took ~3.3 min. Slicing doesn't change results, so the table above stands.

MUVERA `r40_k4_p32` (R=40, k_sim=4, d_proj=32, uncentered), SciFact. smve-lab, seed 0
(`results/scifact_bgem3/muvera/runs.csv`): nDCG@10 0.5530, Recall@100 0.858, `> maxsim@100` 0.6888
(`first_stage/summary.md`), 425 MB, build 13.7 s. laterank bands: **`muvera` alone [0.500, 0.580]**,
**`muvera > maxsim@100` 0.6888 ± 0.01**. Measured 2026-11-03 (M1-5), BGE-M3 embeddings from smve-lab:

| seed | `muvera` nDCG@10 | `muvera` Recall@100 | `muvera > maxsim@100` nDCG@10 | build |
|---|---|---|---|---|
| 0 | 0.5335 | 0.8423 | 0.6895 | 21.6 s |
| 1 | 0.5378 | 0.8560 | 0.6870 | 24.5 s |
| 2 | 0.5118 | 0.8540 | 0.6904 | 25.0 s |

Alone, seeds spread 0.026 (wider than SMVE's 0.010), so the first guess [0.533, 0.573] missed seed 2
and the owner widened it. Reranked, the spread is only 0.003. The golden test runs all three seeds
(~2 min). Build takes ~22-25 s on the owner's Mac against torch's 13.7 s.

## ColBERT models: laterank's own measurement, no smve-lab reference

Measured 2026-11-04 (M1-7) by `tests/golden/test_colbert.py`, SciFact full corpus, `bm25 >
maxsim@100`, Apple-silicon Mac (device auto), sentence-transformers 6.2.0. The test only floors
nDCG@10 at 0.60 (it catches broken vectors); these are not tolerance-checked reference values.
For scale: BGE-M3 `bm25 > maxsim@100` is 0.7086 (above), with a 718 KB/doc token store
(359 tokens x 1024 x 2 B, same KB = 1024 B unit as the table).

| model (pinned sha) | dim | nDCG@10 | Recall@100 | tokens/doc | KB/doc (float16) | encode 5,183 docs |
|---|---|---|---|---|---|---|
| answerdotai/answerai-colbert-small-v1 `934fa8bb` | 96 | 0.7384 | 0.9249 | 235.2 | 44.1 | 31 s |
| mixedbread-ai/mxbai-edge-colbert-v0-32m `bb13a29e` | 64 | 0.7453 | 0.9249 | 267.4 | 33.4 | 46 s |

Recall@100 is BM25's, since the rerank reorders the same 100 documents. The two nDCG@10 values
differ by 0.007 on one dataset and one run each (noise not measured), so quality hardly separates
them; storage does (mxbai 24% smaller per document).

## Conditions behind the numbers

- **BM25**: Lucene variant (k1=1.2, b=0.75), using smve-lab's ASCII regex tokenizer and sklearn
  English stopwords. A different tokenizer gives different numbers, so the golden test must use
  the smve-lab tokenizer mode.
- **hybrid**: score = dense + 0.3 · lexical.
- **ArguAna**: drop hits whose id equals the query id (`ignore_identical_ids`), as smve-lab did.
- **Latencies** in smve-lab exclude query encoding (queries were pre-encoded). Never compare
  end-to-end laterank latency to them without saying so.
- All three datasets use the BEIR test split.
