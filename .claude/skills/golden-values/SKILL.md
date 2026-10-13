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

## Conditions behind the numbers

- **BM25**: Lucene variant (k1=1.2, b=0.75), using smve-lab's ASCII regex tokenizer and sklearn
  English stopwords. A different tokenizer gives different numbers, so the golden test must use
  the smve-lab tokenizer mode.
- **hybrid**: score = dense + 0.3 · lexical.
- **ArguAna**: drop hits whose id equals the query id (`ignore_identical_ids`), as smve-lab did.
- **Latencies** in smve-lab exclude query encoding (queries were pre-encoded). Never compare
  end-to-end laterank latency to them without saying so.
- All three datasets use the BEIR test split.
