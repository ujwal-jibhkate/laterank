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
| `hybrid` | 0.9237 | 0.2886 | 0.9879 |

## SMVE and MUVERA: a band, not a point

laterank draws its random anchors with its own portable generator, not torch's, so its SMVE and
MUVERA scores won't match smve-lab digit for digit. smve-lab saw SMVE nDCG@10 move by up to 0.038
across seeds.
- SMVE `balanced` (w=65,536, k=32, centered), SciFact nDCG@10 alone: must land in
  **[0.537, 0.572]** (smve-lab's fixed setting scored 0.5471 with seed 0).
- With `> maxsim@100` on top, smve-lab's fixed setting scored 0.6898. The proposed tolerance is
  ±0.01, because the rerank absorbs most first-stage noise. This is provisional: confirm it
  across seeds in M1 before relying on it.

## Conditions behind the numbers

- **BM25**: Lucene variant (k1=1.2, b=0.75), using smve-lab's ASCII regex tokenizer and sklearn
  English stopwords. A different tokenizer gives different numbers, so the golden test must use
  the smve-lab tokenizer mode.
- **hybrid**: score = dense + 0.3 · lexical.
- **ArguAna**: drop hits whose id equals the query id (`ignore_identical_ids`), as smve-lab did.
- **Latencies** in smve-lab exclude query encoding (queries were pre-encoded). Never compare
  end-to-end laterank latency to them without saying so.
- All three datasets use the BEIR test split.
