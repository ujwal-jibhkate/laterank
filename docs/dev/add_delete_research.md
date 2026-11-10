# Adding and deleting documents: how others do it (researcher report, 2026-11-10)

Input for the 0.2 design spec (M5-2). Sources were read at the commits listed below.

## Answer

No library found both updates in place and gives results identical to a fresh build.

- **PLAID family** (ColBERT, fast-plaid, NextPlaid): edits chunk files in place and keeps the
  cluster centers from the first build. A delete leaves a hole or renumbers later documents.
- **Lucene / tantivy**: new documents go to new segments; deletes are a per-segment live-docs
  list; a background merge cleans up. BM25 statistics count deleted documents until the merge.
- **bm25s**: no add or delete; rebuild.

laterank already computes BM25 weights at open, so exact BM25 statistics are cheap. **Surprise:**
SMVE also depends on the whole corpus: the mean token vector `smve/center.npy`. The design doc
says only BM25 does.

## Evidence

- **ColBERT** (`colbert/index_updater.py`, cc4f3dc, 2025-10-14)
  - Changes stay in memory until `persist_to_disk()` (L176).
  - New documents use the existing codec, so centers are frozen (L90).
  - New documents fill the last chunk, then new chunks (L196-250).
  - Delete rewrites the chunk in place and sets the length to 0 (L515-518): a hole, ids don't move.
  - Plain `torch.save` / `open("w")`, no temp file: a crash can leave a broken index.
- **NextPlaid** (`next-plaid/src/update.rs`, `delete.rs`, c6c091d, 2026-10-10)
  - Small adds are buffered (`buffer_size` 100); under `start_from_scratch` (999 docs) it rebuilds
    (update.rs L86-104). New outliers get extra k-means centers.
  - Delete renumbers later documents (delete.rs L244-246).
  - Buffer files are atomic (`atomic_write_file`), chunk rewrites are not (`File::create`,
    delete.rs L122-279).
- **fast-plaid** (`rust/index/delete.rs` L44-72, df2be46, 2026-09-10): drops deleted documents
  and rebuilds the IVF; ids are positional and shift.
- **PyLate** (3389837, 2026-06-14)
  - fast-plaid wrapper keeps pickled string-id → internal-id maps, rewritten after each delete
    (`pylate/indexes/fast_plaid.py` L233-276).
  - Voyager deletes use HNSW `mark_deleted`, a tombstone (`voyager.py` L254-275).
- **RAGatouille** (`ragatouille/models/index.py`, e75b8a9, 2025-05-17)
  - `add` rebuilds everything under 5,000 documents or when the batch is over 5% of the index;
    otherwise ColBERT's `IndexUpdater` (L364-414).
  - `delete` always uses `IndexUpdater` (L416-438).
- **bm25s** 0.3.16 (2026-10-28): `index`, `save`, `load`, `retrieve`; no add or delete.
- **tantivy** 0.26.1
  - `meta.json` is the commit point, written with `atomic_write` (`segment_updater.rs` L39-48).
  - Deletes are a per-segment alive bitset, `.del` (`segment_component.rs` L29).
  - BM25 sums `max_doc()` and token counts over segments at search time (`query/bm25.rs` L28-44);
    `max_doc` includes deleted documents.
  - Default merge policy waits for 8 segments (`log_merge_policy.rs` L10).
- **Lucene** 10
  - New files become visible only when a new `segments_N` is written (`IndexWriter.java` L179-190).
  - `TieredMergePolicy` allows up to 20% deleted documents before merging (L82).
  - BM25 statistics "may include statistics on deleted documents that have not yet been merged
    away" (CollectionStatistics javadoc, 10.0.0).
- **laterank today**
  - BM25 stores raw counts and sorted terms; idf and average length computed at open
    (`components/bm25.py` L17, L104-111).
  - SMVE center is `token_mean(flat)` over the build's tokens (`components/smve.py` L78).
  - Dense, lexical and MUVERA depend only on each document plus a seed.
  - `top_k` breaks ties by lower row (`kernels/topk.py` L35), so row order affects results.
  - `Index.open` refuses more than one segment (`orchestration/index.py` L304-307).

## Options (simplest first)

- **A. Rewrite into one new segment on every change.** Read stored arrays (no re-encoding),
  recompute every stage over surviving + new documents, write the next segment with
  `write_segment`, swap the manifest atomically, remove the old folder. Identical to a fresh build
  (BM25 and SMVE center included); existing crash safety covers it; `open` stays single-segment;
  manifest format unchanged. Cost: rewrites the whole index each time (~3.8 GB for SciFact with
  BGE-M3, time not measured). Differs from design §6 ("writes a new segment folder instead of
  rebuilding everything").
- **B. Append segments + deleted-rows list + later compaction** (Lucene/tantivy). Needs
  `format_version` 2, cross-segment row mapping in `open`, masking in every stage. BM25 can match
  a fresh build if terms are merged across segments; SMVE won't unless new segments reuse the
  first center. `compact()` is option A.
- **C. Edit files in place** (PLAID). Rejected: not crash-safe, breaks the atomic-write rule.

Either A or B: `add` must refuse ids already in the index.

## Open questions

- How long option A takes on a real BGE-M3 index.
- Whether a frozen SMVE center changes nDCG measurably.
- Lucene's live-docs file format (codec docs not checked).
- Whether fast-plaid's Python layer stages files atomically (`delete` mentions "staging" from a
  "merged copy", L1334-1337; not traced).
