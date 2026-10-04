"""Term-at-a-time scoring over an inverted index (BM25, BGE-M3 lexical, SMVE).

A sparse document matrix stored row by row (CSR, one row per document) answers "which terms
does document d contain?". Search needs the opposite question, "which documents contain term
t?", so we store it column by column:

    term t  ->  posting list: [doc rows containing t], [their weights]

That is an inverted index, and it is exactly the CSC ("compressed sparse column") layout of the
document matrix: postings for term t live in [indptr[t], indptr[t+1]).

Scoring a query: start every document at 0; for each query term t with weight q_t, add
q_t * w(t, d) to every document d in t's posting list. Documents sharing no term with the query
are never touched. Work = the summed posting-list lengths of the query terms ("postings read"),
which drives latency. This is exhaustive (no WAND/MaxScore pruning), so scores equal the plain
sparse matrix product exactly.

Ported from smve-lab's inverted_index.py, split into pure functions over the three arrays so
that storage can save and memory-map them without this module knowing about files.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def ragged_ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Concatenate range(s, s + l) for every (s, l) pair, without a Python loop.

    e.g. starts=[10, 50], lengths=[3, 2] -> [10, 11, 12, 50, 51]
    """
    starts = np.asarray(starts, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    # Each output position = its range's start + its offset inside that range.
    range_id = np.repeat(np.arange(len(lengths)), lengths)
    offset = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    return starts[range_id] + offset


def to_postings(doc_matrix: sp.spmatrix | sp.sparray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn an (n_docs, n_terms) sparse document matrix into inverted-index arrays.

    Returns:
        indptr: (n_terms + 1,) int64; term t's postings are [indptr[t], indptr[t+1]).
        rows: (nnz,) int32 document row of each posting, ascending within a term.
        weights: (nnz,) float32 weight of the term in that document.
    """
    csc = sp.csc_matrix(doc_matrix, dtype=np.float32)
    csc.sum_duplicates()  # also sorts rows within each column
    return csc.indptr.astype(np.int64), csc.indices.astype(np.int32), csc.data


def postings_read(indptr: np.ndarray, terms: np.ndarray) -> int:
    """How many postings scoring `terms` will touch: the latency-driving work count."""
    terms = np.asarray(terms, dtype=np.int64)
    return int((indptr[terms + 1] - indptr[terms]).sum())


def score_postings(
    indptr: np.ndarray,
    rows: np.ndarray,
    weights: np.ndarray,
    n_docs: int,
    terms: np.ndarray,
    q_weights: np.ndarray,
) -> np.ndarray:
    """Exhaustive term-at-a-time scoring of one query.

    Args:
        indptr, rows, weights: inverted-index arrays from `to_postings`.
        n_docs: number of documents (length of the output).
        terms: (n_q_terms,) term ids of the query.
        q_weights: (n_q_terms,) query weight of each term.

    Returns:
        (n_docs,) float32 scores: sum over query terms t of q_t * w(t, d).
    """
    terms = np.asarray(terms, dtype=np.int64)
    lengths = indptr[terms + 1] - indptr[terms]
    pos = ragged_ranges(indptr[terms], lengths)  # every posting entry to read
    contrib = weights[pos] * np.repeat(np.asarray(q_weights, dtype=np.float32), lengths)
    # bincount accumulates in float64, then we hand back float32 like every other scorer.
    return np.bincount(rows[pos], weights=contrib, minlength=n_docs).astype(np.float32)
