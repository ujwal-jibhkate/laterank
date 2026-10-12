"""The `bm25` first stage: the classic lexical baseline, over the documents' own words.

For a query q and document d:

    BM25(q, d) = sum over query words t of  count(t, q) * weight(t, d)
    weight(t, d) = idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg_len))
    idf(t)       = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

  tf       how often t occurs in d          N       number of documents
  df(t)    how many documents contain t     len(d)  document length in words
  k1       how fast repeated words saturate (1.2): the 5th "protein" adds less than the 1st
  b        how much long documents are penalised (0.75)

Text is lowercased, split into ASCII words, stop words removed and stemmed (Snowball English),
roughly what Lucene/Pyserini do for the standard BEIR BM25 numbers.

idf and avg_len depend on the whole corpus, so a segment stores only raw word counts (postings
of tf) and its sorted terms; the weights are computed when the index is opened (design §6).
Lucene and Tantivy store raw counts the same way.

Ported from smve-lab's bm25.py, which stored the finished weights. The arithmetic follows its
order and float32 casts, so the weights are the same bit for bit.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import scipy.sparse as sp
import Stemmer

from laterank.components.base import Candidates, Encoded
from laterank.components.stopwords import ENGLISH_STOP_WORDS
from laterank.kernels.inverted import score_postings
from laterank.kernels.topk import top_k
from laterank.storage.segment import Segment

K1, B = 1.2, 0.75
TOKENIZER = "ascii-snowball-en"  # recorded in the manifest; another tokenizer gets a new name

_WORD = re.compile(r"[a-z0-9]+")
_stem = Stemmer.Stemmer("english")


def tokenize(text: str) -> list[str]:
    """Lowercase, words = [a-z0-9]+, drop ENGLISH_STOP_WORDS, Snowball English stem (PyStemmer)."""
    words = [w for w in _WORD.findall(text.lower()) if w not in ENGLISH_STOP_WORDS]
    return _stem.stemWords(words)


TERMS, INDPTR, ROWS, TF = "bm25/terms.npy", "bm25/indptr.npy", "bm25/rows.npy", "bm25/tf.npy"


class BM25Stage:
    def __init__(
        self,
        terms: Sequence[str],
        indptr: np.ndarray,
        rows: np.ndarray,
        tf: np.ndarray,
        n_docs: int,
        k1: float = K1,
        b: float = B,
    ):
        """terms: sorted, unique; term i's postings are [indptr[i], indptr[i+1]) of rows/tf.

        rows (nnz,) int and tf (nnz,) int >= 1 are the postings of raw word counts. Computes the
        float32 BM25 weight of every posting once.
        """
        terms = list(terms)
        if (
            indptr.ndim != 1
            or rows.ndim != 1
            or tf.ndim != 1
            or any(a.dtype.kind not in "iu" for a in (indptr, rows, tf))
            or len(indptr) != len(terms) + 1
            or indptr[0] != 0
            or indptr[-1] != len(rows)
            or rows.shape != tf.shape
            or np.any(np.diff(indptr) < 0)
        ):
            raise ValueError(
                f"inconsistent postings: {len(terms)} terms, indptr {indptr.shape} {indptr.dtype}, "
                f"rows {rows.shape} {rows.dtype}, tf {tf.shape} {tf.dtype}"
            )
        if any(s >= t for s, t in pairwise(terms)):
            raise ValueError("terms must be sorted and unique")
        if len(rows) and (rows.min() < 0 or rows.max() >= n_docs):
            raise ValueError(f"postings point at rows outside the {n_docs} documents")
        if len(tf) and tf.min() < 1:
            raise ValueError("word counts (tf) must be >= 1")
        real = all(type(x) in (int, float) and math.isfinite(x) for x in (k1, b))  # not bool
        if not (real and k1 >= 0 and 0 <= b <= 1):
            raise ValueError(f"need k1 >= 0 and 0 <= b <= 1 as finite numbers, got {k1!r}, {b!r}")
        doc_len = np.bincount(rows, weights=tf, minlength=n_docs)  # exact: float64 of ints
        total = doc_len.sum()
        if total == 0:
            raise ValueError("no document has any word BM25 can index (all empty or stop words)")
        k1, b = float(k1), float(b)  # Python floats stay weak next to float32 arrays
        df = np.diff(indptr)
        idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        norm = k1 * (1 - b + b * doc_len.astype(np.float32) / np.float32(total / n_docs))
        t = tf.astype(np.float32)
        term_of = np.repeat(np.arange(len(terms)), np.diff(indptr))
        # ponytail: every weight computed at open and held in RAM (O(nnz)); for a big corpus,
        # weight only the postings each query touches
        self.weights = idf[term_of] * t * (k1 + 1) / (t + norm[rows])  # float32 throughout
        self.indptr, self.rows, self.n_docs = indptr, rows, n_docs
        self.vocab = len(terms)
        self._ids = {term: i for i, term in enumerate(terms)}

    @staticmethod
    def arrays(texts: Sequence[str]) -> dict[str, np.ndarray]:
        """One text per document -> the segment arrays this stage stores."""
        docs = [tokenize(t) for t in texts]
        terms = sorted({w for d in docs for w in d})
        ids = {t: i for i, t in enumerate(terms)}
        rows = np.repeat(np.arange(len(docs)), [len(d) for d in docs])
        cols = np.fromiter((ids[w] for d in docs for w in d), dtype=np.int64, count=len(rows))
        ones = np.ones(len(rows), dtype=np.int32)  # integer counts stay exact (float32 doesn't)
        counts = sp.csc_matrix((ones, (rows, cols)), shape=(len(docs), len(terms)))
        counts.sum_duplicates()  # repeats of a word in a document add up to tf; rows ascending
        joined = np.frombuffer("\n".join(terms).encode("utf-8"), dtype=np.uint8)
        return {
            TERMS: joined,
            INDPTR: counts.indptr.astype(np.int64),
            ROWS: counts.indices.astype(np.int32),
            TF: counts.data.astype(np.int32),
        }

    @classmethod
    def from_segment(cls, seg: Segment, cfg: dict, n_docs: int) -> BM25Stage:
        """Open the stored counts; ValueError for another tokenizer or vocab size."""
        if cfg["tokenizer"] != TOKENIZER:
            raise ValueError(
                f"bm25 tokenizer {cfg['tokenizer']!r} is unknown to this laterank (it knows "
                f"{TOKENIZER!r}); upgrade laterank"
            )
        raw = seg.array(TERMS, mmap=False)
        if raw.dtype != np.uint8 or raw.ndim != 1:
            raise ValueError(f"bm25 terms must be 1-D uint8, got {raw.shape} {raw.dtype}")
        text = raw.tobytes().decode("utf-8")
        terms = text.split("\n") if text else []
        arrays = (seg.array(INDPTR), seg.array(ROWS), seg.array(TF))
        stage = cls(terms, *arrays, n_docs, cfg["k1"], cfg["b"])
        if stage.vocab != cfg["vocab"]:
            raise ValueError(f"bm25 postings cover {stage.vocab} terms, expected {cfg['vocab']}")
        return stage

    def scores(self, q: Encoded) -> np.ndarray:
        """(n_docs,) float32: sum over query words t of count(t in q) * weight(t, d).

        Query words not in the corpus vocabulary are dropped: they match nothing.
        """
        if q.text is None:
            raise ValueError("the bm25 stage needs the query's text (Encoded.text)")
        ids = [self._ids[w] for w in tokenize(q.text) if w in self._ids]
        terms, counts = np.unique(np.asarray(ids, dtype=np.int64), return_counts=True)
        return score_postings(self.indptr, self.rows, self.weights, self.n_docs, terms, counts)

    def search(self, q: Encoded, k: int) -> Candidates:
        scores = self.scores(q)
        rows = top_k(scores, k)
        return Candidates(rows, scores[rows])
