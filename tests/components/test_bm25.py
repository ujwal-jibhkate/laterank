import hashlib
import math
from collections import Counter

import numpy as np
import pytest

from laterank.components.base import Encoded
from laterank.components.bm25 import K1, B, BM25Stage, tokenize
from laterank.components.stopwords import ENGLISH_STOP_WORDS

# Computed from scikit-learn 1.9.1's ENGLISH_STOP_WORDS (the list smve-lab's BM25 used).
STOPWORDS_SHA256 = "40e0a284c5b9a220efffd18d4d739fbd3270091d6ce2c75b6effe289d3be5487"

# PyStemmer 3.1.0's English stems, as in smve-lab. A stemmer upgrade that changes any of these
# would make new queries miss words stored in existing indexes.
STEMS = {
    "proteins": "protein",
    "running": "run",
    "generalization": "general",
    "studies": "studi",
    "cells": "cell",
    "cellular": "cellular",
    "mutations": "mutat",
    "mutated": "mutat",
    "apoptosis": "apoptosi",
    "antibodies": "antibodi",
    "inflammatory": "inflammatori",
    "receptors": "receptor",
    "expression": "express",
    "regulated": "regul",
    "regulation": "regul",
    "diabetes": "diabet",
    "patients": "patient",
    "vaccination": "vaccin",
    "hypothesis": "hypothesi",
    "analyses": "analys",
}


def test_stopwords_match_sklearn():
    words = sorted(ENGLISH_STOP_WORDS)
    assert len(words) == 318
    assert hashlib.sha256("\n".join(words).encode()).hexdigest() == STOPWORDS_SHA256


def test_tokenize_lowercases_splits_drops_stopwords_and_stems():
    assert tokenize("The Proteins, running!") == ["protein", "run"]
    assert tokenize("IL-6 café") == ["il", "6", "caf"]  # ASCII words only: é splits a word
    assert tokenize("") == tokenize("the of and") == []


def test_stems_are_pinned():
    assert tokenize(" ".join(STEMS)) == list(STEMS.values())


TOKENS = np.ones((1, 2), dtype=np.float32)
DOCS = [
    "Proteins regulate cell growth.",
    "The cell divides; cells divide often, cells grow.",
    "",  # an empty document: length 0, scores 0
    "Diabetes patients and insulin.",
    "Growth of proteins in diabetes patients, a long study of growth and protein regulation.",
]


def _stage(texts):
    a = BM25Stage.arrays(texts)
    text = a["bm25/terms.npy"].tobytes().decode()
    terms = text.split("\n") if text else []
    return BM25Stage(terms, a["bm25/indptr.npy"], a["bm25/rows.npy"], a["bm25/tf.npy"], len(texts))


def _naive(texts, query):
    docs = [Counter(tokenize(t)) for t in texts]
    n, avg = len(docs), sum(sum(d.values()) for d in docs) / len(docs)
    out = []
    for d in docs:
        score = 0.0
        for t, c in Counter(tokenize(query)).items():
            df = sum(t in e for e in docs)
            if t in d:
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                tf, dl = d[t], sum(d.values())
                score += c * idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / avg))
        out.append(score)
    return out


@pytest.mark.parametrize(
    "query",
    [
        "cell growth proteins",
        "growth growth diabetes",  # a repeated word counts twice
        "zebrafish cell",  # an unknown word is dropped
        "zebrafish",  # nothing known: every document scores 0
    ],
)
def test_scores_match_the_formula(query):
    got = _stage(DOCS).scores(Encoded(TOKENS, text=query))
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, _naive(DOCS, query), rtol=1e-5)


def test_ties_in_row_order_and_text_required():
    stage = _stage(["apple pie", "banana", "apple pie", "cherry"])
    assert stage.search(Encoded(TOKENS, text="apple"), 99).rows.tolist() == [0, 2, 1, 3]
    with pytest.raises(ValueError, match="text"):
        stage.scores(Encoded(TOKENS))
    with pytest.raises(ValueError, match="str"):
        Encoded(TOKENS, text=3)


def _arrays(**over):
    a = {
        "terms": ["a", "b"],
        "indptr": np.array([0, 2, 3]),
        "rows": np.array([0, 1, 1]),
        "tf": np.array([1, 2, 1]),
        "n_docs": 2,
    }
    return a | over


@pytest.mark.parametrize(
    "over",
    [
        {"terms": ["b", "a"]},  # unsorted
        {"terms": ["a", "a"]},  # duplicate
        {"rows": np.array([0, 2, 1])},  # row >= n_docs
        {"tf": np.array([1, 0, 1])},  # tf < 1
        {"tf": np.array([1, 2])},  # length mismatch
        {"terms": ["a"]},  # indptr doesn't fit the terms
        {"k1": float("nan")},
        {"b": 1.5},
        {"k1": True},
    ],
)
def test_rejects_bad_input(over):
    with pytest.raises(ValueError):
        BM25Stage(**_arrays(**over))


def test_rejects_a_corpus_without_words():
    with pytest.raises(ValueError, match="no document"):
        _stage(["", "the of and"])
