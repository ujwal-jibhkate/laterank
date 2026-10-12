"""What passes between components: an encoded query in, ranked candidates out.

A first stage picks candidates from the whole corpus; a reranker re-scores only those. Both
speak in global document rows (0 .. n_docs - 1), best first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Encoded:
    """One encoded query: its token vectors, and a single dense vector if the encoder has one."""

    tokens: np.ndarray  # (t, dim) float, t >= 1
    dense: np.ndarray | None = None  # (d,) float

    def __post_init__(self) -> None:
        t = np.asarray(self.tokens)
        if t.ndim != 2 or len(t) == 0 or not np.issubdtype(t.dtype, np.floating):
            raise ValueError(
                f"tokens must be a (t, dim) float array with t >= 1, got {t.shape} {t.dtype}"
            )
        if self.dense is not None:
            d = np.asarray(self.dense)
            if d.ndim != 1 or not np.issubdtype(d.dtype, np.floating):
                raise ValueError(f"dense must be a 1-D float array, got {d.shape} {d.dtype}")
            object.__setattr__(self, "dense", d)
        object.__setattr__(self, "tokens", t)  # lists become arrays


@dataclass(frozen=True)
class Candidates:
    """Ranked documents: global rows (int64) and their scores (float32), best first."""

    rows: np.ndarray  # (n,) int64
    scores: np.ndarray  # (n,) float32

    def __post_init__(self) -> None:
        if self.rows.dtype != np.int64 or self.scores.dtype != np.float32:
            raise ValueError(
                f"rows must be int64 and scores float32, "
                f"got {self.rows.dtype} and {self.scores.dtype}"
            )
        if self.rows.ndim != 1 or self.rows.shape != self.scores.shape:
            raise ValueError(
                f"rows and scores must be 1-D and the same length, got {self.rows.shape} and "
                f"{self.scores.shape}"
            )


class FirstStage(Protocol):
    def search(self, q: Encoded, k: int) -> Candidates:
        """The top min(k, n_docs) documents of the whole corpus."""
        ...


class Reranker(Protocol):
    def rerank(self, q: Encoded, cands: Candidates) -> Candidates:
        """The same rows, re-scored and re-ordered."""
        ...
