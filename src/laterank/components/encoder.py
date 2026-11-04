"""Encoders: texts in, the vectors an index stores (documents) or searches with (queries) out.

An `Encoder` knows its model (`info`, which `Index` writes into the manifest). The torch-backed
ones live in their own modules and import torch only when built, so importing this file never
needs a model library. `load_encoder` rebuilds the encoder an index was made with.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import scipy.sparse as sp

from laterank.components.base import Encoded
from laterank.storage.manifest import EncoderInfo


@dataclass(frozen=True)
class EncodedDocs:
    """Encoded documents, in the layout `Index.create` takes (it validates them)."""

    tokens: np.ndarray  # (T, dim) float16, every document's token vectors
    offsets: np.ndarray  # (n + 1,) int64
    dense: np.ndarray | None  # (n, d) float32
    lexical: sp.csr_matrix | None  # (n, vocab) float32


class Encoder(Protocol):
    info: EncoderInfo

    def encode_docs(self, texts: Sequence[str]) -> EncodedDocs: ...

    def encode_query(self, text: str) -> Encoded:
        """Tokens, dense and lexical heads (as far as the model has them), and `text`."""
        ...


def _bgem3(info: EncoderInfo) -> Encoder:
    from laterank.components.bgem3 import BGEM3Encoder  # imports torch when built

    if info.hf_id is None:
        raise ValueError("this index records no hf_id for 'bge-m3', so the model can't be found")
    if set(info.settings) != {"max_length", "fp16"}:
        raise ValueError(
            f"bge-m3 settings must be max_length and fp16, got {sorted(info.settings)}"
        )
    return BGEM3Encoder(
        info.hf_id,
        info.revision,
        max_length=info.settings["max_length"],
        use_fp16=info.settings["fp16"],
    )


# EncoderInfo.name -> how to rebuild that encoder from its manifest entry
ENCODERS: dict[str, Callable[[EncoderInfo], Encoder]] = {"bge-m3": _bgem3}


def load_encoder(info: EncoderInfo) -> Encoder:
    """Rebuild the encoder an index was made with, from its manifest entry.

    Raises:
        ValueError: `info.name` is unknown, or the rebuilt encoder isn't the one `info` records
            (another revision, dim or settings would encode queries differently).
    """
    if info.name not in ENCODERS:
        raise ValueError(
            f"no encoder named {info.name!r} (known: {sorted(ENCODERS)}). An index built from "
            "precomputed vectors has no encoder: pass Encoded queries to Index.search yourself"
        )
    encoder = ENCODERS[info.name](info)
    if encoder.info != info:
        raise ValueError(
            f"the model that loaded ({encoder.info}) isn't the one the index was built with "
            f"({info}): queries would be encoded differently. Rebuild the index, or restore "
            "the recorded revision"
        )
    return encoder
