"""BGE-M3 through FlagEmbedding: token, dense and lexical vectors from one model call.

Needs the `bge-m3` extra (`pip install "laterank[bge-m3]"`); FlagEmbedding and torch are imported
inside `BGEM3Encoder.__init__`, so importing this module costs nothing.

The model is pinned. FlagEmbedding has no `revision` argument and would take `main`, so we
download one commit with `snapshot_download` ourselves, record its sha in `EncoderInfo`, and hand
FlagEmbedding that folder. The folder's name in the Hugging Face cache IS the commit sha, which
is how a `revision=None` ("whatever main is today") becomes a recorded sha.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from laterank.components.base import Encoded
from laterank.components.encoder import EncodedDocs
from laterank.storage.manifest import EncoderInfo


def _pick_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class BGEM3Encoder:
    def __init__(
        self,
        hf_id: str = "BAAI/bge-m3",
        revision: str | None = None,
        *,
        max_length: int = 512,
        use_fp16: bool = True,
        device: str = "auto",
        batch_size: int = 32,
    ):
        """Download (or find in the cache) one commit of `hf_id` and load it.

        revision: a full 40-hex commit sha, or None for today's `main` (its sha is recorded in
            `self.info`). max_length is used for documents and queries alike, as in smve-lab.

        Raises:
            ImportError: the `bge-m3` extra isn't installed.
            ValueError: `revision` isn't a full commit sha.
        """
        try:
            from FlagEmbedding import BGEM3FlagModel
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError(
                'BGEM3Encoder needs the bge-m3 extra: pip install "laterank[bge-m3]"'
            ) from e
        if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
            raise ValueError(  # checked before the download: a branch or tag moves
                f"revision must be a full 40-character commit sha or None (today's main), "
                f"got {revision!r}"
            )
        folder = Path(snapshot_download(hf_id, revision=revision))  # named by the commit sha
        dim = json.loads((folder / "config.json").read_text(encoding="utf-8"))["hidden_size"]
        self.info = EncoderInfo(
            "bge-m3", hf_id, folder.name, dim, {"max_length": max_length, "fp16": use_fp16}
        )
        self.max_length, self.batch_size = max_length, batch_size
        self._model = BGEM3FlagModel(str(folder), use_fp16=use_fp16, devices=_pick_device(device))
        self._vocab = len(self._model.tokenizer)  # lexical term ids are token ids

    def _encode(self, texts: list[str]) -> dict:
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )

    def encode_docs(self, texts: Sequence[str]) -> EncodedDocs:
        texts = list(texts)
        if not texts:
            raise ValueError("encode_docs needs at least one text")
        dense, tokens, lexical = [], [], []
        for start in tqdm(range(0, len(texts), self.batch_size), desc="encoding"):
            out = self._encode(texts[start : start + self.batch_size])
            dense.append(out["dense_vecs"])
            tokens += [t.astype(np.float16) for t in out["colbert_vecs"]]
            lexical += out["lexical_weights"]  # {token id (str): weight}
        lengths = [len(t) for t in tokens]
        indptr = np.concatenate([[0], np.cumsum([len(d) for d in lexical])])
        cols = np.fromiter((int(k) for d in lexical for k in d), np.int64, indptr[-1])
        vals = np.fromiter((v for d in lexical for v in d.values()), np.float32, indptr[-1])
        return EncodedDocs(
            tokens=np.concatenate(tokens),
            offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
            dense=np.vstack(dense).astype(np.float32),
            lexical=sp.csr_matrix((vals, cols, indptr), shape=(len(texts), self._vocab)),
        )

    def encode_query(self, text: str) -> Encoded:
        out = self._encode([text])
        weights = out["lexical_weights"][0]
        return Encoded(
            out["colbert_vecs"][0].astype(np.float32),
            dense=out["dense_vecs"][0].astype(np.float32),
            lexical=(
                np.array([int(k) for k in weights], dtype=np.int64),
                np.array(list(weights.values()), dtype=np.float32),
            ),
            text=text,
        )
