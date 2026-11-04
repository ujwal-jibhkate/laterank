"""A ColBERT checkpoint through sentence-transformers' `MultiVectorEncoder`: token vectors only.

Needs the `colbert` extra (`pip install "laterank[colbert]"`); sentence-transformers and torch
are imported inside `ColBERTEncoder.__init__`, so importing this module costs nothing.

The model is pinned the way `BGEM3Encoder` pins BGE-M3: we `snapshot_download` one commit
ourselves, record its sha, and load from that folder. The checkpoint decides its own prefixes,
query expansion and punctuation skiplist; we read them off the loaded model into
`EncoderInfo.settings`, so the manifest determines the vectors whatever the repo's defaults are.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from tqdm import tqdm

from laterank.components.base import Encoded
from laterank.components.bgem3 import _pick_device
from laterank.components.encoder import EncodedDocs
from laterank.storage.manifest import EncoderInfo


class ColBERTEncoder:
    def __init__(
        self,
        hf_id: str,
        revision: str | None = None,
        *,
        device: str = "auto",
        batch_size: int = 32,
    ):
        """Download (or find in the cache) one commit of `hf_id` and load it.

        revision: a full 40-hex commit sha, or None for today's `main` (its sha is recorded in
            `self.info`). The ONNX copies in the repo are skipped (about 134 MB instead of 993).

        Raises:
            ImportError: the `colbert` extra isn't installed.
            ValueError: `revision` isn't a full commit sha.
        """
        try:
            from huggingface_hub import snapshot_download
            from sentence_transformers import MultiVectorEncoder
        except ImportError as e:
            raise ImportError(
                'ColBERTEncoder needs the colbert extra: pip install "laterank[colbert]"'
            ) from e
        if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
            raise ValueError(  # checked before the download: a branch or tag moves
                f"revision must be a full 40-character commit sha or None (today's main), "
                f"got {revision!r}"
            )
        folder = Path(
            snapshot_download(hf_id, revision=revision, ignore_patterns=["*.onnx", "onnx/*"])
        )  # named by the commit sha
        model = MultiVectorEncoder(str(folder), device=_pick_device(device))
        transformer = model[0]
        mask = next((m for m in model.children() if type(m).__name__ == "MultiVectorMask"), None)
        expansion = transformer.query_expansion
        settings = {
            "document_length": transformer.document_length,
            "query_length": transformer.query_length,
            "query_expansion": None if expansion is None else dict(expansion),
            "query_prefix": model.prompts.get("query", ""),
            "document_prefix": model.prompts.get("document", ""),
            "skiplist": [] if mask is None else sorted(mask.skiplist_words),
            "skiplist_tasks": [] if mask is None else sorted(mask.skiplist_tasks),
            "normalize": True,
            "vector_dtype": "float16",
        }
        self.info = EncoderInfo(
            "colbert", hf_id, folder.name, int(model.get_embedding_dimension()), settings
        )
        self.batch_size = batch_size
        self._model = model

    def encode_docs(self, texts: Sequence[str]) -> EncodedDocs:
        texts = list(texts)
        if not texts:
            raise ValueError("encode_docs needs at least one text")
        tokens = []
        for start in tqdm(range(0, len(texts), self.batch_size), desc="encoding"):
            out = self._model.encode_document(
                texts[start : start + self.batch_size],
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            tokens += [t.astype(np.float16) for t in out]
        lengths = [len(t) for t in tokens]
        return EncodedDocs(
            tokens=np.concatenate(tokens),
            offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
            dense=None,
            lexical=None,
        )

    def encode_query(self, text: str) -> Encoded:
        out = self._model.encode_query([text], convert_to_numpy=True, show_progress_bar=False)
        return Encoded(np.asarray(out[0], dtype=np.float32), text=text)
