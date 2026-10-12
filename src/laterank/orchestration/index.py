"""Index: an index folder, opened for search.

`Index.create` writes one segment (documents, the token store, and each first stage's arrays)
and then the manifest, which records the built stages and the default pipeline. `Index.open`
reads them back. `search` runs a pipeline for one already-encoded query and returns `Hit`s.

The token store belongs to the index, not to a stage: every pipeline can rerank with `maxsim`,
so rerankers have no entry in the manifest's `stages`.
"""

from __future__ import annotations

import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from laterank.components.base import Encoded
from laterank.components.dense import VECTORS, DenseStage
from laterank.components.hybrid import HYBRID_ALPHA, HybridStage
from laterank.components.lexical import WEIGHTS, LexicalStage
from laterank.components.maxsim import TOKENS, MaxSimReranker
from laterank.errors import IndexExistsError, IndexFormatError
from laterank.orchestration.pipeline import Pipeline, parse_pipeline
from laterank.storage.manifest import EncoderInfo, Manifest, write_manifest
from laterank.storage.segment import Segment, open_index, write_segment

STORED_STAGES = {"dense": DenseStage, "lexical": LexicalStage}  # manifest "type" -> class
FIRST_STAGE_TYPES = {*STORED_STAGES, "hybrid"}  # hybrid is built from dense + lexical
RERANKERS = {"maxsim": MaxSimReranker}


@dataclass(frozen=True)
class Hit:
    id: str
    rank: int  # starts at 1
    score: float  # the last stage's score; not comparable across pipelines
    text: str | None
    metadata: dict


def _check_pipeline(p: Pipeline, first_stages: Mapping[str, object]) -> None:
    if p.first not in first_stages:
        raise ValueError(
            f"'{p}' starts with '{p.first}', but this index has the first stages "
            f"{sorted(first_stages)}: start the pipeline with one of them, or build the index "
            f"with that stage (e.g. Index.create(..., dense=..., lexical=...))"
        )
    unknown = [name for name, _ in p.rerankers if name not in RERANKERS]
    if unknown:
        raise ValueError(f"'{p}' uses unknown rerankers {unknown}; known: {sorted(RERANKERS)}")


class Index:
    def __init__(self, path: Path, manifest: Manifest, segment: Segment, pipeline: Pipeline):
        """Use `Index.create` or `Index.open`."""
        self.path, self.manifest = path, manifest
        self._default = pipeline
        self._docs = segment.docs
        n = len(self._docs)
        try:  # a hand-edited or damaged index must not escape as KeyError/IndexError
            self._first = {}
            # stored stages first, then hybrid, which reuses the dense and lexical ones
            for name, cfg in sorted(
                manifest.stages.items(), key=lambda s: s[1]["type"] == "hybrid"
            ):
                if cfg["type"] == "hybrid":
                    alpha = cfg["alpha"]
                    if type(alpha) not in (int, float) or not math.isfinite(alpha):  # not bool
                        raise ValueError(f"hybrid alpha must be a finite number, got {alpha!r}")
                    stage = HybridStage(self._first["dense"], self._first["lexical"], alpha)
                else:  # each stored stage checks its arrays against cfg and n
                    stage = STORED_STAGES[cfg["type"]].from_segment(segment, cfg, n)
                self._first[name] = stage
            reranker = {name: c.from_segment(segment) for name, c in RERANKERS.items()}
            self._stages = {**self._first, **reranker}
            tokens = self._stages["maxsim"]
            problems = [] if len(tokens.offsets) - 1 == n else ["token offsets"]
            if tokens.flat.shape[1] != manifest.encoder.dim:
                problems.append("token dim")
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise IndexFormatError(
                f"{path}: a stored array or a manifest stage is malformed ({e!r}). "
                "The index is damaged; rebuild it."
            ) from None
        if problems:
            raise IndexFormatError(
                f"{path}: {problems} don't match the {n} documents and the manifest's dims. "
                "The index is damaged; rebuild it."
            )

    @classmethod
    def create(
        cls,
        path: Path,
        docs: Sequence[Mapping],
        tokens: tuple[np.ndarray, np.ndarray],
        *,
        encoder: EncoderInfo,
        dense: np.ndarray | None = None,
        lexical: sp.spmatrix | sp.sparray | None = None,
        pipeline: str = "dense > maxsim@100",
    ) -> Index:
        """Write a new index folder at `path` and open it.

        Args:
            docs: one {"id", "title"?, "text"?, "metadata"?} mapping per document.
            tokens: (flat, offsets): (T, encoder.dim) float token vectors of every document,
                and (n_docs + 1,) int offsets. Stored in the dtype given (float16 saves half).
            dense: (n_docs, d) float, one vector per document; builds the `dense` stage.
            lexical: (n_docs, vocab) scipy sparse weights; builds the `lexical` stage. With
                `dense` too, the `hybrid` stage (dense + 0.3 * lexical) is built as well.
            pipeline: the default pipeline, recorded in the manifest.

        Raises:
            IndexExistsError: `path` exists and isn't an empty folder.
            ValueError: the inputs disagree with each other or the pipeline can't run on them
                (nothing is written).
        """
        path = Path(path)
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise IndexExistsError(f"{path} already exists and isn't empty; pick a new folder")
        flat, offsets = np.asarray(tokens[0]), np.asarray(tokens[1])
        if flat.ndim != 2 or flat.shape[1] != encoder.dim:
            raise ValueError(f"token vectors must be (T, {encoder.dim}), got {flat.shape}")
        if not np.issubdtype(flat.dtype, np.floating) or not np.issubdtype(
            offsets.dtype, np.integer
        ):
            raise ValueError(
                f"tokens need float vectors and int offsets, got {flat.dtype}, {offsets.dtype}"
            )
        MaxSimReranker(flat, offsets)  # checks the ragged layout
        n = len(offsets) - 1
        docs = list(docs)
        if len(docs) != n:
            raise ValueError(f"got {len(docs)} documents but token offsets for {n}")
        arrays = {f"{TOKENS}/flat.npy": flat, f"{TOKENS}/offsets.npy": offsets.astype(np.int64)}
        stages: dict[str, dict] = {}
        if dense is not None:
            dense = np.asarray(dense)
            if dense.ndim != 2 or len(dense) != n:
                raise ValueError(f"dense must hold {n} rows (one per document), got {dense.shape}")
            arrays |= DenseStage.arrays(dense)
            if not np.isfinite(arrays[VECTORS]).all():
                raise ValueError("dense vectors must be finite (no NaN or inf)")
            stages["dense"] = {"type": "dense", "dim": int(dense.shape[1])}
        if lexical is not None:
            if not sp.issparse(lexical) or lexical.ndim != 2 or lexical.shape[0] != n:
                raise ValueError(
                    f"lexical must be an ({n}, vocab) scipy sparse matrix (one row per "
                    f"document), got {type(lexical).__name__} {getattr(lexical, 'shape', '')}"
                )
            if lexical.dtype.kind not in "iuf":  # before the float32 cast drops complex parts
                raise ValueError(f"lexical weights must be real numbers, got {lexical.dtype}")
            arrays |= LexicalStage.arrays(lexical)  # any scipy format, incl. lil and dok
            if not np.isfinite(arrays[WEIGHTS]).all():
                raise ValueError("lexical weights must be finite (no NaN or inf)")
            stages["lexical"] = {"type": "lexical", "vocab": int(lexical.shape[1])}
        if dense is not None and lexical is not None:
            stages["hybrid"] = {"type": "hybrid", "alpha": HYBRID_ALPHA}
        p = parse_pipeline(pipeline)
        _check_pipeline(p, stages)
        existed = path.exists()
        try:
            info = write_segment(path, 1, arrays, docs)
            write_manifest(path, Manifest(encoder, stages, str(p), (info,)))
            return cls.open(path)
        except BaseException:  # leave the folder as we found it: absent, or empty
            # ponytail: on Windows, memmaps still open in the traceback can block the delete
            # (only if open() rejects what create() wrote); close arrays first if that shows up
            shutil.rmtree(path, ignore_errors=True)
            if existed:
                path.mkdir(exist_ok=True)
            raise

    @classmethod
    def open(cls, path: Path) -> Index:
        """Open an index folder written by `Index.create`.

        Raises:
            IndexFormatError: the folder, its manifest or its default pipeline can't be used.
        """
        path = Path(path)
        manifest, segments = open_index(path)
        # ponytail: one segment only; map global rows across segments when add() lands (0.2)
        if len(segments) != 1:
            raise IndexFormatError(
                f"{path} has {len(segments)} segments, but this laterank reads single-segment "
                "indexes only. Rebuild it with Index.create."
            )
        unknown = sorted(
            n for n, c in manifest.stages.items() if c["type"] not in FIRST_STAGE_TYPES
        )
        if unknown:
            raise IndexFormatError(
                f"{path}: stages {unknown} have types this laterank doesn't know. Upgrade laterank "
                "(pip install -U laterank) or rebuild the index."
            )
        try:
            p = parse_pipeline(manifest.default_pipeline)
            _check_pipeline(p, manifest.stages)
        except ValueError as e:
            raise IndexFormatError(
                f"{path}: the manifest's default pipeline can't run ({e}). Restore the manifest "
                "or rebuild the index."
            ) from None
        return cls(path, manifest, segments[0], p)

    @property
    def default_pipeline(self) -> Pipeline:
        return self._default

    def search(
        self, q: Encoded, k: int = 10, *, pipeline: str | Pipeline | None = None
    ) -> list[Hit]:
        """The top k documents for one encoded query, best first.

        Raises:
            PipelineSyntaxError: `pipeline` doesn't parse.
            ValueError: the pipeline uses a stage this index lacks, k is out of range, or the
                query doesn't fit the index.
        """
        if pipeline is None:
            p = self._default
        else:
            p = pipeline if isinstance(pipeline, Pipeline) else parse_pipeline(pipeline)
            _check_pipeline(p, self._first)
        cands = p.run(self._stages, q, k)
        hits = []
        for rank, (row, score) in enumerate(zip(cands.rows, cands.scores, strict=True), 1):
            doc = self._docs[int(row)]
            hits.append(Hit(doc["id"], rank, float(score), doc["text"], doc["metadata"]))
        return hits
