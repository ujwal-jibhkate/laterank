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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from laterank.components.base import Encoded
from laterank.components.bm25 import INDPTR, K1, TF, TOKENIZER, B, BM25Stage
from laterank.components.dense import VECTORS, DenseStage
from laterank.components.encoder import ENCODERS, Encoder, load_encoder
from laterank.components.hybrid import HYBRID_ALPHA, HybridStage
from laterank.components.lexical import WEIGHTS, LexicalStage
from laterank.components.maxsim import TOKENS, MaxSimReranker
from laterank.components.muvera import MUVERA_DEFAULT, MuveraStage
from laterank.components.smve import SMVE_BALANCED, SMVEStage
from laterank.errors import IndexExistsError, IndexFormatError
from laterank.kernels.maxsim import subset_ragged
from laterank.orchestration.evaluate import Evaluation, evaluate
from laterank.orchestration.pipeline import Pipeline, parse_pipeline
from laterank.orchestration.tune import (
    EXHAUSTIVE,
    KEY_K,
    Budget,
    TuneReport,
    candidates,
    span_queries,
    tune,
    tune_with_labels,
)
from laterank.storage.docs import _line, _record
from laterank.storage.manifest import EncoderInfo, Manifest, write_manifest
from laterank.storage.segment import Segment, open_index, write_segment

# manifest "type" -> class
STORED_STAGES = {
    "dense": DenseStage,
    "lexical": LexicalStage,
    "bm25": BM25Stage,
    "smve": SMVEStage,
    "muvera": MuveraStage,
}
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
    if p.first != EXHAUSTIVE and p.first not in first_stages:
        raise ValueError(
            f"'{p}' starts with '{p.first}', but this index has the first stages "
            f"{sorted([EXHAUSTIVE, *first_stages])}: start the pipeline with one of them, or "
            f"build the index with that stage (e.g. Index.create(..., dense=..., lexical=...))"
        )
    unknown = [name for name, _ in p.rerankers if name not in RERANKERS]
    if unknown:
        raise ValueError(f"'{p}' uses unknown rerankers {unknown}; known: {sorted(RERANKERS)}")


def _check_free(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise IndexExistsError(f"{path} already exists and isn't empty; pick a new folder")


def _encoder_text(doc: Mapping) -> str:
    """smve-lab's text joining: "title. text", or just the text when there is no title."""
    title, text = doc.get("title") or "", doc.get("text") or ""
    if not isinstance(title, str) or not isinstance(text, str):
        raise ValueError(f"document {doc.get('id')!r}: title and text must be strings")
    title, text = title.strip(), text.strip()
    return f"{title}. {text}" if title else text


def _stage_arrays(
    docs: list,
    flat: np.ndarray,
    offsets: np.ndarray,
    encoder: EncoderInfo,
    dense: np.ndarray | None,
    lexical: sp.spmatrix | sp.sparray | None,
    *,
    bm25: bool,
    smve: bool,
    muvera: bool,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Check the inputs and compute every array the segment stores, plus the manifest stages.

    Shared by `Index.create` and the add/delete rewrite. Raises ValueError as `create` documents.
    """
    flat, offsets = np.asarray(flat), np.asarray(offsets)
    if flat.ndim != 2 or flat.shape[1] != encoder.dim:
        raise ValueError(f"token vectors must be (T, {encoder.dim}), got {flat.shape}")
    if not np.issubdtype(flat.dtype, np.floating) or not np.issubdtype(offsets.dtype, np.integer):
        raise ValueError(
            f"tokens need float vectors and int offsets, got {flat.dtype}, {offsets.dtype}"
        )
    MaxSimReranker(flat, offsets)  # checks the ragged layout
    n = len(offsets) - 1
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
    if bm25:
        if not all(isinstance(d, Mapping) for d in docs):
            raise ValueError("every document must be a mapping like {'id': ..., 'text': ...}")
        texts = [f"{d.get('title') or ''}\n{d.get('text') or ''}" for d in docs]
        counts = BM25Stage.arrays(texts)
        if len(counts[TF]) == 0:
            raise ValueError("bm25: no document has a word to index (all empty or stop words)")
        arrays |= counts
        vocab = len(counts[INDPTR]) - 1
        stages["bm25"] = {
            "type": "bm25",
            "tokenizer": TOKENIZER,
            "k1": K1,
            "b": B,
            "vocab": vocab,
        }
    if smve or muvera:
        # min/max carry NaN and show +-inf without a (T, d) temporary; checked before an
        # encode that can take minutes
        if not (np.isfinite(flat.min()) and np.isfinite(flat.max())):
            who = " and ".join(n for n, on in (("smve", smve), ("muvera", muvera)) if on)
            raise ValueError(f"{who}: token vectors must be finite (no NaN or inf)")
    if smve:
        smve_arrays, sha = SMVEStage.arrays(flat, offsets, SMVE_BALANCED)
        arrays |= smve_arrays
        stages["smve"] = {"type": "smve", **SMVE_BALANCED, "anchor_sha256": sha}
    if muvera:
        arrays |= MuveraStage.arrays(flat, offsets, MUVERA_DEFAULT)
        stages["muvera"] = {"type": "muvera", **MUVERA_DEFAULT}
    return arrays, stages


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
            self._stages[EXHAUSTIVE] = tokens
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
        bm25: bool = False,
        smve: bool = False,
        muvera: bool = False,
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
            bm25: build the `bm25` stage from each document's title and text.
            smve: build the `smve` stage from the token vectors (setting SMVE_BALANCED; about
                5 minutes for SciFact's 1.86M tokens).
            muvera: build the `muvera` stage from the token vectors (setting MUVERA_DEFAULT;
                20,480 floats per document, 425 MB for SciFact).
            pipeline: the default pipeline, recorded in the manifest.

        Raises:
            IndexExistsError: `path` exists and isn't an empty folder.
            ValueError: the inputs disagree with each other or the pipeline can't run on them
                (nothing is written).
        """
        path = Path(path)
        _check_free(path)
        docs = list(docs)
        arrays, stages = _stage_arrays(
            docs, tokens[0], tokens[1], encoder, dense, lexical, bm25=bm25, smve=smve, muvera=muvera
        )
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
    def build(
        cls,
        path: Path,
        docs: Sequence[Mapping],
        encoder: Encoder,
        *,
        bm25: bool = False,
        smve: bool = False,
        muvera: bool = False,
        pipeline: str = "hybrid > maxsim@100",
    ) -> Index:
        """Encode each document's text with `encoder`, then `create` the index.

        Builds the stages the encoder has heads for (`dense`, `lexical`, and `hybrid` from both)
        plus any of `bm25`, `smve`, `muvera` you ask for. The text sent to the encoder is
        "title. text" (just the text when there is no title).

        Raises:
            IndexExistsError: `path` exists and isn't an empty folder (checked before encoding).
            ValueError: as in `create`, e.g. the pipeline needs a head this encoder lacks
                (found after encoding; nothing is written).
        """
        path = Path(path)
        _check_free(path)
        docs = list(docs)
        if not all(isinstance(d, Mapping) for d in docs):
            raise ValueError("every document must be a mapping like {'id': ..., 'text': ...}")
        enc = encoder.encode_docs([_encoder_text(d) for d in docs])
        return cls.create(
            path,
            docs,
            (enc.tokens, enc.offsets),
            encoder=encoder.info,
            dense=enc.dense,
            lexical=enc.lexical,
            bm25=bm25,
            smve=smve,
            muvera=muvera,
            pipeline=pipeline,
        )

    @classmethod
    def open(cls, path: Path) -> Index:
        """Open an index folder written by `Index.create`.

        Raises:
            IndexFormatError: the folder, its manifest or its default pipeline can't be used.
        """
        path = Path(path)
        manifest, segments = open_index(path)
        # ponytail: one segment only (add/delete rewrite into one, M5-2); map global rows across
        # segments if appended segments (research option B) ever land
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

    def evaluate(
        self,
        queries: Mapping[str, Encoded],
        qrels: Mapping[str, Mapping[str, float]],
        pipelines: Sequence[str | Pipeline],
        *,
        repeats: int = 3,
        ignore_identical_ids: bool = False,
    ) -> Evaluation:
        """Run each pipeline on every judged query and score it.

        Only queries in `qrels` run; extra queries are ignored. Each pipeline returns
        k = min(100, its last depth) documents, so Recall@100 of "x > maxsim@10" counts 10.
        Reports nDCG@10, Recall@100 and p50 ms (see `orchestration/evaluate.py`).

        ignore_identical_ids: drop a hit whose doc id equals its query id (BEIR's rule;
        ArguAna's queries are corpus documents). Each search then asks for min(k + 1, last
        depth) and keeps the first k that aren't the query itself.

        Raises:
            PipelineSyntaxError: a pipeline string doesn't parse.
            ValueError: qrels or pipelines is empty, a judged query is missing from `queries`,
                a value isn't an Encoded or has the wrong token dim, repeats < 1,
                ignore_identical_ids isn't a bool, or a pipeline uses a stage this index
                lacks. All checked before any search runs.
        """
        return evaluate(
            lambda p, q, k: p.run(self._stages, q, k),
            lambda row: self._docs[row]["id"],
            queries,
            qrels,
            [self._pipeline(p) for p in pipelines],
            repeats,
            self.manifest.encoder.dim,
            ignore_identical_ids,
        )

    def tune(
        self,
        budget: Budget,
        *,
        queries: Mapping[str, Encoded] | None = None,
        qrels: Mapping[str, Mapping[str, float]] | None = None,
        ignore_identical_ids: bool = False,
        encoder: Encoder | None = None,
        n_queries: int | None = None,
        seed: int | None = None,
    ) -> TuneReport:
        """Pick the pipeline for this index within the budget. Both modes try the same candidates:
        `exhaustive`, every built first stage alone, and every built first stage `> maxsim@d` for
        d in DEPTHS below n_docs.

        With labels (recommended): give `queries` and `qrels` like `Index.evaluate`; the pick has
        the highest nDCG@10 within the budget (`ignore_identical_ids` as in `evaluate`).

        Without labels: cuts `n_queries` (default 500) 8-24-word spans out of documents and picks
        the pipeline whose top 10 best agrees with exhaustive MaxSim's (the span's own document
        dropped everywhere; `seed` default 0; `encoder` defaults to
        `load_encoder(self.manifest.encoder)`). Agreeing with MaxSim isn't finding relevant
        documents: this can't pick a pipeline that beats MaxSim, and it can miss `bm25` (up to
        0.052 nDCG@10 lost in M2-3, docs/dev/tune_validation.md). See `orchestration/tune.py`.

        Raises:
            ValueError: budget isn't a Budget; only one of `queries`/`qrels` is given; `encoder`,
                `n_queries` or `seed` is given with `qrels`; `ignore_identical_ids` is given
                without `qrels`; anything `evaluate` refuses (with labels); n_queries not an
                int >= 1, seed out of range, fewer than 11 documents, no document has 8 words,
                `encoder.info != self.manifest.encoder`, or the index has no loadable encoder and
                none was given (without labels). All checked before any search runs.
        """
        if not isinstance(budget, Budget):
            raise ValueError(f"budget must be a Budget, e.g. Budget(latency_ms=20), got {budget!r}")
        if (queries is None) != (qrels is None):
            raise ValueError(
                "queries and qrels go together: give both (tune with labels) or neither"
            )
        n = len(self._docs)
        if qrels is not None:
            extra = [
                name
                for name, v in (("encoder", encoder), ("n_queries", n_queries), ("seed", seed))
                if v is not None
            ]
            if extra:
                raise ValueError(
                    f"{', '.join(extra)} only apply without labels (span queries); "
                    "drop them, or drop qrels"
                )
            return tune_with_labels(
                lambda p, q, k: p.run(self._stages, q, k),
                lambda row: self._docs[row]["id"],
                queries,
                qrels,
                candidates(self._first, n),
                budget,
                self.manifest.encoder.dim,
                ignore_identical_ids,
            )
        if ignore_identical_ids is not False:
            raise ValueError("ignore_identical_ids only applies with qrels (tune with labels)")
        n_queries = 500 if n_queries is None else n_queries
        seed = 0 if seed is None else seed
        if type(n_queries) is not int or n_queries < 1:  # not bool
            raise ValueError(f"n_queries must be an integer >= 1, got {n_queries!r}")
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError(f"seed must be an integer in [0, 2**64), got {seed!r}")
        if n <= KEY_K:
            raise ValueError(f"tune needs at least {KEY_K + 1} documents, this index has {n}")
        spans = span_queries([_encoder_text(self._docs[i]) for i in range(n)], n_queries, seed)
        if not spans:
            raise ValueError("tune cuts 8-24-word spans out of documents, but none has 8 words")
        if encoder is None:
            if self.manifest.encoder.name not in ENCODERS:
                raise ValueError(
                    f"this index was built from precomputed vectors ({self.manifest.encoder.name!r}"
                    "), so tune can't load its model: pass encoder= the one that made them"
                )
            encoder = load_encoder(self.manifest.encoder)
        elif encoder.info != self.manifest.encoder:
            raise ValueError(
                f"encoder {encoder.info} isn't the one this index was built with "
                f"({self.manifest.encoder}): its queries wouldn't match the documents"
            )
        pipelines = candidates(self._first, n)
        encoded = [(row, encoder.encode_query(span)) for row, span in spans]
        return tune(lambda p, q, k: p.run(self._stages, q, k), encoded, pipelines, budget)

    def set_default(self, pipeline: str | Pipeline) -> None:
        """Make `pipeline` the default: checked like `search` checks it, then written to the
        manifest atomically (`write_manifest`) and used by this Index at once.

        Raises:
            PipelineSyntaxError: the string doesn't parse.
            ValueError: the pipeline uses a stage this index lacks.
        """
        p = self._pipeline(pipeline)
        manifest = replace(self.manifest, default_pipeline=str(p))
        write_manifest(self.path, manifest)
        self.manifest, self._default = manifest, p

    def add(self, docs: Sequence[Mapping], encoder: Encoder | None = None) -> None:
        """Encode `docs` with `encoder` and add them after the existing documents.

        `encoder=None` loads the index's own model, load_encoder(self.manifest.encoder), only
        after the document checks below have passed, so a bad document fails without the wait.

        Rewrites the whole index: every stage is recomputed over old + new documents, so the
        result equals Index.build on (old docs in their order) + (new docs in the given order).
        Measured on SciFact with BGE-M3 (5,183 docs, all stages): 354 s, of which SMVE is 318 s
        and MUVERA 26 s; about 36 s without SMVE, and about 10 s without SMVE and MUVERA. Old
        documents are not re-encoded.

        This object reads the new index at once. Any other open Index on this folder (in this
        or another process) must be reopened: the old segment's files are deleted.

        Raises (all before encoding; the index is unchanged):
            ValueError: `docs` is empty or not all mappings; an id is already in the index or
                repeats within `docs`; `encoder.info != self.manifest.encoder` (another model,
                revision or settings would put the new vectors in a different space).
            ValueError, ImportError, OSError: `encoder` is None and load_encoder fails (no such
                encoder, the loaded model differs from the recorded one, or the model library or
                download is missing).
            ValueError: as in `create`, e.g. bm25 finds no word (raised before anything is written).
            ValueError (after encoding): the encoder's dense/lexical heads differ from the index's
                stages.
        """
        docs = list(docs)
        if not docs or not all(isinstance(d, Mapping) for d in docs):
            raise ValueError("add needs one or more documents like {'id': ..., 'text': ...}")
        for d in docs:  # storage's own checks (keys, id, metadata), before the long rewrite
            _line(_record(d))
        ids = [d["id"] for d in docs]
        clash = sorted(set(ids) & set(self._docs.ids()))
        if clash:
            raise ValueError(f"ids already in the index: {clash[:5]}; delete them first")
        if len(set(ids)) != len(ids):
            raise ValueError("document ids repeat within the documents to add")
        if encoder is None:
            encoder = load_encoder(self.manifest.encoder)
        if encoder.info != self.manifest.encoder:
            raise ValueError(
                f"the encoder ({encoder.info}) isn't the one this index was built with "
                f"({self.manifest.encoder}); use load_encoder(index.manifest.encoder)"
            )
        enc = encoder.encode_docs([_encoder_text(d) for d in docs])
        tok = self._stages["maxsim"]
        flat = np.concatenate([tok.flat, np.asarray(enc.tokens).astype(tok.flat.dtype)])
        offsets = np.concatenate([tok.offsets, tok.offsets[-1] + np.asarray(enc.offsets)[1:]])
        del tok  # no memmap of the old segment may stay open, or Windows can't delete it
        dense, lexical = self._old_heads()
        if (dense is None) != (enc.dense is None) or (lexical is None) != (enc.lexical is None):
            raise ValueError("the encoder's heads (dense, lexical) differ from the index's stages")
        if dense is not None:
            dense = np.concatenate([dense, np.asarray(enc.dense)])
        if lexical is not None:
            lexical = sp.vstack([lexical, enc.lexical])
        old = [self._docs[r] for r in range(len(self._docs))]
        self._rewrite(old + docs, flat, offsets, dense, lexical)

    def delete(self, ids: Iterable[str]) -> None:
        """Remove the documents with these ids. The rest keep their order.

        Rewrites the whole index, as `add` does; the result equals Index.build on the remaining
        documents. As with `add`, any other open Index on this folder must be reopened.

        Raises (the index is unchanged):
            ValueError: `ids` is a bare str or bytes, is empty, names an id not in the index
                (lists up to 5), or would remove every document.
        """
        if isinstance(ids, (str, bytes)):
            raise ValueError(f"delete takes a list of ids, got the single value {ids!r}")
        gone = set(ids)
        if not gone:
            raise ValueError("delete needs one or more ids")
        old = self._docs.ids()
        missing = sorted(gone - set(old), key=str)
        if missing:
            raise ValueError(f"ids not in the index: {missing[:5]}")
        keep = [r for r, i in enumerate(old) if i not in gone]
        if not keep:
            raise ValueError("delete would remove every document; build a new index instead")
        tok = self._stages["maxsim"]
        flat, offsets = subset_ragged(tok.flat, tok.offsets, keep)
        del tok  # no memmap of the old segment may stay open, or Windows can't delete it
        dense, lexical = self._old_heads()
        dense = None if dense is None else dense[keep]  # copies; drops the memmap
        lexical = None if lexical is None else lexical[keep]
        self._rewrite([self._docs[r] for r in keep], flat, offsets, dense, lexical)

    def _old_heads(self) -> tuple[np.ndarray | None, sp.csr_matrix | None]:
        """The stored dense vectors and lexical weights (n_docs rows), or None without the stage."""
        dense = self._first.get("dense")
        lex = self._first.get("lexical")
        lexical = None
        if lex is not None:
            shape = (lex.n_docs, lex.vocab)
            lexical = sp.csc_matrix((lex.weights, lex.rows, lex.indptr), shape=shape).tocsr()
        return (None if dense is None else dense.vectors), lexical

    def _rewrite(self, docs, flat, offsets, dense, lexical) -> None:
        """Write the index anew as one segment, swap the manifest, then drop the old segments."""
        # ponytail: add/delete hand over the whole token matrix in RAM (3.6 GiB for SciFact);
        # stream it into a .npy memmap when a corpus doesn't fit
        types = {c["type"] for c in self.manifest.stages.values()}
        arrays, stages = _stage_arrays(
            docs,
            flat,
            offsets,
            self.manifest.encoder,
            dense,
            lexical,
            bm25="bm25" in types,
            smve="smve" in types,
            muvera="muvera" in types,
        )
        # numbers of every seg-NNNNNN* entry, listed or left over from a crash
        used = [int(p.name[4:10]) for p in self.path.glob("seg-*") if p.name[4:10].isdigit()]
        info = write_segment(self.path, 1 + max(used, default=0), arrays, docs)
        old = self.manifest
        # created_with defaults to this laterank: it wrote every file now in the index
        write_manifest(
            self.path,
            Manifest(old.encoder, stages, old.default_pipeline, (info,), created_at=old.created_at),
        )
        self.__dict__.update(Index.open(self.path).__dict__)  # drops our old memmaps
        # another open Index on this folder can still hold memmaps, and on Windows that blocks
        # the delete; ignore it, the next rewrite removes every unlisted segment
        for p in self.path.glob("seg-*"):
            if p.name != info.id:
                shutil.rmtree(p, ignore_errors=True)

    def _pipeline(self, pipeline: str | Pipeline) -> Pipeline:
        p = pipeline if isinstance(pipeline, Pipeline) else parse_pipeline(pipeline)
        _check_pipeline(p, self._first)
        return p

    def search(
        self, q: Encoded, k: int = 10, *, pipeline: str | Pipeline | None = None
    ) -> list[Hit]:
        """The top k documents for one encoded query, best first.

        Raises:
            PipelineSyntaxError: `pipeline` doesn't parse.
            ValueError: the pipeline uses a stage this index lacks, k is out of range, or the
                query doesn't fit the index.
        """
        p = self._default if pipeline is None else self._pipeline(pipeline)
        cands = p.run(self._stages, q, k)
        hits = []
        for rank, (row, score) in enumerate(zip(cands.rows, cands.scores, strict=True), 1):
            doc = self._docs[int(row)]
            hits.append(Hit(doc["id"], rank, float(score), doc["text"], doc["metadata"]))
        return hits
