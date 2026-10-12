"""Pipeline strings: "dense > maxsim@100" = "take dense's top 100, then rerank them with MaxSim".

    pipeline := name ( ">" name "@" depth )*      spaces around tokens are ignored
    name     := [a-z][a-z0-9-]*                   depth := positive integer

The first name is the first stage; every later name is a reranker with the number of
candidates it re-scores. Depths may not grow along the chain (a reranker can't see documents
the step before it dropped). The same string is used in Python, the manifest and reports, and
`parse_pipeline(str(p)) == p` always holds.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise

from laterank.components.base import Candidates, Encoded, FirstStage, Reranker
from laterank.errors import PipelineSyntaxError

_NAME = re.compile(r"[a-z][a-z0-9-]*")
_RERANK = re.compile(rf"({_NAME.pattern})\s*@\s*([1-9][0-9]*)")
_HELP = 'write it like "dense" or "dense > maxsim@100"'


@dataclass(frozen=True)
class Pipeline:
    first: str  # "dense"
    rerankers: tuple[tuple[str, int], ...] = ()  # (("maxsim", 100),)

    def __post_init__(self) -> None:
        """Every Pipeline obeys the grammar, however it was made (so parse(str(p)) == p)."""
        if not isinstance(self.first, str) or not _NAME.fullmatch(self.first):
            raise PipelineSyntaxError(
                f"a pipeline must start with a first-stage name in lower case, like 'dense', "
                f"got {self.first!r}; {_HELP}"
            )
        rerankers = tuple(self.rerankers)
        for r in rerankers:
            if not (
                isinstance(r, tuple)
                and len(r) == 2
                and isinstance(r[0], str)
                and _NAME.fullmatch(r[0])
                and type(r[1]) is int
                and r[1] >= 1
            ):
                raise PipelineSyntaxError(
                    f"each reranker must be a (name, depth) pair like ('maxsim', 100) with a "
                    f"lower-case name and a depth >= 1, got {r!r}"
                )
        depths = [d for _, d in rerankers]
        if any(b > a for a, b in pairwise(depths)):
            raise PipelineSyntaxError(
                f"depths grow ({depths}): each reranker can only re-score what the step "
                "before it kept, so depths must stay the same or shrink"
            )
        object.__setattr__(self, "rerankers", rerankers)

    def __str__(self) -> str:
        return " > ".join([self.first, *(f"{name}@{depth}" for name, depth in self.rerankers)])

    def run(self, stages: Mapping[str, FirstStage | Reranker], q: Encoded, k: int) -> Candidates:
        """Run the pipeline for one query and return its top k.

        Raises:
            TypeError: k isn't an integer.
            ValueError: k < 1, k larger than the last depth, or a name missing from `stages`.
        """
        k = operator.index(k)
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if self.rerankers and k > self.rerankers[-1][1]:
            raise ValueError(
                f"k={k} is more than the {self.rerankers[-1][1]} documents '{self}' reranks; "
                "ask for fewer, or raise the last depth"
            )
        missing = [n for n in [self.first, *(n for n, _ in self.rerankers)] if n not in stages]
        if missing:
            raise ValueError(f"'{self}' needs {missing}, but only {sorted(stages)} are available")
        depth = self.rerankers[0][1] if self.rerankers else k
        cands = stages[self.first].search(q, depth)
        for name, depth in self.rerankers:
            top = Candidates(cands.rows[:depth], cands.scores[:depth])
            cands = stages[name].rerank(q, top)
        return Candidates(cands.rows[:k], cands.scores[:k])


def parse_pipeline(spec: str) -> Pipeline:
    """Parse a pipeline string (see the module docstring) without running it.

    Raises:
        PipelineSyntaxError: the string doesn't follow the grammar, or a depth grows.
    """
    if not isinstance(spec, str):
        raise PipelineSyntaxError(f"a pipeline must be a string, got {type(spec).__name__}")
    first, *rest = [part.strip() for part in spec.split(">")]
    rerankers = []
    for part in rest:
        m = _RERANK.fullmatch(part)
        if not m:
            raise PipelineSyntaxError(
                f"{part!r} in {spec!r} must be a reranker with a depth, like 'maxsim@100'; {_HELP}"
            )
        rerankers.append((m[1], int(m[2])))
    return Pipeline(first, tuple(rerankers))
