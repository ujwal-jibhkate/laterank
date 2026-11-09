"""laterank: local late-interaction retrieval.

A cheap first stage finds candidates; an exact MaxSim rerank orders them.
"""

from laterank._version import __version__
from laterank.components.bgem3 import BGEM3Encoder
from laterank.components.colbert import ColBERTEncoder
from laterank.components.encoder import load_encoder
from laterank.orchestration.index import Index
from laterank.orchestration.tune import Budget

__all__ = ["BGEM3Encoder", "Budget", "ColBERTEncoder", "Index", "__version__", "load_encoder"]
