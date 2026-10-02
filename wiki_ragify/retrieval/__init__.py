"""Retrieval — ANN search, dedup, and relevance filtering over the vector store.

Public surface:

* :func:`retrieval.search.search_top_k` — ANN search + retrieval dedup (top-k).
* :func:`retrieval.relevance.filter_relevant_passages` — LLM relevance filtering.
* :func:`retrieval.store.load_db` — open the sqlite-vec store.
"""

from .search import search_top_k
from .relevance import filter_relevant_passages
from .store import load_db

__all__ = ["search_top_k", "filter_relevant_passages", "load_db"]
