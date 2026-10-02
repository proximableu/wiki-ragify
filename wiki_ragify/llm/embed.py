"""Embeddings client — batch embed + single-query embed, moved from build_db_vec/retriever.

Wraps the Ollama ``/api/embed`` endpoint. Same retry/backoff-and-fail pattern as the
gate: a wedged Ollama must not hang a multi-hour ingest, so we give up to a few retries
and only then let the caller handle the failure.
"""

from __future__ import annotations

import logging
import time
from typing import List

from ollama import Client

from ..config import Config

logger = logging.getLogger(__name__)


class EmbedClient:
    """Generates embeddings for batches of texts and for a single query."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.client = Client(host=config.ollama_url, timeout=config.gate_timeout)
        self.model = config.embed_model
        self.dim = config.embed_dim

    def embed_batch(self, texts: List[str], retries: int = 3) -> List[List[float]]:
        """Embed a batch of texts in one Ollama call.

        Returns a list of vectors (one per input text). Raises on unrecoverable failure.
        """
        last_error: BaseException | None = None
        for attempt in range(1, retries + 1):
            try:
                response = self.client.embed(model=self.model, input=texts)
                embeddings = response.get("embeddings", [])
                if not embeddings or not isinstance(embeddings[0], list):
                    raise ValueError("Invalid embedding structure from Ollama")
                return [[float(x) for x in emb] for emb in embeddings]
            except Exception as e:  # noqa: BLE001
                last_error = e
                delay = min(0.5 * (2 ** (attempt - 1)), 5.0)
                logger.warning(
                    f"Embedding attempt {attempt}/{retries} failed ({type(e).__name__}), "
                    f"retrying in {delay:.1f}s: {e}"
                )
                time.sleep(delay)
        raise RuntimeError(f"Embedding failed after {retries} attempts: {last_error}")

    def embed_query(self, text: str) -> List[float]:
        """Generate a single embedding vector for a query string."""
        vectors = self.embed_batch([text])
        if not vectors:
            raise ValueError("Embedding returned an empty result")
        return vectors[0]
