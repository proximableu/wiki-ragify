"""Relevance filter — trims candidate passages with a lightweight routing LLM.

Faithful port of ``filter_relevant_passages`` from ``reference/retriever.py``.
Given the query and a pool of candidate ``(similarity, filename, text)`` passages
(sorted by descending similarity), it asks the light model to return the indices
of the passages to keep, then filters. The indices are into the sorted list, so
both the selection and the fallback padding draw from it.

The reference sorts first so its deterministic fallback (``sorted_passages[:min_passages]``)
and its padding (highest-scoring remaining passages) both return the most relevant
passages. That ordering, and the seed=42 determinism, are preserved.
"""

from __future__ import annotations

import logging
from ollama import Client

from ..config import Config
from ..llm.models import RelevanceFilter

logger = logging.getLogger(__name__)

# Truncate snippets to 500 chars to keep the routing model's token load bounded.
_SNIPPET_LIMIT = 500


def filter_relevant_passages(
    query: str,
    passages: list[tuple[float, str, str]],
    config: Config,
    min_passages: int = 16,
    chat_client: Client | None = None,
) -> list[tuple[float, str, str]]:
    """Return the passages relevant to ``query``, using the light routing model.

    Passages are treated as ``(similarity, filename, text)`` tuples sorted by
    descending similarity. On any LLM failure the highest-scoring ``min_passages``
    are returned as a deterministic fallback, matching the reference.
    """
    if not passages:
        return []

    sorted_passages = sorted(passages, key=lambda p: p[0], reverse=True)

    # FIX: Truncate snippets to reduce token load on the light routing model.
    snippets = [
        f"[{i}] {text[:_SNIPPET_LIMIT].replace(chr(10), ' ')}" for i, (_, _, text) in enumerate(sorted_passages)
    ]
    prompt = (
        f"You are an inclusive RAG context retrieval assistant evaluating passages for the query: \"{query}\"\n\n"
        f"INCLUSION CRITERIA:\n"
        f"- Keep any passage that directly answers, provides useful background context, explores adjacent concepts, or offers alternative viewpoints/debunking details related to the query.\n"
        f"- Err on the side of caution: if a passage is even partially relevant or helps build a complete picture, include it.\n\n"
        f"EXCLUSION CRITERIA:\n"
        f"- Only exclude a passage if it is completely off-topic, entirely corrupted, or shares zero conceptual overlap with the query.\n\n"
        f"Passages to evaluate:\n{chr(10).join(snippets)}\n\n"
        f"Output JSON only."
    )

    if chat_client is None:
        chat_client = Client(host=config.ollama_url, timeout=config.gate_timeout)

    try:
        response = chat_client.chat(
            model=config.lightweight_model,
            messages=[
                {"role": "system", "content": "You are a precise routing filter. Reasoning effort: low."},
                {"role": "user", "content": prompt},
            ],
            format=RelevanceFilter.model_json_schema(),
            options={"temperature": 0.0, "seed": 42},
        )
        data = RelevanceFilter.model_validate_json(response["message"]["content"])
        valid_indices = set(data.relevant_indices)

        filtered = [p for i, p in enumerate(sorted_passages) if i in valid_indices]

        # FIX: Safety net to guarantee minimum context volume (min_passages).
        if len(filtered) < min_passages:
            logger.warning(
                f"Relevance filter returned {len(filtered)} passages. Padding to {min_passages}."
            )
            remaining = [p for i, p in enumerate(sorted_passages) if i not in valid_indices]
            filtered.extend(remaining[: min_passages - len(filtered)])

        logger.info(f"Relevance filter: {len(passages)} -> {len(filtered)} passages")
        return filtered
    except Exception as e:
        logger.warning(f"Relevance filtering failed ({e}), bypassing filter")
        return sorted_passages[:min_passages]
