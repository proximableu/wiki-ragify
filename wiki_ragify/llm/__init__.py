"""Ollama surface: Pydantic output schemas, the LLM gate, and embeddings.

Every LLM use in the pipeline goes through here, so schema enforcement and fail-safe
behaviour live in one place.
"""

from .models import GatingResponse, RelevanceFilter
from .gateway import LLMEvaluator
from .embed import EmbedClient

__all__ = ["GatingResponse", "RelevanceFilter", "LLMEvaluator", "EmbedClient"]
