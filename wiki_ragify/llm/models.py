"""Pydantic schemas for the structured (JSON-enforced) LLM outputs.

These are the exact schemas the Ollama ``format=`` parameter enforces at the token
engine level, so the SDK never returns anything we have to parse by hand. The gate
model and the retrieval relevance model both use them.
"""

from pydantic import BaseModel, Field


class GatingResponse(BaseModel):
    """Article/chunk gate decision. Strictly ``accepted`` or ``rejected``."""

    decision: str = Field(
        description="Strictly 'accepted' if criteria met, 'rejected' otherwise.",
    )


class RelevanceFilter(BaseModel):
    """Which retrieved passages to keep. Indices into the caller's passage list."""

    relevant_indices: list[int] = Field(
        description="Indices of passages directly relevant to the query",
    )
