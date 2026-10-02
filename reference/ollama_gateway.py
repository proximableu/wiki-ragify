"""
Centralized Ollama Gateway Module
---------------------------------
Encapsulates client lifecycle, environment-driven configuration, and deterministic
fail-safe routing. Designed for sequential pipeline execution without async overhead.
"""

import os
import time
import logging
from dataclasses import dataclass
from typing import Literal, Optional

from dotenv import load_dotenv
from ollama import Client
from pydantic import BaseModel, Field

# Load environment variables at module initialization
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Resolves infrastructure parameters from .env to prevent hardcoded dependencies."""
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    gate_model: str = os.getenv("GATE_MODEL_NAME", "qwen3-30b-dataset")
    num_ctx: int = int(os.getenv("NUM_CTX", "8192"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.0"))
    # Hard per-call budget on the chat generation. Without this a wedged/hung
    # Ollama would block evaluate() forever and defeat both the retry loop and
    # the pipeline's wall-clock timeout. httpx read-timeout fires when no bytes
    # arrive within the window, so a stalled generation is surfaced as an error.
    client_timeout: float = float(os.getenv("GATE_TIMEOUT", "60.0"))


class GatingResponse(BaseModel):
    """Pydantic schema for strict JSON output enforcement at the token engine level."""
    decision: Literal["accepted", "rejected"] = Field(
        description="Strictly 'accepted' if criteria met, 'rejected' otherwise."
    )


class LLMEvaluator:
    """
    Manages the Ollama HTTP client, schema validation, and error handling.
    Guarantees deterministic boolean routing regardless of execution order.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        # httpx 0.28: Timeout(float) applies connect/read/write/pool equally.
        # A float is fine because Ollama never streams; each call is a single
        # full-response read. The read timeout is what bounds a hung generation.
        self.client = Client(host=config.ollama_url, timeout=config.client_timeout)
        self.model = config.gate_model
        self.options = {
            "temperature": config.temperature,
            "num_ctx": config.num_ctx,
            "think": False  # Explicitly disable chain-of-thought for latency reduction
        }

    def evaluate(self, text: str, gating_prompt: str, retries: int = 3) -> bool:
        """
        Gates content via Ollama structured outputs.
        Returns True on acceptance, False on rejection or any transient failure.

        The gateway is a network/HTTP service, so transient timeouts or a brief
        outage should not permanently sink a crawl. Retry a few times with
        exponential backoff before falling back to the safe-default (reject).
        """
        # Minimal prompt assembly; schema enforcement is handled natively by the SDK
        full_prompt = f"{gating_prompt}\n\n<ARTICLE>\n{text}\n</ARTICLE>"

        last_error: Optional[BaseException] = None
        for attempt in range(1, retries + 1):
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a binary classifier. Output ONLY the required JSON object."},
                        {"role": "user", "content": full_prompt}
                    ],
                    format=GatingResponse.model_json_schema(),
                    options=self.options
                )

                # Validate against Pydantic model and extract decision
                result = GatingResponse.model_validate_json(response.message.content)
                return result.decision == "accepted"

            except Exception as e:  # noqa: BLE001 - fail-safe is deliberately broad
                last_error = e
                # Exponential backoff: ~0.5s, ~1s, ~2s ... capped so a long crawl still terminates.
                delay = min(0.5 * (2 ** (attempt - 1)), 5.0)
                logger.warning(f"LLM evaluation attempt {attempt}/{retries} failed ({type(e).__name__}), "
                               f"retrying in {delay:.1f}s: {e}")
                time.sleep(delay)

        logger.warning(f"LLM evaluation failed after {retries} attempts, defaulting to reject: {last_error}")
        return False

    def close(self) -> None:
        """Gracefully release underlying httpx connection pool resources."""
        self.client._client.close()