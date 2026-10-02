"""LLM gate — the article/chunk classifier, moved verbatim from ``reference/ollama_gateway.py``.

Only change: ``LLMEvaluator`` now takes the package ``Config`` instead of the old
``PipelineConfig`` dataclass (the two are structurally identical; ``Config`` is the
single source of truth in v2).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from ollama import Client

from ..config import Config
from .models import GatingResponse

logger = logging.getLogger(__name__)


class LLMEvaluator:
    """Manages the Ollama HTTP client, schema validation, and error handling.

    Guarantees deterministic boolean routing regardless of execution order.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        # httpx Timeout(float) applies connect/read/write/pool equally; a float is fine
        # because Ollama never streams — each call is a single full-response read. The
        # read timeout is what bounds a hung generation.
        self.client = Client(host=config.ollama_url, timeout=config.gate_timeout)
        self.model = config.gate_model
        self.options = {
            "temperature": config.gate_temperature,
            "num_ctx": config.num_ctx,
            # Explicitly disable chain-of-thought for latency reduction.
            "think": False,
        }

    def evaluate(self, text: str, gating_prompt: str, retries: int = 3) -> bool:
        """Gate content via Ollama structured outputs.

        Returns True on acceptance, False on rejection or any transient failure.

        The gateway is a network/HTTP service, so transient timeouts or a brief outage
        should not permanently sink a crawl. Retry a few times with exponential backoff
        before falling back to the safe-default (reject).
        """
        # Minimal prompt assembly; schema enforcement is handled natively by the SDK.
        full_prompt = f"{gating_prompt}\n\n<ARTICLE>\n{text}\n</ARTICLE>"

        last_error: Optional[BaseException] = None
        for attempt in range(1, retries + 1):
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a binary classifier. Output ONLY the required JSON object.",
                        },
                        {"role": "user", "content": full_prompt},
                    ],
                    format=GatingResponse.model_json_schema(),
                    options=self.options,
                )

                # Validate against Pydantic model and extract decision (native enforcement).
                result = GatingResponse.model_validate_json(response.message.content)
                return result.decision == "accepted"

            except Exception as e:  # noqa: BLE001 - fail-safe is deliberately broad
                last_error = e
                # Exponential backoff: ~0.5s, ~1s, ~2s ... capped so a long crawl still terminates.
                delay = min(0.5 * (2 ** (attempt - 1)), 5.0)
                logger.warning(
                    f"LLM evaluation attempt {attempt}/{retries} failed ({type(e).__name__}), "
                    f"retrying in {delay:.1f}s: {e}"
                )
                time.sleep(delay)

        logger.warning(
            f"LLM evaluation failed after {retries} attempts, defaulting to reject: {last_error}"
        )
        return False

    def close(self) -> None:
        """Gracefully release underlying httpx connection pool resources.

        ``self.client._client`` is the underlying httpx.Client and ``.close()`` is httpx's
        public API (verified in TASKS.md retraction note).
        """
        inner = getattr(self.client, "_client", None)
        if inner is not None:
            inner.close()
