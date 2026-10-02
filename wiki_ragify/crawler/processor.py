"""Page processing — title-to-filename, intro extraction, and bad-page heuristics.

Moved verbatim from the reference crawler. Kept as static helpers so it's trivial to unit
test without hitting Wikipedia.
"""

from __future__ import annotations

import re
from typing import Optional


class PageProcessor:
    """Heuristics for turning a raw Wikipedia extract into a gateable intro paragraph."""

    def __init__(self, config) -> None:
        self.config = config

    @staticmethod
    def title_to_filename(title: str, length_limit: int | None = None) -> str:
        limit = length_limit or 200
        safe = title.replace("/", "_").replace(" ", "_").replace("&", "_")
        return safe[:limit] + ".txt"

    def extract_first_paragraph(self, text: str) -> Optional[str]:
        """Take the lead section; if too short, fall back to the first N words of the text.

        The lead is the text before the first ``==`` heading (Wikipedia intro). Fewer than
        ``min_words`` words there means the article has no separate lead, so we use the whole
        text as the gating window.
        """
        if not text:
            return None
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lead_section = text.split("==")[0].strip()
        words = lead_section.split()

        if len(words) < self.config.min_words:
            words = text.split()

        return " ".join(words[: self.config.max_tokens]) if words else None

    def is_bad_page(self, title: str, text: str) -> bool:
        """Heuristic pre-filter before the (costly) LLM gate.

        Rejects disambiguation pages, "List of" pages, and anything shorter than
        ``min_text_length`` characters.
        """
        if not text or "(disambiguation)" in title.lower() or title.startswith("List of"):
            return True
        return len(text) < self.config.min_text_length
