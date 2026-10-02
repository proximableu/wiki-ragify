"""Cache for accepted/rejected pages — moved verbatim from reference/ollama_gateway + wiki_crawler.

Kept keyed by *original Wikipedia title* (not filename), so the canonical list of accepted
titles is available without reverse-engineering the non-invertible filename encoding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Cache:
    """Accepted/rejected page cache, keyed by original Wikipedia title.

    ``accepted`` maps original title -> on-disk filename so the canonical list of accepted
    titles is available without reverse-engineering the (non-invertible) filename encoding.
    ``rejected`` stores titles we have already declined.
    """

    accepted: dict[str, str] = field(default_factory=dict)
    rejected: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: "os.PathLike[str] | str") -> "Cache":
        """Load from disk, or return an empty cache if missing/corrupt."""
        import os

        path = Path(path)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return cls(
                    accepted=data.get("accepted", {}),
                    rejected=data.get("rejected", []),
                )
            except Exception:
                # Corrupt cache -> start fresh (a partial write, not a security issue).
                return cls()
        return cls()

    def save(self, path: "os.PathLike[str] | str") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"accepted": self.accepted, "rejected": self.rejected}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def add_accepted(self, title: str, filename: str) -> None:
        self.accepted[title] = filename

    def add_rejected(self, title: str) -> None:
        if title not in self.rejected:
            self.rejected.append(title)

    @property
    def titles(self) -> list[str]:
        """Original accepted titles, in insertion order (verbatim list for next crawl)."""
        return list(self.accepted.keys())
