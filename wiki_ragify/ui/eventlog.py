"""A bounded, in-memory ring buffer of pipeline log lines.

Pure data structure with no Textual import, so its behaviour (wrapping, ordering, colour
classification) is unit-tested offline. The live log panel is only a thin adapter that:
subscribes to runner events, hands each to :class:`EventLog` for folding, and renders the
buffer's current rows into a Textual widget.

Each row carries a *style* the panel uses for colouring; the styles mirror the colours of
the old ``chunk_gate.py`` output (green accept, red reject, muted skip/info). The colour
choice lives here (with the classification) so the panel never hardcodes a colour map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Log-row styles. These are Rich markup tokens (the panel emits ``[style]text[/]``),
# not Textual CSS class names. ``dim`` renders as a bold-dim muted row.
STYLE_INFO = "dim"   # neutral / fallback


@dataclass(frozen=True)
class LogRow:
    """One rendered line in the log panel, with its classification for colouring."""

    text: str
    style: str
    index: Optional[int] = None


# The set of rows the panel shows newest-first. ``max`` bounds memory during long crawls.
@dataclass
class EventLog:
    """Fold runner events into a bounded newest-first log."""

    max: int = 500
    _rows: list[LogRow] = None

    def __post_init__(self) -> None:
        if self._rows is None:
            self._rows = []

    @staticmethod
    def classify(kind: str, stage: str = "") -> str:
        """Map an event kind to a colour style.

        The whole funnel shares one palette — accept is cyan, reject is magenta,
        skip/warn are yellow, bookkeeping (info/tick) is dim, stage_done is bold.
        Falls back to ``info`` for any unexpected kind so the log never looks broken.
        """
        return {
            "accept": "cyan",
            "reject": "magenta",
            "skip": "yellow",
            "warn": "yellow",
            "tick": "dim",
            "info": "dim",
            "stage_done": "bold",
        }.get(kind, STYLE_INFO)

    @staticmethod
    def _is_link_tick(message: str) -> bool:
        """True for the per-[LINK] crawl bookkeeping rows, which are now hidden."""
        return message.startswith("[LINK]")

    def append(self, event) -> None:
        """Fold a single event into the buffer, wrapping past ``max``."""
        kind = getattr(event, "kind", "info")
        message = getattr(event, "message", "") or ""
        stage = getattr(event, "stage", "")

        # The per-[LINK] crawl rows are now hidden from the log; drop them here.
        if self._is_link_tick(message):
            return

        index = getattr(event, "index", None)

        # Compose the display string the way the old chunk_gate log did: "[N/total]" prefix.
        total = getattr(event, "total", None)
        if index is not None and total is not None:
            text = f"[{index}/{total}] {message}"
        else:
            text = message

        self._rows.append(LogRow(text=text, style=self.classify(kind, stage), index=index))
        if len(self._rows) > self.max:
            # Drop the oldest; the log is newest-first so the head is what we keep.
            self._rows = self._rows[-self.max :]

    def rows(self) -> list[LogRow]:
        """Return the current log rows, newest-first."""
        return list(reversed(self._rows))

    def clear(self) -> None:
        self._rows = []
