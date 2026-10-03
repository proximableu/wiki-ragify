"""Pure aggregation helpers for the rolling statistics panel.

These functions take the stream of :class:`~wiki_ragify.pipeline.events.ProgressEvent`s and
:class:`~wiki_ragify.pipeline.events.StatEvent`s the pipeline emits and fold them into the
totals the statistics panel displays. They contain **no UI**, no Textual import, and no
wall-clock dependence beyond an explicit ``now`` argument, which is what lets the whole
suite exercise them deterministically and offline.

The panel is only ever a thin adapter: it feeds each event in (``update_stats``) and reads
the current snapshot (``snapshot``) for display. Keeping the fold here means the behaviour is
provable without launching an app.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# Metric keys the statistics panel renders, in display order.
METRIC_PAGES_CRAWLED = "pages_crawled"
METRIC_ACCEPTED = "accepted"
METRIC_REJECTED = "rejected"
METRIC_CHUNKS = "chunks"
METRIC_GATE2_ACCEPT = "gate2_accept"
METRIC_GATE2_REJECT = "gate2_reject"
METRIC_EMBEDDINGS = "embeddings"
METRIC_DEDUP_REMOVED = "dedup_removed"
METRIC_DB_ROWS = "db_rows"
METRIC_ELAPSED = "elapsed_s"
METRIC_CRAWL_RATE = "crawl_rate"
METRIC_GATE_RATE = "gate_rate"


# Maps a stage onto the stat metric it drives. Gate 1 is the crawl stage (whole pages);
# Gate 2 is the chunk stage. ``split`` and ``archive`` feed the chunk/db counts.
_STAGE_TO_STAT = {
    "crawl": METRIC_PAGES_CRAWLED,
    "split": METRIC_CHUNKS,
    "gate": None,  # handled directly: chunk accept/reject counts below
    "archive": None,
    "ingest": None,
}


@dataclass
class StatsSnapshot:
    """A point-in-time view of the rolling statistics, in display order."""

    pages_crawled: int = 0
    accepted: int = 0
    rejected: int = 0
    chunks: int = 0
    gate2_accept: int = 0
    gate2_reject: int = 0
    embeddings: int = 0
    dedup_removed: int = 0
    db_rows: int = 0
    elapsed_s: float = 0.0
    crawl_rate: float = 0.0
    gate_rate: float = 0.0

    def ordered_rows(self) -> list[tuple[str, str, float]]:
        """Return ``(label, metric, value)`` rows in stable display order."""
        rows = [
            (PAGES_CRAWLED_LABEL, METRIC_PAGES_CRAWLED, self.pages_crawled),
            (ACCEPTED_LABEL, METRIC_ACCEPTED, self.accepted),
            (REJECTED_LABEL, METRIC_REJECTED, self.rejected),
            (CHUNKS_LABEL, METRIC_CHUNKS, self.chunks),
            (GATE2_ACCEPT_LABEL, METRIC_GATE2_ACCEPT, self.gate2_accept),
            (GATE2_REJECT_LABEL, METRIC_GATE2_REJECT, self.gate2_reject),
            (EMBEDDINGS_LABEL, METRIC_EMBEDDINGS, self.embeddings),
            (DEDUP_REMOVED_LABEL, METRIC_DEDUP_REMOVED, self.dedup_removed),
            (DB_ROWS_LABEL, METRIC_DB_ROWS, self.db_rows),
            (ELAPSED_LABEL, METRIC_ELAPSED, self.elapsed_s),
            (CRAWL_RATE_LABEL, METRIC_CRAWL_RATE, self.crawl_rate),
            (GATE_RATE_LABEL, METRIC_GATE_RATE, self.gate_rate),
        ]
        return [(label, metric, f"{value:g}") for label, metric, value in rows]


# Human-readable labels, kept alongside their metric key so the panel never hardcodes a map.
METRIC_LABELS: dict[str, str] = {
    METRIC_PAGES_CRAWLED: "pages crawled",
    METRIC_ACCEPTED: "accepted",
    METRIC_REJECTED: "rejected",
    METRIC_CHUNKS: "chunks",
    METRIC_GATE2_ACCEPT: "gate2_accept",
    METRIC_GATE2_REJECT: "gate2_reject",
    METRIC_EMBEDDINGS: "embeddings",
    METRIC_DEDUP_REMOVED: "dedup_removed",
    METRIC_DB_ROWS: "db rows",
    METRIC_ELAPSED: "elapsed",
    METRIC_CRAWL_RATE: "crawl_rate",
    METRIC_GATE_RATE: "gate_rate",
}

# Label strings (label + the value unit the panel appends).
PAGES_CRAWLED_LABEL = "pages crawled"
ACCEPTED_LABEL = "accepted"
REJECTED_LABEL = "rejected"
CHUNKS_LABEL = "chunks"
GATE2_ACCEPT_LABEL = "gate2_accept"
GATE2_REJECT_LABEL = "gate2_reject"
EMBEDDINGS_LABEL = "embeddings"
DEDUP_REMOVED_LABEL = "dedup_removed"
DB_ROWS_LABEL = "db rows"
ELAPSED_LABEL = "elapsed"
CRAWL_RATE_LABEL = "crawl_rate"
GATE_RATE_LABEL = "gate_rate"


@dataclass
class _Accumulator:
    pages_crawled: int = 0
    accepted: int = 0
    rejected: int = 0
    chunks: int = 0
    gate2_accept: int = 0
    gate2_reject: int = 0
    embeddings: int = 0
    dedup_removed: int = 0
    db_rows: int = 0
    start_s: Optional[float] = None
    last_tick_s: Optional[float] = None

    def touch(self, now_s: float) -> None:
        """Record the run start on first contact; keep ``last_tick_s`` current."""
        if self.start_s is None:
            self.start_s = now_s
        self.last_tick_s = now_s


class Statistics:
    """Fold a stream of events into a :class:`StatsSnapshot`.

    Feed events with :meth:`update`; read a display-ready view with :meth:`snapshot`.
    """

    def __init__(self) -> None:
        self._acc = _Accumulator()

    def _record_elapsed(self) -> None:
        if self._acc.start_s is not None and self._acc.last_tick_s is not None:
            elapsed = self._acc.last_tick_s - self._acc.start_s
        else:
            elapsed = 0.0
        # Rates are per-minute (items/min) — a more stable number for a slow CPU gate.
        if self._acc.last_tick_s is not None and elapsed > 0:
            minutes = elapsed / 60.0
            crawl_rate = self._acc.pages_crawled / minutes
            gate_total = self._acc.gate2_accept + self._acc.gate2_reject
            gate_rate = gate_total / minutes
        else:
            crawl_rate = 0.0
            gate_rate = 0.0

    def update_stats(self, event, now_s: float) -> None:
        """Fold a single :class:`ProgressEvent` into the accumulator."""
        self._acc.touch(now_s)
        kind = getattr(event, "kind", "")
        if event.stage == "crawl":
            # Only accept/reject mark a page crawled; ticks and skips are bookkeeping.
            if kind in ("accept", "reject"):
                self._acc.pages_crawled += 1
                if kind == "accept":
                    self._acc.accepted += 1
                else:
                    self._acc.rejected += 1
        elif event.stage == "gate":
            if kind == "accept":
                self._acc.gate2_accept += 1
            elif kind == "reject":
                self._acc.gate2_reject += 1
        elif event.stage == "split":
            self._acc.chunks += 1
        elif event.stage == "ingest":
            # build_db emits only info/tick/stage_done; the usable counters are the
            # dedup message ("Dedup: {unique}/{total} unique") and the final tally
            # ("Ingested {n} documents"). Parse those rather than counting ticks.
            self._ingest_message(getattr(event, "message", ""))
        self._record_elapsed()

    def _ingest_message(self, message: str) -> None:
        """Derive dedup count and final row count from build_db's textual messages."""
        if not message:
            return
        if "Dedup:" in message:
            match = re.search(r"Dedup:\s*(\d+)\s*/\s*(\d+)\s*unique", message)
            if match:
                unique, total = int(match.group(1)), int(match.group(2))
                self._acc.dedup_removed = max(total - unique, 0)
        elif "Ingested" in message:
            match = re.search(r"Ingested\s*(\d+)\s*documents", message)
            if match:
                self._acc.db_rows = int(match.group(1))

    def update_stat(self, event, now_s: float) -> None:
        """Fold a :class:`StatEvent` (cumulative counter) into the accumulator."""
        self._acc.touch(now_s)
        name = event.name
        value = event.value
        if name == METRIC_PAGES_CRAWLED:
            self._acc.pages_crawled = value
        elif name == METRIC_ACCEPTED:
            self._acc.accepted = value
        elif name == METRIC_REJECTED:
            self._acc.rejected = value
        elif name == METRIC_CHUNKS:
            self._acc.chunks = value
        elif name == METRIC_GATE2_ACCEPT:
            self._acc.gate2_accept = value
        elif name == METRIC_GATE2_REJECT:
            self._acc.gate2_reject = value
        elif name == METRIC_EMBEDDINGS:
            self._acc.embeddings = value
        elif name == METRIC_DEDUP_REMOVED:
            self._acc.dedup_removed = value
        elif name == METRIC_DB_ROWS:
            self._acc.db_rows = value
        self._record_elapsed()

    def snapshot(self) -> StatsSnapshot:
        """Return the current display-ready snapshot."""
        elapsed = (
            self._acc.last_tick_s - self._acc.start_s
            if (self._acc.start_s is not None and self._acc.last_tick_s is not None)
            else 0.0
        )
        if elapsed > 0:
            minutes = elapsed / 60.0
            crawl_rate = self._acc.pages_crawled / minutes
            gate_total = self._acc.gate2_accept + self._acc.gate2_reject
            gate_rate = gate_total / minutes
        else:
            crawl_rate = 0.0
            gate_rate = 0.0
        return StatsSnapshot(
            pages_crawled=self._acc.pages_crawled,
            accepted=self._acc.accepted,
            rejected=self._acc.rejected,
            chunks=self._acc.chunks,
            gate2_accept=self._acc.gate2_accept,
            gate2_reject=self._acc.gate2_reject,
            embeddings=self._acc.embeddings,
            dedup_removed=self._acc.dedup_removed,
            db_rows=self._acc.db_rows,
            elapsed_s=elapsed,
            crawl_rate=crawl_rate,
            gate_rate=gate_rate,
        )


def format_duration(seconds: float) -> str:
    """Render elapsed seconds as ``Hh Mm Ss`` (dropping leading-zero groups).

    Pure time formatting — no dependency on the event clock.
    """
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"
