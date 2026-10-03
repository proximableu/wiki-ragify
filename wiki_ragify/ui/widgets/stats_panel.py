"""Statistics panel widget — rolling metrics, throttled repaint (FR-4).

The folding logic (folding the event stream into totals) lives in
:class:`~wiki_ragify.ui.aggregation.Statistics`, which is already unit-tested offline.
This widget is the thin Textual adapter: it feeds events in (``on_event`` / ``on_stat``)
and reads a snapshot (``snapshot()``) for display.

The *throttle* is the one piece of UI-timing logic that lives here: the runner emits many
events per second, but the panel only rebuilds its rows at most once per ``refresh_every``
seconds. Throttling is a tiny pure function (``throttle``) so it's covered by a test too.
"""

from __future__ import annotations

import time
from typing import Optional

from textual.widgets import RichLog

from ..aggregation import Statistics


def throttle(last_ts: float, now: float, interval: float) -> tuple[float, bool]:
    """Return ``(now, repaint)`: repaint only if ``interval`` seconds have elapsed.

    Pure and testable: ``last_ts`` is the previous repaint time, ``now`` the current one.
    """
    if now - last_ts >= interval:
        return now, True
    return last_ts, False


class StatsPanel(RichLog):
    """A RichLog of rolling metrics, throttled to ``refresh_every`` seconds.

    Rendered as ``label: value`` rows, right-aligned so the numbers stay in a column.
    Thread-safe: :meth:`feed` folds events from the pipeline worker thread (cheap only);
    :meth:`maybe_repaint` rewrites the widget's contents on the main thread.
    """

    def __init__(self, refresh_every: float = 0.25) -> None:
        super().__init__(markup=True, id="stats-panel")
        self.stats = Statistics()
        self.refresh_every = refresh_every
        self._last_repaint = 0.0
        self._rows_cache: list[tuple[str, str, str]] = []

    def feed(self, event) -> None:
        """Feed a ProgressEvent (or StatEvent) into the rolling statistics.

        The app calls this from the pipeline worker thread; only the cheap fold happens here,
        never the repaint.
        """
        if hasattr(event, "stage") and hasattr(event, "kind"):
            self.stats.update_stats(event, now_s=event.ts or time.monotonic())
        elif hasattr(event, "name"):
            self.stats.update_stat(event, now_s=event.ts or time.monotonic())

    def snapshot_rows(self) -> list[tuple[str, str, str]]:
        """Return the current ``(label, value, metric)`` display rows."""
        return self.stats.snapshot().ordered_rows()

    def _render_rows(self) -> None:  # pragma: no cover - Textual render
        self.clear()
        for label, _metric, value in self._rows_cache:
            self.write(f"[primary]{label}[/]: [accent2]{value}[/]")

    def maybe_repaint(self, now: Optional[float] = None) -> bool:
        """Repaint if the throttle has elapsed. Returns True when a repaint happened."""
        now = time.monotonic() if now is None else now
        new_ts, repaint = throttle(self._last_repaint, now, self.refresh_every)
        if repaint:
            self._last_repaint = new_ts
            self._rows_cache = self.snapshot_rows()
            self._render_rows()
        return repaint

    @property
    def rows(self) -> list[tuple[str, str, str]]:
        return self._rows_cache
