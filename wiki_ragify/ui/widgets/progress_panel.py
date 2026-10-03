"""Progress window widget — live decision log + per-stage status (FR-3).

Two layers of behaviour live here, cleanly separated so the folding logic can be tested
without launching an app:

* :class:`LogLine` + :func:`render_line` — pure: turn a :class:`ProgressEvent` into the
  ``(text, style)`` tuple the log shows. ``render_line`` composes the ``[index/total]``
  prefix and maps the event kind to a Textual color.
* :class:`ProgressPanel` — a Textual :class:`~textual.widgets.RichLog` that owns the ring
  buffer (:class:`~wiki_ragify.ui.eventlog.EventLog`) and repaints on new events.

The panel accepts events from the pipeline worker (thread-safe append) and shows the same
``[N/total] ACCEPT title`` rhythm as the old ``chunk_gate.py``. Each log line is written
with the row's color as Rich markup so it renders in the panel's native palette
(see :mod:`wiki_ragify.ui.eventlog`).
"""

from __future__ import annotations

from typing import Optional

from textual.widgets import RichLog

from ...pipeline.checkpoint import State, phase_progress
from ...pipeline.events import ProgressEvent
from ..eventlog import LogRow, EventLog


# A single rendered decision log line.
class LogLine:
    text: str
    style: str

    def __init__(self, text: str, style: str) -> None:
        self.text = text
        self.style = style


def render_line(event: ProgressEvent) -> LogLine:
    """Turn a :class:`ProgressEvent` into a ``(text, style)`` log line (pure)."""
    style = EventLog.classify(event.kind)
    if event.total is not None and event.index:
        prefix = f"[{event.index}/{event.total}] "
        text = f"{prefix}{event.message}".strip()
    else:
        text = event.message
    return LogLine(text, style)


class ProgressPanel(RichLog):
    """A RichLog showing a scrolling decision log + current-stage banner.

    Thread-safe: :meth:`add_event` appends from the pipeline worker thread; rendering is a
    plain :meth:`RichLog.write` call. Keeps at most the last N rows (see
    :class:`~wiki_ragify.ui.eventlog.EventLog`).
    """

    def __init__(self, max_rows: int = 500) -> None:
        super().__init__(markup=True, id="progress-panel")
        self._eventlog = EventLog(max=max_rows)
        self.current_stage: Optional[str] = None
        self.state: Optional[State] = None
        self.status_label: str = ""
        self.write(
            "[subtle]No project selected. Choose a project in the folder picker.[/]"
        )

    def add_event(self, event: ProgressEvent) -> None:
        """Append a ProgressEvent to the log (safe to call from any thread)."""
        line = render_line(event)
        self._eventlog.append(event)
        self.current_stage = event.stage
        self._write_line(line)

    def set_state(self, state: State) -> None:
        """Attach a checkpoint State and re-render the per-stage status banner."""
        self.state = state
        done, total = phase_progress(state)
        self.status_label = f"{done}/{total} phases complete"
        self.write(f"[accent2]{self.status_label}[/]")
        self.write(
            f"[subtle]checkpoint: {state.state_path.name}[/]",
        )

    def clear(self) -> None:
        self._eventlog.clear()
        self.current_stage = None
        self.status_label = ""
        self.clear()

    def _write_line(self, line: LogLine) -> None:  # pragma: no cover - Textual render
        self.write(f"[{line.style}]{line.text}[/]")

    def render_snapshot_rows(self) -> list[tuple[str, str]]:
        """Return the current ``(text, style)`` rows, newest-first — for testing / diffing."""
        return [(row.text, row.style) for row in self._eventlog.rows()]
