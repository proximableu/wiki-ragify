"""Progress window widget — stage banner + progress bar + live decision log (FR-3).

The window is a Textual :class:`~textual.widgets.Container` (:class:`ProgressPanel`)
holding two child layers, cleanly separated so the display logic can be tested without
launching an app:

* A banner :class:`~textual.widgets.Static` above, showing the active stage label, an
  ``index/total`` progress bar with a percentage, and the ETA/rate. When no stage is
  running (e.g. a checkpoint was just loaded) the bar reflects ``done/total phases``.
  The pure banner math lives in :func:`progress_bar` / :func:`eta_str` / :func:`rate_str`
  / :func:`render_banner` — all unit-tested offline.
* A :class:`ProgressLog` (Textual :class:`~textual.widgets.RichLog`) below, which owns the
  ring buffer (:class:`~wiki_ragify.ui.eventlog.EventLog`) and repaints on new events. Each
  line is written with the row's color as Rich markup so it renders in the native palette
  (see :mod:`wiki_ragify.ui.eventlog`).

The panel is thread-safe: :meth:`add_event` appends from the pipeline worker thread and
re-renders the banner off the worker thread; Textual marshals the widget writes onto its own
loop. The banner and the log re-render on every event so the bar tracks the active stage
lively.
"""

from __future__ import annotations

import time
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import RichLog, Static

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
    if event.total is not None and event.index is not None:
        prefix = f"[{event.index}/{event.total}] "
        text = f"{prefix}{event.message}".strip()
    else:
        text = event.message
    return LogLine(text, style)


# --- Pure banner helpers (unit-tested) -----------------------------------------


def progress_bar(fraction: float, width: int = 20, fill: str = "█", empty: str = "░") -> str:
    """A fixed-width horizontal bar for ``fraction`` in ``[0, 1]`` (pure)."""
    fraction = min(1.0, max(0.0, fraction))
    filled = round(fraction * width)
    return fill * filled + empty * (width - filled)


def rate_str(per_sec: float) -> str:
    """Render a per-second rate, e.g. ``0.5/s`` (pure)."""
    if per_sec <= 0:
        return "0.0/s"
    return f"{per_sec:.1f}/s"


def eta_str(remaining: Optional[int], per_sec: float) -> str:
    """Render ``remaining / per_sec`` as ``Mm Ss`` (or ``Ss``), or ``--:--`` if unknown.

    Pure: no dependency on the wall clock beyond the explicit inputs.
    """
    if per_sec <= 0 or remaining is None:
        return "--:--"
    secs = max(0, int(remaining / per_sec))
    m, s = divmod(secs, 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def render_banner(
    stage: Optional[str],
    index: Optional[int],
    total: Optional[int],
    phases_done: int,
    phases_total: int,
    rate: float,
) -> str:
    """Render the stage banner text (pure).

    When a live ``index/total`` is available it drives the bar (the active stage), with the
    ETA/rate derived from the per-item ``rate``; otherwise the bar reflects
    ``phases_done / phases_total`` (a checkpoint / idle window). ``rate`` is only meaningful
    while a stage is actively advancing.
    """
    if index is not None and total:
        frac = index / total
        remaining = max(0, total - index)
        eta = eta_str(remaining, rate)
        suffix = f"ETA {eta}  ({rate_str(rate)})"
        label = f"[accent2]Stage · {stage}[/]" if stage else "[accent2]Stage[/]"
        return f"{label}\n{progress_bar(frac)} {frac * 100:3.0f}%  {suffix}"

    frac = (phases_done / phases_total) if phases_total else 0.0
    label = f"[accent2]{phases_done}/{phases_total} phases[/]"
    return f"{label}\n{progress_bar(frac)} {frac * 100:3.0f}%  ETA --:--"


class ProgressLog(RichLog):
    """The scrollable decision log (``[N/total] ACCEPT title`` rows)."""

    def __init__(self, max_rows: int = 500) -> None:
        super().__init__(markup=True, id="progress-log")
        self._eventlog = EventLog(max=max_rows)
        self.write("[subtle]No project selected. Choose a project in the folder picker.[/]")

    def add_event(self, event: ProgressEvent) -> None:
        """Append a ProgressEvent to the log (safe to call from any thread)."""
        self._eventlog.append(event)
        # The per-[LINK] crawl bookkeeping rows are hidden from the panel.
        if EventLog._is_link_tick(getattr(event, "message", "")):
            return
        self._write_line(render_line(event))

    def clear(self) -> None:
        self._eventlog.clear()
        super().clear()

    def _write_line(self, line: LogLine) -> None:  # pragma: no cover - Textual render
        self.write(f"[{line.style}]{line.text}[/]")

    def render_snapshot_rows(self) -> list[tuple[str, str]]:
        """Return the current ``(text, style)`` rows, newest-first — for testing / diffing."""
        return [(row.text, row.style) for row in self._eventlog.rows()]


class ProgressPanel(Container):
    """A stage banner (name + progress bar + ETA/rate) above the live decision log.

    The banner sits above the log so the user sees the *stage's* progress bar and ETA first,
    then the rolling ``[N/total] ACCEPT/REJECT`` decision log beneath it.

    Thread-safe: :meth:`add_event` appends from the pipeline worker thread and re-renders the
    banner; :meth:`set_state` re-renders the banner from a checkpoint (done/total phases).
    """

    CSS_PATH = "styles.tcss"

    def __init__(self, max_rows: int = 500) -> None:
        super().__init__(id="progress-panel")
        self._banner = Static("", id="stage-banner")
        self._log = ProgressLog(max_rows=max_rows)
        # Live stage progress (index/total for the current stage).
        self.current_stage: Optional[str] = None
        self._stage_index: Optional[int] = None
        self._stage_total: Optional[int] = None
        self._stage_start: Optional[float] = None
        self._stage_track: Optional[str] = None
        self.status_label: str = ""
        # Checkpoint view (idle / after CONFIG).
        self.state: Optional[State] = None
        self._phases_done = 0
        self._phases_total = 0

    def compose(self) -> ComposeResult:
        yield self._banner
        yield self._log

    # --- public API (kept on the panel, per app.py references) ------------------

    def add_event(self, event: ProgressEvent) -> None:
        """Append a ProgressEvent to the log and update the stage banner.

        Safe to call from any thread. Records the current stage's ``index/total`` so the
        banner bar tracks the active stage, and resets the per-stage start when a new stage
        begins (so the ETA rate is per-stage, not for the whole run).
        """
        self._log.add_event(event)
        self.current_stage = event.stage

        if event.total is not None and event.index is not None:
            if event.stage != self._stage_track:
                # New stage: start counting its rate from scratch.
                self._stage_track = event.stage
                self._stage_start = event.ts or time.monotonic()
            self._stage_index = event.index
            self._stage_total = event.total
        # A stage_done/info event with no index leaves the last-known index; the next
        # stage's first event resets the tracker above.

        self._render_banner()

    def set_state(self, state: State) -> None:
        """Attach a checkpoint State and switch the banner to the done/total phases view."""
        self.state = state
        done, total = phase_progress(state)
        self._phases_done = done
        self._phases_total = total
        # Drop any live-stage tracking so the banner shows the phase bar, not a stage bar.
        self._stage_index = None
        self._stage_total = None
        self._stage_start = None
        self._stage_track = None
        self.status_label = f"{done}/{total} phases complete"
        self._render_banner()
        self._log.write(
            f"[subtle]checkpoint: {state.state_path.name}[/]",
        )

    def clear(self) -> None:
        self._log.clear()
        self.current_stage = None
        self._stage_index = None
        self._stage_total = None
        self._stage_start = None
        self._stage_track = None
        self._phases_done = 0
        self._phases_total = 0
        self.status_label = ""
        self._render_banner()

    # --- banner rendering ------------------------------------------------------

    @property
    def _elapsed_s(self) -> float:
        if self._stage_start is None:
            return 0.0
        return max(0.0, (time.monotonic() - self._stage_start))

    @property
    def _stage_rate(self) -> float:
        """Per-item rate for the active stage (items/sec)."""
        elapsed = self._elapsed_s
        if elapsed <= 0 or not self._stage_index:
            return 0.0
        return self._stage_index / elapsed

    def _render_banner(self) -> None:  # pragma: no cover - Textual render
        self._banner.content = render_banner(
            stage=self.current_stage,
            index=self._stage_index,
            total=self._stage_total,
            phases_done=self._phases_done,
            phases_total=self._phases_total,
            rate=self._stage_rate,
        )
