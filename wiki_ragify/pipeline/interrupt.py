"""Shared interruption primitives for the pipeline.

The runner (``pipeline/runner.py``) owns the pause/stop :class:`threading.Event`s, but the
long phase bodies -- crawl, split, chunk gate, ingest -- run in their own modules. They
share one contract through a small neutral module so neither imports the other (no
circular import):

* :func:`interrupt` -- the runner's between-iterations probe: block while paused, raise
  :class:`StopRequested` when stopped.
* :class:`StopRequested` -- raised by a phase body to exit cleanly mid-item, leaving its
  phase marked incomplete so a resumed run re-runs it (safe default) rather than skipping
  interrupted work as if finished.
"""

from __future__ import annotations

import threading
import time


class StopRequested(Exception):
    """Raised inside a phase body when a stop was requested between items."""


def make_interrupt(pause: "threading.Event", stop: "threading.Event", poll: float = 0.05):
    """Return a between-iterations probe bound to ``pause`` / ``stop`` events.

    Blocks while paused (returning immediately if a stop is pending) and raises
    :class:`StopRequested` when the run has been stopped. ``poll`` sets how often the
    events are checked while waiting.
    """
    def interrupt() -> None:
        while pause.is_set():
            if stop.is_set():
                raise StopRequested()
            time.sleep(poll)
        if stop.is_set():
            raise StopRequested()

    return interrupt
