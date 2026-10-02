"""Event contract between the pipeline runners and the Textual UI.

Every stage runner emits ``ProgressEvent``s; the UI subscribes and renders. The library
never depends on the UI and vice-versa — only these dataclasses bridge them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Stage identifiers used across the pipeline.
STAGE_CRAWL = "crawl"
STAGE_SPLIT = "split"
STAGE_GATE = "gate"
STAGE_ARCHIVE = "archive"
STAGE_INGEST = "ingest"

# Event kinds.
KIND_TICK = "tick"
KIND_ACCEPT = "accept"
KIND_REJECT = "reject"
KIND_SKIP = "skip"
KIND_INFO = "info"
KIND_WARN = "warn"
KIND_STAGE_DONE = "stage_done"


@dataclass
class ProgressEvent:
    """A single tick of progress from a pipeline stage."""

    stage: str
    index: int = 0
    total: Optional[int] = None
    kind: str = KIND_TICK
    message: str = ""
    ts: float = 0.0


@dataclass
class StatEvent:
    """A cumulative statistic to display in the rolling stats panel."""

    name: str
    value: float
    ts: float = 0.0
