"""Pipeline — the orchestrator as a library (Phase 2).

Public surface:

* :class:`pipeline.runner.PipelineRunner` — drives the whole funnel in-process.
* :class:`pipeline.checkpoint.State` — resumable JSON checkpoint.
* :func:`pipeline.archive.archive_accepted` — gzip-tar + integrity + empty-input no-op.
* :func:`pipeline.lock.acquire_lock` / :class:`pipeline.lock.PipelineLock` — single-run flock.
* :class:`pipeline.events.ProgressEvent`, :class:`pipeline.events.StatEvent` — the contract.
"""

from .checkpoint import State
from .archive import archive_accepted, ArchiveResult
from .runner import PipelineRunner
from .lock import PipelineLock, acquire_lock
from .events import ProgressEvent, StatEvent

__all__ = [
    "State",
    "archive_accepted",
    "ArchiveResult",
    "PipelineRunner",
    "PipelineLock",
    "acquire_lock",
    "ProgressEvent",
    "StatEvent",
]
