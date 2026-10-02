"""Single-run advisory lock — in-process flock over a lock file.

Faithful port of the reference orchestrator's ``fcntl.flock(LOCK_EX)`` guard, which
keeps two concurrent runs sharing one output directory from racing on the archive /
purge step. Unlike the old code, the lock here lives in a small context-manager-style
object that the :class:`~wiki_ragify.pipeline.runner.PipelineRunner` holds for the
whole run and releases on exit.

The OS advisory lock is *advisory*: only processes that also take this lock honour it.
That matches the old orchestrator (which also used ``fcntl`` alone) and is the
convention for a single-host tool.
"""

from __future__ import annotations

import errno
import fcntl
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PipelineLock:
    """Owns an ``flock``-based exclusive lock for one pipeline run.

    ``.release()`` unlocks and closes the backing file handle; the lock is
    released again when the process exits.
    """

    def __init__(self, lock_path: Path):
        self.lock_path = Path(lock_path)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.lock_path, "w")

    @staticmethod
    def acquire(lock_path: Path, timeout: float = 30.0) -> Optional["PipelineLock"]:
        """Try to acquire an exclusive lock, backing off briefly if another hold owns it.

        Returns the :class:`PipelineLock` on success, or ``None`` if it could not be
        acquired within ``timeout`` seconds (another run is in progress).
        """
        candidate = PipelineLock(lock_path)
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(candidate._file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return candidate
            except OSError as e:
                if e.errno not in (errno.EACCES, errno.EAGAIN):
                    logger.error("Unexpected error acquiring pipeline lock: %s", e)
                    candidate._file.close()
                    return None
                if time.monotonic() >= deadline:
                    logger.error("Timed out acquiring pipeline lock at %s", lock_path)
                    candidate._file.close()
                    return None
                time.sleep(0.5)

    def release(self) -> None:
        """Release the flock and close the backing file handle."""
        try:
            fcntl.flock(self._file, fcntl.LOCK_UN)
        finally:
            self._file.close()


def acquire_lock(lock_path: Path, timeout: float = 30.0) -> Optional[PipelineLock]:
    """Convenience wrapper around :meth:`PipelineLock.acquire`."""
    return PipelineLock.acquire(lock_path, timeout=timeout)
