"""Archive stage — move accepted chunks, gzip-tar them, verify integrity, purge.

Faithful port of ``ArchivalHandler.archive_and_purge`` from
``reference/pipeline_orchestrator.py``. The one structural change is that it works
in-process (no subprocess) and returns a small result dataclass so the runner and TUI
can report exactly what happened.

Faithful behaviors preserved:

* Move every accepted chunk into ``knowledge/`` before taring.
* Create ``<archive_name>.tgz`` with ``tarfile`` (``w:gz``).
* **Full-stream integrity verification** — read each member's complete decompressed
  data and check it against the declared header size. A bare ``getmembers()`` only
  parses tar headers (and skips gzip); a truncated or partially-written ``.tgz`` would
  still "pass" that, so this guarantees the gzip stream and contents are fully intact.
* **Empty-input graceful no-op** — if the gate accepted nothing, archive is a
  successful no-op rather than a confusing "directory not found" failure.
* Silent purge of the intermediate ``chunks/`` directory afterward.
"""

from __future__ import annotations

import logging
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ArchiveResult:
    """Outcome of an archive attempt."""

    archived: bool
    skipped: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.archived or self.skipped


def archive_accepted(
    output_dir: Path,
    chunks_dirname: str,
    accept_dirname: str,
    knowledge_dirname: str,
    archive_name: str,
    dry_run: bool = False,
) -> ArchiveResult:
    """Archive accepted chunks to ``<archive_name>.tgz`` with full-stream verification.

    Mirrors the reference: moves accepted chunks into ``knowledge/``, tars, verifies
    every member fully decompresses, then purges ``chunks/``. On empty input this is a
    successful no-op (``ArchiveResult(skipped=True)``).
    """
    output_dir = Path(output_dir)
    chunks_dir = output_dir / chunks_dirname
    accept_dir = chunks_dir / accept_dirname
    knowledge_dir = output_dir / knowledge_dirname
    archive_path = output_dir / archive_name

    # Empty-input graceful no-op — gate may have accepted nothing (e.g. every tiny
    # article dropped by the splitter). Archiving an empty set would fail with a
    # confusing "directory not found" error, so treat it as success.
    if not dry_run and (not accept_dir.exists() or not any(accept_dir.iterdir())):
        logger.warning("No chunks accepted by the gate — nothing to archive. Skipping phase 5 (no failure).")
        return ArchiveResult(archived=False, skipped=True)

    if dry_run:
        logger.info("[DRY-RUN] Would archive accepted chunks to %s", archive_path)
        return ArchiveResult(archived=True, skipped=False)

    # 1. Ensure knowledge directory exists.
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    # 2. Move accepted chunks into knowledge/ (staging area before taring).
    if not accept_dir.exists():
        logger.error("Accepted chunks directory not found.")
        return ArchiveResult(archived=False, error="accepted chunks directory not found")

    for item in accept_dir.iterdir():
        shutil.move(str(item), str(knowledge_dir / item.name))
        logger.info("Moved: %s", item.name)

    # 3. Create the gzip tar archive.
    try:
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(knowledge_dir, arcname="knowledge")
        logger.info("Archived to: %s", archive_path)
    except Exception as e:  # noqa: BLE001
        logger.error("Archival failed: %s", e)
        return ArchiveResult(archived=False, error=f"tar failed: {e}")

    # 4. Verify archive integrity — read every member's full decompressed data so a
    #    truncated or partially-written .tgz cannot pass.
    if not archive_path.exists() or archive_path.stat().st_size == 0:
        logger.error("Archive verification failed: empty or missing.")
        return ArchiveResult(archived=False, error="archive empty or missing")

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isreg():
                    data = tar.extractfile(member).read()
                    if member.size and len(data) != member.size:
                        raise ValueError(
                            f"Archive member '{member.name}' decompressed to {len(data)} "
                            f"bytes but header claims {member.size}"
                        )
    except (tarfile.TarError, ValueError, OSError) as e:
        logger.error("Archive integrity check failed: %s", e)
        return ArchiveResult(archived=False, error=f"integrity check failed: {e}")

    # 5. Silent purge of the intermediate chunks directory.
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir)
        logger.info("Intermediate chunks directory purged.")

    return ArchiveResult(archived=True)
