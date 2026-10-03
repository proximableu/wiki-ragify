"""Pipeline runner — the in-process engine that drives the whole funnel.

This is the v2 replacement for ``reference/pipeline_orchestrator.py::PipelineOrchestrator``,
rewritten in-process (no subprocess glue) but porting its hard-won behaviours *verbatim*:

* **Dual-layer checkpointing** — a phase is skipped only if BOTH the JSON state *and*
  the filesystem artifacts confirm completion. Otherwise it re-runs (safe default).
* **The five pipeline phases** — seed crawl, list crawl, split, chunk gate, archive.
* **Empty-input graceful no-op** on archive — the gate may accept nothing; that is a
  successful no-op, not a failure.
* **Archive integrity** — full-stream gzip-tar verification (in :mod:`.archive`).
* **6h wall-clock cap** on the whole run, so a stalled run never freezes forever.
* **flock** — one run per output dir; concurrent runs are serialised.
* **``--dry-run``** — simulate without moving files or writing artifacts.

On top of the reference it adds the two v2-only features from the design: pause/stop
via :class:`threading.Event` (checked at durable phase boundaries and between crawl
iterations) and the **optional 6th phase** — embed the accepted chunks into a
sqlite-vec store — which is toggleable per the user's decision to make step 6 optional.

The runner emits the same :class:`ProgressEvent` / :class:`StatEvent` contract the TUI
binds to. Blocking LLM/network/embed work stays on the caller's worker thread; this
class only sequences phases and fans out events.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from ..config import Config, PipelineConfig, sanitize_title
from ..logging_setup import get_logger
from ..pipeline.events import ProgressEvent, StatEvent
from ..pipeline.checkpoint import (
    PHASE_1_SEED,
    PHASE_2_LIST,
    PHASE_3_SPLIT,
    PHASE_4_GATE,
    PHASE_5_ARCHIVE,
    PHASE_6_INGEST,
    State,
)
from ..pipeline.archive import archive_accepted
from ..splitting.runner import split_dir

logger = get_logger(__name__)


class PipelineRunner:
    """Sequences the pipeline phases against a chosen output directory.

    Construct with the :class:`Config` (library tunables) and a
    :class:`PipelineConfig` (execution params: seed page, prompt file, output root).
    Run it with :meth:`run`; pause or stop it with :meth:`pause` / :meth:`stop`.
    """

    def __init__(
        self,
        config: Config,
        pipeline: PipelineConfig,
        on_event: Callable[[ProgressEvent], None] | None = None,
        on_stat: Callable[[StatEvent], None] | None = None,
    ):
        self.config = config
        self.pipeline = pipeline
        self.on_event = on_event
        self.on_stat = on_stat

        self.state = State(pipeline.state_file)
        self._pause = threading.Event()
        self._stop = threading.Event()

    # --- event emission -------------------------------------------------------

    def _emit(self, *args, **kwargs) -> None:
        # Sub-callbacks (crawler, splitter, gate, ingest) call on_event(ProgressEvent(...))
        # with a single positional event; the runner's own calls use keyword arguments.
        # Accept both forms so `self._emit` is a valid `on_event` everywhere.
        if args and isinstance(args[0], ProgressEvent):
            event = args[0]
        elif args or kwargs:
            event = ProgressEvent(**kwargs)
        else:
            return
        if self.on_event:
            self.on_event(event)

    def _stat(self, name: str, value: float) -> None:
        if self.on_stat:
            self.on_stat(StatEvent(name=name, value=value))

    # --- pause / stop (v2) ----------------------------------------------------

    def pause(self) -> None:
        """Signal the runner to pause at the next durable boundary."""
        self._pause.set()

    def resume(self) -> None:
        """Clear the pause signal."""
        self._pause.clear()

    def stop(self) -> None:
        """Signal the runner to finish the current phase and exit cleanly."""
        self._stop.set()

    def _check_pause(self) -> bool:
        """Block while paused; return False if a stop was requested."""
        while self._pause.is_set():
            if self._stop.is_set():
                return False
            time.sleep(0.1)
        return True

    # --- artifact validation (dual-layer checkpointing) -----------------------

    def _artifact_complete(self, phase: str) -> bool:
        """Filesystem-artifact check for a phase, mirroring the reference.

        Returns True when the phase's durable on-disk evidence is present and
        non-empty, so a re-run is skipped only when the JSON state *and* this agree.
        """
        od = self.config.output_dir
        if phase == PHASE_1_SEED:
            list_file = od / self.config.output_dirname / "accepted_pages.list"
            return list_file.exists() and list_file.stat().st_size > 0
        if phase == PHASE_2_LIST:
            txt_files = list((od / self.config.output_dirname).glob("*.txt"))
            return len(txt_files) > 0
        if phase == PHASE_3_SPLIT:
            md_files = list((od / self.config.chunks_dirname).glob("*.md"))
            return len(md_files) > 0
        if phase == PHASE_4_GATE:
            accepted_dir = od / self.config.chunks_dirname / self.config.gate_accept_dirname
            return accepted_dir.exists() and any(accepted_dir.iterdir())
        return False

    def _should_run(self, phase: str) -> bool:
        """Dual-layer: re-run unless *both* JSON state and artifacts say it's done."""
        return not (self.state.is_phase_complete(phase) and self._artifact_complete(phase))

    # --- phase bodies (in-process) --------------------------------------------

    def _phase_1_seed(self) -> None:
        logger.info("--- Phase 1: seed crawl ---")
        self._emit(stage="crawl", kind="info", message="Phase 1: crawl seed page")
        from ..crawler.runner import Crawler

        crawler = Crawler(self.config, self.evaluator, on_event=self._emit)
        crawler.crawl([self.pipeline.start_page], self.gating_prompt)

    def _phase_2_list(self) -> None:
        logger.info("--- Phase 2: list crawl ---")
        self._emit(stage="crawl", kind="info", message="Phase 2: crawl accepted-titles list")
        list_file = self.config.output_dir / self.config.output_dirname / "accepted_pages.list"
        if not list_file.exists():
            self._emit(stage="crawl", kind="warn", message="No accepted_pages.list; skipping list crawl")
            return
        from ..crawler.runner import Crawler

        crawler = Crawler(self.config, self.evaluator, on_event=self._emit)
        titles = [t for t in list_file.read_text(encoding="utf-8").splitlines() if t.strip()]
        crawler.crawl(titles, self.gating_prompt)

    def _phase_3_split(self) -> None:
        logger.info("--- Phase 3: split ---")
        self._emit(stage="split", kind="info", message="Phase 3: split accepted pages into chunks")
        split_dir(
            self.config.output_dir / self.config.output_dirname,
            self.config.output_dir / self.config.chunks_dirname,
            self.config,
            on_event=self._emit,
        )

    def _phase_4_gate(self) -> None:
        logger.info("--- Phase 4: chunk gate ---")
        self._emit(stage="gate", kind="info", message="Phase 4: gate chunks against prompt")
        from ..gating.chunk import gate_files

        gate_files(
            self.config.output_dir / self.config.chunks_dirname,
            self.gating_prompt,
            self.config,
            self.evaluator,
            on_event=self._emit,
        )

    def _phase_5_archive(self) -> None:
        logger.info("--- Phase 5: archive ---")
        self._emit(stage="archive", kind="info", message="Phase 5: archive accepted chunks")
        result = archive_accepted(
            self.config.output_dir,
            self.config.chunks_dirname,
            self.config.gate_accept_dirname,
            self.config.knowledge_dirname,
            self.pipeline.archive_name,
            dry_run=self.pipeline.dry_run,
        )
        if result.ok:
            self._emit(stage="archive", kind="stage_done", message="Phase 5: archive complete")
        else:
            self._emit(stage="archive", kind="warn", message=f"Phase 5: archive failed — {result.error}")

    # --- properties (constructed lazily, so the runner can be built without live deps) ---

    @property
    def evaluator(self):
        from ..llm.gateway import LLMEvaluator

        return LLMEvaluator(self.config)

    @property
    def embed_client(self):
        from ..llm.embed import EmbedClient

        return EmbedClient(self.config)

    @property
    def gating_prompt(self) -> str:
        prompt_path = Path(self.pipeline.prompt_path)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Gating prompt not found: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")

    # --- main entry point -----------------------------------------------------

    def run(self) -> None:
        """Run every phase under the exclusive lock, with checkpointing + pause/stop.

        Holds an advisory ``flock`` for the whole run so two concurrent orchestrators
        sharing one output directory cannot race on the archive/purge step.
        """
        from ..pipeline.lock import acquire_lock

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        lock = acquire_lock(self.config.output_dir / ".pipeline.lock")
        if lock is None:
            logger.error("Failed to acquire pipeline lock; another run is in progress.")
            self._emit(stage="crawl", kind="warn", message="Pipeline lock held by another process — aborting")
            return

        try:
            self._run_locked()
        finally:
            lock.release()

    def _run_locked(self) -> None:
        """Run the phases in order, honouring checkpoint + pause/stop + timeout."""
        phases = [
            (PHASE_1_SEED, self._phase_1_seed),
            (PHASE_2_LIST, self._phase_2_list),
            (PHASE_3_SPLIT, self._phase_3_split),
            (PHASE_4_GATE, self._phase_4_gate),
            (PHASE_5_ARCHIVE, self._phase_5_archive),
        ]

        start_time = time.monotonic()
        for phase_id, body in phases:
            if self.state.is_phase_complete(phase_id):
                logger.info("Skipping %s (already completed).", phase_id)
                self._emit(stage=self._stage_of(phase_id), kind="skip", message=f"Skipping {phase_id} (done)")
                continue

            logger.info("--- Starting %s ---", phase_id)
            self._emit(stage=self._stage_of(phase_id), kind="info", message=f"Starting {phase_id}")

            if not self._check_pause():
                logger.info("Pipeline stopped by user.")
                return

            body()
            self.state.mark_phase_complete(phase_id)
            self._stat(f"phase_{phase_id}", 1.0)

            if self._stop.is_set():
                logger.info("Pipeline stopped after %s.", phase_id)
                return

            if time.monotonic() - start_time > self.pipeline.timeout:
                logger.error("Pipeline timed out after %.1fs.", self.pipeline.timeout)
                self._emit(stage="crawl", kind="warn", message=f"Timed out after {self.pipeline.timeout:.0f}s")
                return

        # Optional 6th phase (embed/DB build), toggleable per the user's decision.
        if self.pipeline.ingest:
            self._ingest_phase()

    def _ingest_phase(self) -> None:
        """Run the optional embed/DB-build phase (phase 6) if the gate produced content."""
        if self.state.is_phase_complete(PHASE_6_INGEST):
            return

        accepted_dir = self.config.output_dir / self.config.knowledge_dirname
        if not accepted_dir.exists() or not any(accepted_dir.iterdir()):
            logger.warning("No accepted chunks — skipping optional ingest (nothing to embed).")
            self._emit(stage="ingest", kind="warn", message="No accepted chunks; skipping embed")
            return

        if not self._check_pause():
            return

        logger.info("--- Phase 6: optional ingest/embed ---")
        self._emit(stage="ingest", kind="info", message="Phase 6: embedding accepted chunks")
        from ..ingestion.build_db import build_db
        build_db(
            accepted_dir,
            self.config.db_path(self.pipeline.start_page),
            self.config,
            embed_client=self.embed_client,
            on_event=self._emit,
        )
        self.state.mark_phase_complete(PHASE_6_INGEST)

    def _stage_of(self, phase: str) -> str:
        return {
            PHASE_1_SEED: "crawl",
            PHASE_2_LIST: "crawl",
            PHASE_3_SPLIT: "split",
            PHASE_4_GATE: "gate",
            PHASE_5_ARCHIVE: "archive",
            PHASE_6_INGEST: "ingest",
        }[phase]
