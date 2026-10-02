#!/usr/bin/env python3
"""
Sequential Pipeline Orchestrator
---------------------------------
Orchestrates the Wikipedia crawling, splitting, gating, and archival pipeline.
Implements dual-layer checkpointing, real-time subprocess streaming, and deterministic fail-safe routing.
"""

import argparse
import fcntl
import hashlib
import json
import logging
import re
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Artifact folder layout under the output directory. Defined once so the splitter
# (phase 3), the gate (phase 4), and the archiver/validator (phases 4 & 5) all
# agree on where chunks live. The gate emits the "accept" dir (see chunk_gate.py),
# so this must stay "accept" — any mismatch silently makes the gate look empty.
OUTPUT_DIRNAME = "output"
CHUNKS_DIRNAME = "chunks"
GATE_ACCEPT_DIRNAME = "accept"
KNOWLEDGE_DIRNAME = "knowledge"


@dataclass
class PipelineConfig:
    """Centralized configuration for the pipeline execution."""
    start_page: str
    prompt_path: Path
    output_dir: Path
    dry_run: bool = False
    state_file: Path = field(init=False)
    archive_name: str = field(init=False)

    def __post_init__(self):
        self.state_file = self.output_dir / "pipeline_state.json"
        self.archive_name = f"knowledge_{self.sanitize_title(self.start_page)}.tgz"

    @staticmethod
    def sanitize_title(title: str) -> str:
        """Sanitizes page title for filesystem-safe archive naming.

        A short SHA-256 digest of the *full* original title is appended to the
        truncated, cleaned prefix. This keeps names readable and bounded while
        guaranteeing that two genuinely different long titles never collapse onto
        the same archive filename (which would let one run silently overwrite the
        other's archive).
        """
        safe = re.sub(r'[^a-zA-Z0-9\s]', '', title.lower())
        safe = re.sub(r'\s+', '_', safe)
        digest = hashlib.sha256(title.encode("utf-8")).hexdigest()[:8]
        return f"{safe[:100]}_{digest}"


class StateManager:
    """Manages JSON checkpointing and directory-based artifact validation."""

    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.state: Dict[str, str] = self._load_state()

    @staticmethod
    def _default_state() -> Dict[str, str]:
        """Fresh pipeline state with every phase marked pending."""
        return {
            "phase_1_seed": "pending",
            "phase_2_list": "pending",
            "phase_3_split": "pending",
            "phase_4_gate": "pending",
            "phase_5_archive": "pending",
        }

    @staticmethod
    def _normalize_state(raw: dict) -> Dict[str, str]:
        """Normalize a persisted state file to the {phase_N_xxx: status} schema.

        Handles the legacy shape that nested progress under ``steps``
        (``crawl_seed``, ``crawl_list``, ``split``, ``gate_chunks``, ``archive``).
        Only statuses that are explicitly ``"completed"`` count as finished;
        anything else (``"pending"``, ``"failed"``, missing) is treated as
        not-done, which is the safe default — a failed phase always re-runs.
        """
        normalized = StateManager._default_state()
        legacy_phase_map = {
            "crawl_seed": "phase_1_seed",
            "crawl_list": "phase_2_list",
            "split": "phase_3_split",
            "gate_chunks": "phase_4_gate",
            "archive": "phase_5_archive",
        }

        # Legacy format: progress lives under "steps".
        if isinstance(raw.get("steps"), dict):
            for legacy_key, phase_key in legacy_phase_map.items():
                if raw["steps"].get(legacy_key) == "completed":
                    normalized[phase_key] = "completed"
            return normalized

        # New format: phases are top-level keys (fall through to plain passthrough
        # so unknown/extra keys are preserved rather than silently dropped).
        for key, value in raw.items():
            if key in normalized:
                normalized[key] = value
        return normalized

    def _load_state(self) -> Dict[str, str]:
        """Loads existing state or initializes a fresh pipeline state."""
        if self.state_path.exists():
            try:
                raw = json.loads(self.state_path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    raise ValueError("state file is not a JSON object")
                return self._normalize_state(raw)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Corrupted state file found ({e}). Resetting.")
        return self._default_state()

    def save_state(self):
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def is_phase_complete(self, phase: str) -> bool:
        return self.state.get(phase) == "completed"

    def mark_phase_complete(self, phase: str):
        self.state[phase] = "completed"
        self.save_state()

    def check_artifact_fallback(self, phase: str, output_dir: Path, artifact: str | None = None) -> bool:
        """Directory-based artifact validation fallback for resilience."""
        if phase == "phase_1_seed":
            list_file = output_dir / OUTPUT_DIRNAME / "accepted_pages.list"
            return list_file.exists() and list_file.stat().st_size > 0
        elif phase == "phase_2_list":
            txt_files = list((output_dir / OUTPUT_DIRNAME).glob("*.txt"))
            return len(txt_files) > 0
        elif phase == "phase_3_split":
            md_files = list((output_dir / CHUNKS_DIRNAME).glob("*.md"))
            return len(md_files) > 0
        elif phase == "phase_4_gate":
            accepted_dir = output_dir / CHUNKS_DIRNAME / GATE_ACCEPT_DIRNAME
            return accepted_dir.exists() and any(accepted_dir.iterdir())
        elif phase == "phase_5_archive":
            # Phase 5 is complete only if the specific archive for this run exists and is non-empty.
            # (knowledge/ is created by phase 5 itself and is not a reliable signal on its own.)
            if not artifact:
                return False
            archive_file = output_dir / artifact
            return archive_file.exists() and archive_file.stat().st_size > 0
        return False


class ProcessRunner:
    """Handles subprocess invocation with real-time line-buffered streaming."""

    def __init__(self, timeout: float = 6 * 60 * 60) -> None:
        """
        Args:
            timeout: Per-command wall-clock limit in seconds (default 6h). A command
                that runs longer is killed so a stalled downstream process
                (e.g. Ollama hanging) cannot freeze the whole orchestrator forever.
        """
        self.timeout = timeout

    def run(self, command: List[str], dry_run: bool = False, timeout: float | None = None) -> int:
        """
        Args:
            timeout: Per-command wall-clock limit in seconds. Defaults to the runner's
                configured timeout (6h). A command that runs longer is killed so a
                stalled downstream process cannot freeze the orchestrator forever.
        """
        if timeout is None:
            timeout = self.timeout
        if dry_run:
            logger.info(f"[DRY-RUN] Would execute: {' '.join(command)}")
            return 0

        logger.info(f"Executing: {' '.join(command)}")
        try:
            # Popen enables non-blocking stdout consumption for real-time console visibility
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Stream output line-by-line to preserve ANSI colors and progress indicators
            if process.stdout:
                for line in process.stdout:
                    print(line, end="", flush=True)

            process.wait(timeout=timeout)
            return process.returncode
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout}s, killing: {command[0]}")
            process.kill()
            process.wait()
            return 1
        except FileNotFoundError:
            logger.error(f"Executable not found: {command[0]}")
            return 1
        except Exception as e:
            logger.error(f"Subprocess execution failed: {e}")
            return 1


class ArchivalHandler:
    """Manages atomic file movement, compression, integrity verification, and cleanup."""

    @staticmethod
    def archive_and_purge(output_dir: Path, archive_name: str, dry_run: bool = False) -> bool:
        chunks_accepted = output_dir / CHUNKS_DIRNAME / GATE_ACCEPT_DIRNAME
        knowledge_dir = output_dir / KNOWLEDGE_DIRNAME  # staging area the accepted chunks are moved into before taring
        archive_path = output_dir / archive_name

        if dry_run:
            logger.info(f"[DRY-RUN] Would move {chunks_accepted}/* to {knowledge_dir}")
            logger.info(f"[DRY-RUN] Would archive to {archive_path}")
            logger.info(f"[DRY-RUN] Would purge {output_dir / 'chunks'}")
            return True

        # 1. Ensure knowledge directory exists
        knowledge_dir.mkdir(parents=True, exist_ok=True)

        # 2. Move accepted chunks to knowledge directory
        if not chunks_accepted.exists():
            logger.error("Accepted chunks directory not found.")
            return False

        for item in chunks_accepted.iterdir():
            shutil.move(str(item), str(knowledge_dir / item.name))
            logger.info(f"Moved: {item.name}")

        # 3. Create tar.gz archive
        try:
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(knowledge_dir, arcname="knowledge")
            logger.info(f"Archived to: {archive_path}")
        except Exception as e:
            logger.error(f"Archival failed: {e}")
            return False

        # 4. Verify archive integrity.
        #    A bare ``getmembers()`` only parses the tar *headers* (and even skips
        #    gzip) — a truncated or partially-written .tgz would still "pass". Read
        #    each member's full data to guarantee the gzip stream and contents are
        #    fully intact and extractable.
        if not archive_path.exists() or archive_path.stat().st_size == 0:
            logger.error("Archive verification failed: empty or missing.")
            return False

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
            logger.error(f"Archive integrity check failed: {e}")
            return False

        # 5. Silent purge of intermediate chunks directory
        chunks_dir = output_dir / CHUNKS_DIRNAME
        if chunks_dir.exists():
            shutil.rmtree(chunks_dir)
            logger.info("Intermediate chunks directory purged.")

        return True


class PipelineOrchestrator:
    """Sequences pipeline phases, enforces checkpointing, and handles execution flow."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.state_manager = StateManager(config.state_file)
        self.runner = ProcessRunner()

    def _should_run_phase(self, phase: str, artifact: str | None = None) -> bool:
        """Dual-layer checkpointing: skips only if JSON state AND filesystem artifacts confirm completion."""
        json_complete = self.state_manager.is_phase_complete(phase)
        artifact_complete = self.state_manager.check_artifact_fallback(phase, self.config.output_dir, artifact)
        return not (json_complete and artifact_complete)

    def run(self):
        logger.info("Pipeline Orchestrator Starting...")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Guard against concurrent runs clobbering each other's artifacts. Two
        # orchestrators sharing an output dir could both pass the checkpoint
        # checks, then race on the archive/purge step. Hold an advisory lock for
        # the whole run so only one process manipulates this output directory.
        lock_path = self.config.output_dir / ".pipeline.lock"
        lock_file = open(lock_path, "w")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        except Exception as e:
            logger.error(f"Failed to acquire pipeline lock at {lock_path}: {e}")
            lock_file.close()
            return

        try:
            self._run_locked()
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()

    def _run_locked(self):
        """Runs the pipeline steps under an held exclusive lock."""

        phases = [
            ("phase_1_seed", self._build_phase_1_cmd),
            ("phase_2_list", self._build_phase_2_cmd),
            ("phase_3_split", self._build_phase_3_cmd),
            ("phase_4_gate", self._build_phase_4_cmd),
        ]

        for phase_id, cmd_builder in phases:
            if self._should_run_phase(phase_id):
                logger.info(f"--- Starting {phase_id} ---")
                cmd = cmd_builder()
                exit_code = self.runner.run(cmd, self.config.dry_run)
                if exit_code != 0:
                    logger.error(f"Phase {phase_id} failed with exit code {exit_code}. Halting pipeline.")
                    return
                self.state_manager.mark_phase_complete(phase_id)
            else:
                logger.info(f"Skipping {phase_id} (already completed).")

        # Phase 5 handles archival logic separately
        if self._should_run_phase("phase_5_archive", artifact=self.config.archive_name):
            logger.info("--- Starting phase_5_archive ---")

            # Graceful empty-input path: phase 4 may have accepted nothing (e.g. every
            # tiny article dropped by the splitter). Archiving an empty set would fail
            # with a confusing "directory not found" error and halt the run, so treat it
            # as a successful no-op instead — there's simply nothing to archive this time.
            if not self.config.dry_run:
                accepted_dir = self.config.output_dir / CHUNKS_DIRNAME / GATE_ACCEPT_DIRNAME
                if not accepted_dir.exists() or not any(accepted_dir.iterdir()):
                    logger.warning(
                        "No chunks accepted by the gate — nothing to archive. "
                        "Skipping phase 5 (no failure)."
                    )
                    self.state_manager.mark_phase_complete("phase_5_archive")
                    logger.info("Pipeline completed (empty result set).")
                    return

            success = ArchivalHandler.archive_and_purge(
                self.config.output_dir,
                self.config.archive_name,
                self.config.dry_run
            )
            if success:
                self.state_manager.mark_phase_complete("phase_5_archive")
                logger.info("Pipeline completed successfully.")
            else:
                logger.error("Archival phase failed. Halting pipeline.")
        else:
            logger.info("Skipping phase_5_archive (already completed).")

    def _build_phase_1_cmd(self) -> List[str]:
        return [
            sys.executable, "wiki_crawler.py",
            "--start-page", self.config.start_page,
            str(self.config.prompt_path),
            str(self.config.output_dir / OUTPUT_DIRNAME)
        ]

    def _build_phase_2_cmd(self) -> List[str]:
        return [
            sys.executable, "wiki_crawler.py",
            "--start-list", str(self.config.output_dir / OUTPUT_DIRNAME / "accepted_pages.list"),
            str(self.config.prompt_path),
            str(self.config.output_dir / OUTPUT_DIRNAME)
        ]

    def _build_phase_3_cmd(self) -> List[str]:
        return [
            sys.executable, "wiki_splitter.py",
            str(self.config.output_dir / OUTPUT_DIRNAME),
            str(self.config.output_dir / CHUNKS_DIRNAME),
            "--max-tokens", "2048"
        ]

    def _build_phase_4_cmd(self) -> List[str]:
        # Fixed: Dynamically resolve prompt path from CLI configuration instead of hardcoding "prompt.txt"
        return [
            sys.executable, "chunk_gate.py",
            str(self.config.output_dir),
            str(self.config.output_dir / CHUNKS_DIRNAME),
            "--prompt-file", str(self.config.prompt_path)
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential Wikipedia Pipeline Orchestrator")
    parser.add_argument("--start-page", type=str, required=True, help="Initial Wikipedia article title")
    parser.add_argument("--prompt", type=Path, required=True, help="Path to gating prompt file")
    parser.add_argument("output_directory", type=Path, help="Base directory for pipeline artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Simulate execution without running subprocesses")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.prompt.exists():
        logger.error(f"Prompt file not found: {args.prompt}")
        sys.exit(1)

    config = PipelineConfig(
        start_page=args.start_page,
        prompt_path=args.prompt,
        output_dir=args.output_directory,
        dry_run=args.dry_run
    )

    orchestrator = PipelineOrchestrator(config)
    orchestrator.run()


if __name__ == "__main__":
    main()