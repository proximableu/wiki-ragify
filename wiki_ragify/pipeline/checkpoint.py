"""Resumable pipeline state — the JSON checkpoint layer.

Faithful port of ``StateManager`` from ``reference/pipeline_orchestrator.py``,
minus the directory-based artifact validation (that lives in the runner, which can
stat files directly). Every durable phase boundary records ``"completed"`` so the
runner can resume from a stopped or crashed run.

The ``_normalize_state`` handler also covers the *legacy* shape that nested progress
under ``steps`` (``crawl_seed`` / ``crawl_list`` / ``split`` / ``gate_chunks`` /
``archive``). Only statuses explicitly equal to ``"completed"`` count as finished;
anything else — ``"pending"``, ``"failed"``, or missing — is treated as not-done,
which is the safe default: a failed phase always re-runs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# The five pipeline phases, in execution order. The names mirror the old
# orchestrator so state files remain comparable across the two implementations.
PHASE_1_SEED = "phase_1_seed"
PHASE_2_LIST = "phase_2_list"
PHASE_3_SPLIT = "phase_3_split"
PHASE_4_GATE = "phase_4_gate"
PHASE_5_ARCHIVE = "phase_5_archive"
PHASE_6_INGEST = "phase_6_ingest"

# Maps the legacy ``steps`` keys onto the current phase keys.
_LEGACY_PHASE_MAP = {
    "crawl_seed": PHASE_1_SEED,
    "crawl_list": PHASE_2_LIST,
    "split": PHASE_3_SPLIT,
    "gate_chunks": PHASE_4_GATE,
    "archive": PHASE_5_ARCHIVE,
}

# All six pipeline phases, in execution order (including the optional ingest phase).
ALL_PHASES = (
    PHASE_1_SEED,
    PHASE_2_LIST,
    PHASE_3_SPLIT,
    PHASE_4_GATE,
    PHASE_5_ARCHIVE,
    PHASE_6_INGEST,
)


class State:
    """Loads, normalizes, persists, and queries the pipeline checkpoint."""

    def __init__(self, state_path: Path):
        self.state_path = Path(state_path)
        self.state: dict[str, str] = self._load_state()

    @staticmethod
    def _default_state() -> dict[str, str]:
        """Fresh pipeline state with every phase marked pending."""
        return {
            PHASE_1_SEED: "pending",
            PHASE_2_LIST: "pending",
            PHASE_3_SPLIT: "pending",
            PHASE_4_GATE: "pending",
            PHASE_5_ARCHIVE: "pending",
        }

    @staticmethod
    def _normalize_state(raw: dict) -> dict[str, str]:
        """Normalize a persisted state file to the {phase: status} schema.

        Handles the legacy shape that nested progress under ``steps``. Unknown/extra
        keys are preserved rather than silently dropped.
        """
        normalized = State._default_state()

        # Legacy format: progress lives under "steps".
        if isinstance(raw.get("steps"), dict):
            for legacy_key, phase_key in _LEGACY_PHASE_MAP.items():
                if raw["steps"].get(legacy_key) == "completed":
                    normalized[phase_key] = "completed"
            return normalized

        # New format: phases are top-level keys.
        for key, value in raw.items():
            if key in normalized:
                normalized[key] = value
        return normalized

    def _load_state(self) -> dict[str, str]:
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

    def save(self) -> None:
        """Persist the current state to disk (indented JSON)."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def is_phase_complete(self, phase: str) -> bool:
        return self.state.get(phase) == "completed"

    def mark_phase_complete(self, phase: str) -> None:
        self.state[phase] = "completed"
        self.save()


def phase_progress(state: "State") -> tuple[int, int]:
    """Return ``(completed, total)`` phase counts for a checkpoint :class:`State`.

    Pure helper shared by the progress panel and the app: maps the persisted
    per-phase statuses onto the :data:`ALL_PHASES` pipeline phases. Only a status
    of ``"completed"`` counts; ``"pending"`` / ``"failed"`` / missing all count as
    not-done — the safe default.
    """
    total = len(ALL_PHASES)
    done = sum(1 for phase in ALL_PHASES if state.is_phase_complete(phase))
    return done, total
