"""Folder picker widget — browse the chosen output root (FR-1).

A :class:`textual.widgets.Tree` subclass over an output root. Selecting a project folder
loads its :class:`~wiki_ragify.pipeline.checkpoint.State` into the progress panel so the
user can see where a run stopped and resume from it.

The directory-walking + checkpoint-mapping logic lives in the pure helpers
:func:`list_projects` and :class:`ProjectInfo`, which can be unit-tested without launching
an app; the Tree construction lives entirely in :meth:`on_mount`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from textual.message import Message
from textual.widgets import Tree

from ...pipeline.checkpoint import (
    PHASE_1_SEED,
    PHASE_2_LIST,
    PHASE_3_SPLIT,
    PHASE_4_GATE,
    PHASE_5_ARCHIVE,
    PHASE_6_INGEST,
)


# The per-phase checkpoint key a project directory carries, in phase order.
_PHASE_KEYS = (
    PHASE_1_SEED,
    PHASE_2_LIST,
    PHASE_3_SPLIT,
    PHASE_4_GATE,
    PHASE_5_ARCHIVE,
    PHASE_6_INGEST,
)


@dataclass
class ProjectInfo:
    """Where a project lives and how far its pipeline has progressed."""

    project_dir: Path
    state_file: Path

    @property
    def name(self) -> str:
        return self.project_dir.name

    def status(self) -> Optional[dict[str, str]]:
        """Return the loaded phase->status map, or None when no checkpoint exists."""
        if not self.state_file.exists():
            return None
        try:
            raw = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(raw, dict):
            return None

        # New schema: phases are top-level keys.
        if any(k in raw for k in _PHASE_KEYS):
            return {k: raw.get(k, "pending") for k in _PHASE_KEYS}

        # Legacy schema: progress nested under ``steps: {crawl_seed, ...}``.
        steps = raw.get("steps")
        if isinstance(steps, dict):
            legacy_map = {
                "crawl_seed": PHASE_1_SEED,
                "crawl_list": PHASE_2_LIST,
                "split": PHASE_3_SPLIT,
                "gate_chunks": PHASE_4_GATE,
                "archive": PHASE_5_ARCHIVE,
            }
            return {
                phase: "completed" if steps.get(legacy_key) == "completed" else "pending"
                for legacy_key, phase in legacy_map.items()
            }

        return {k: "pending" for k in _PHASE_KEYS}


def list_projects(root: Union[str, Path]) -> list[ProjectInfo]:
    """Return a :class:`ProjectInfo` for each project directory directly under ``root``.

    A project is any immediate child directory of the output root (``output/Autistic-supremacism``,
    ``output/project_B``, …). Pure directory walking — the only file I/O is reading each
    project's ``pipeline_state.json``, guarded by a try/except so one corrupt file never breaks
    the tree.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    projects: list[ProjectInfo] = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            projects.append(
                ProjectInfo(project_dir=child, state_file=child / "pipeline_state.json")
            )
    return projects


class ProjectSelected(Message):
    """Textual message carrying the chosen :class:`ProjectInfo`."""

    def __init__(self, project: ProjectInfo) -> None:
        super().__init__()
        self.project = project


class FolderTree(Tree):
    """A Textual ``Tree`` over an output root that reports project selections.

    Each node carries the :class:`ProjectInfo` as its ``data`` so the app can recover it
    from :data:`event.node.data` in :meth:`on_node_selected`.
    """

    def __init__(
        self, root: Union[str, Path], *, id: Optional[str] = None
    ) -> None:
        self._root = Path(root)
        super().__init__(label=root.name, id=id)

    def on_mount(self) -> None:  # pragma: no cover - Textual lifecycle
        root_node = self.root
        for project in list_projects(self._root):
            status = project.status()
            label = project.name
            if status:
                done = sum(1 for v in status.values() if v == "completed")
                label = f"{label}  ({done}/{len(_PHASE_KEYS)})"
            root_node.add(label, data=project)

    def on_node_selected(self, event: "Tree.NodeSelected") -> None:  # pragma: no cover
        node = event.node
        project: Optional[ProjectInfo] = node.data if node else None
        if isinstance(project, ProjectInfo):
            self.post_message(ProjectSelected(project))
