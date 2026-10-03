"""python -m wiki_ragify — boots the Textual TUI."""

from __future__ import annotations

import os
from pathlib import Path

from .config import Config, PipelineConfig
from .ui.app import WikiRagifyApp


def main() -> None:
    """Launch the full multi-screen TUI (FR-1..8)."""
    config = Config()
    root = Path(os.path.expanduser("~/wiki"))
    root.mkdir(parents=True, exist_ok=True)
    WikiRagifyApp(root, config).run()


if __name__ == "__main__":
    main()
