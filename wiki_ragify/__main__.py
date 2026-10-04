"""python -m wiki_ragify — boots the Textual TUI."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .config import Config, PipelineConfig
from .ui.app import WikiRagifyApp


def main() -> None:
    """Launch the full multi-screen TUI (FR-1..8).

    The output root defaults to ``~/wiki`` but can be overridden with the first
    command-line argument or the ``WIKI_RAGIFY_ROOT`` environment variable
    (DESIGN FR-1: a chosen ``output/`` root — "or whatever the user picks"). The
    folder picker then walks that root and lists each immediate child directory as
    a project; selecting one and pressing RUN starts the funnel for it.
    """
    config = Config()
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        os.environ.get("WIKI_RAGIFY_ROOT", "~/wiki")
    )
    root.mkdir(parents=True, exist_ok=True)
    WikiRagifyApp(root, config).run()


if __name__ == "__main__":
    main()
