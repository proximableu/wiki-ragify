"""Wiki Ragify v2 — a contained, homogenous TUI-driven wiki → knowledge-base pipeline.

Everything in the funnel (crawl → split → gate → archive → embed → retrieve) is importable
library code driven in-process by a Textual TUI. Nothing shells out to another script.
"""

from .config import Config
from .logging_setup import setup_logging

__all__ = ["Config", "setup_logging"]
__version__ = "2.0.0"
