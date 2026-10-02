"""Uniform logging setup.

The old scripts each called ``logging.basicConfig(...)`` on import — which means the first
module that imported ``logging`` wonned the root config and later ones were no-ops. Every
module here instead gets its own named logger and relies on a single shared handler.
"""

from __future__ import annotations

import logging

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger exactly once. Safe to call repeatedly."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.setLevel(level)
    # Avoid attaching duplicate handlers if setup_logging is somehow called twice.
    if not root.handlers:
        root.addHandler(handler)
    # httpx is noisy on every Ollama call; keep it quiet unless something is actually wrong.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
