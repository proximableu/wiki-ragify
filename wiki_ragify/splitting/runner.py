"""Splitter runner — drives ``process_file`` over a directory and emits events."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List

from ..config import Config
from ..logging_setup import get_logger
from ..pipeline.events import ProgressEvent
from .splitter import process_file

logger = get_logger(__name__)


def split_dir(
    src_dir: Path,
    out_dir: Path,
    config: Config,
    on_event: Callable[[ProgressEvent], None],
    interrupt: Callable[[], None] = None,
) -> List[Path]:
    """Split every .txt/.md in ``src_dir`` into ``out_dir``; return all output paths."""
    src_dir = Path(src_dir)
    if not src_dir.exists():
        on_event(ProgressEvent(stage="split", kind="info", message=f"[WARN] Source dir not found: {src_dir}"))
        return []

    in_files = [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in (".txt", ".md")]
    if not in_files:
        on_event(ProgressEvent(stage="split", kind="info", message=f"[WARN] No .txt/.md files in {src_dir}"))
        return []

    # Fall back to char-based chunking if tiktoken isn't installed.
    max_tokens = config.max_tokens if _tokenizer_available() else None
    on_event(ProgressEvent(stage="split", kind="info", message=f"[INFO] Splitting {len(in_files)} files -> {out_dir}"))

    all_paths: List[Path] = []
    for f in sorted(in_files):
        if interrupt is not None:
            interrupt()
        try:
            saved = process_file(
                f, out_dir, max_tokens=max_tokens, max_chars=config.max_chars, tokenizer_model=config.tokenizer_model
            )
            on_event(ProgressEvent(stage="split", kind="tick", message=f"[SPLIT] {f.name} -> {len(saved)} chunks"))
            all_paths.extend(saved)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Error processing {f.name}: {e}")

    on_event(ProgressEvent(stage="split", kind="stage_done", message=f"[DONE] {len(all_paths)} chunks written"))
    return all_paths


def _tokenizer_available() -> bool:
    try:
        import tiktoken  # noqa: F401
        return True
    except Exception:
        return False
