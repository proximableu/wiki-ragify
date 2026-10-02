"""Article gating — the chunk-level accept/reject filter (Gate 2).

``chunk.py`` holds the faithfully ported chunk gate (ported from
``reference/chunk_gate.py``). It reads split ``*.md`` artifacts, decides accept
vs. reject, and moves each into an ``accept/`` or ``reject/`` subdirectory.
"""

from .chunk import gate_files, process_article, move_article, read_full_content, has_insufficient_sentences

__all__ = [
    "gate_files",
    "process_article",
    "move_article",
    "read_full_content",
    "has_insufficient_sentences",
]
