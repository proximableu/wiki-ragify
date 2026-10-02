"""Stage 3: section-aware text splitting into Markdown chunks."""

from .splitter import (
    find_sections,
    chunk_section,
    process_file,
    clean_filename,
    save_unique_file,
    remove_trailing_sections,
    is_potential_bare_header,
)
from .runner import split_dir

__all__ = [
    "find_sections",
    "chunk_section",
    "process_file",
    "clean_filename",
    "save_unique_file",
    "remove_trailing_sections",
    "is_potential_bare_header",
    "split_dir",
]
