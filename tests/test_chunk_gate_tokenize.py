"""Unit tests for the chunk gate's fast-path helpers in ``gating.chunk``.

``has_insufficient_sentences`` gates short artifacts on ``nltk.sent_tokenize`` before
the LLM is ever called, so a missing ``nltk`` must fail loudly here (with a pip hint)
rather than blowing up deep in the pipeline.
"""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from wiki_ragify.gating import chunk as chunk_module


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_single_sentence_is_insufficient(tmp_path: Path) -> None:
    p = _write(tmp_path, "one.md", "a single sentence here")
    assert chunk_module.has_insufficient_sentences(p) is True


def test_multi_sentence_is_sufficient(tmp_path: Path) -> None:
    p = _write(tmp_path, "many.md", "First sentence. Second sentence. Third one.")
    assert chunk_module.has_insufficient_sentences(p) is False


def test_non_md_is_never_insufficient(tmp_path: Path) -> None:
    p = _write(tmp_path, "raw.txt", "one sentence.")
    assert chunk_module.has_insufficient_sentences(p) is False


def test_header_lines_do_not_count_as_sentences(tmp_path: Path) -> None:
    # Only the non-header content counts; a lone header means no content at all.
    p = _write(tmp_path, "headeronly.md", "# Heading\n## Subhead")
    assert chunk_module.has_insufficient_sentences(p) is True


def test_missing_nltk_raises_runtime_not_module_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate nltk being uninstalled: monkeypatch the module's real import so the
    # guard inside ``has_insufficient_sentences`` sees the ImportError and turns it
    # into a clear RuntimeError with a pip hint.
    real_import = builtins.__import__

    def _blocking_import(name: str, *args, **kwargs):
        if name == "nltk" or name.startswith("nltk."):
            raise ModuleNotFoundError("No module named 'nltk'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)

    p = _write(tmp_path, "x.md", "a sentence here")
    with pytest.raises(RuntimeError) as excinfo:
        chunk_module.has_insufficient_sentences(p)
    assert "nltk" in str(excinfo.value)
    assert "pip install nltk" in str(excinfo.value)
