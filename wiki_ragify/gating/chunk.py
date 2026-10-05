"""Chunk gate — the second LLM gate in the pipeline (Gate 2 in the reference).

This is a faithful in-process port of ``reference/chunk_gate.py``. Unlike the
article-level gate (which runs inline in the crawler, Gate 1), the chunk gate
operates on split article artifacts on disk (``*.txt`` / ``*.md``), reading each
one, deciding accept/reject, and *moving* the file into an ``accept/`` or
``reject/`` subdirectory of the source directory.

Faithful notes / deliberate divergence from the reference:
* The reference gates ``*.txt`` AND ``*.md`` files. In this package only
  ``*.md`` files are ever produced by the splitter, so we gate ``*.md``.
* All other behaviour (content stripping, insufficient-sentence fast reject,
  the ``_introduction`` accept fast-path, word truncation, move-on-decision) is
  ported verbatim.
* The reference hard-reads the prompt from ``base_dir/prompt_file``; this
  package takes the prompt as a parameter (it lives elsewhere).
* A small ``gate_files`` runner adds event emission for the UI.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Callable

from ..config import Config
from ..llm.gateway import LLMEvaluator
from ..pipeline.events import ProgressEvent

logger = logging.getLogger(__name__)

GATE_MAX_WORDS = 2048
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def read_full_content(path: Path) -> str:
    """Read an article artifact, stripping boilerplate and truncating to words.

    Mirrors ``reference/chunk_gate.py::read_full_content``: discards lines that
    look like template/section markers, removes ``[[...]]`` links, then keeps the
    first ``GATE_MAX_WORDS`` words.
    """
    keep_lines = ["{{", "{|", "|}", "[[File:", "[[Image:", "="]
    lines = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if any(line.lstrip().startswith(marker) for marker in keep_lines):
            continue
        if stripped and not stripped.startswith(("#", "*", "|")):
            lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\[\[.*?\]\]", "", text)
    words = _WORD_RE.findall(text)
    if len(words) > GATE_MAX_WORDS:
        words = words[:GATE_MAX_WORDS]
    return " ".join(words)


def has_insufficient_sentences(path: Path) -> bool:
    """Return True when a ``.md`` artifact is too short to gate.

    Faithful port of the reference: strip leading ``#`` header lines and use
    ``nltk.sent_tokenize`` — articles with fewer than two sentences are skipped
    (rejected) so the LLM is not wasted on noise.
    """
    try:
        from nltk import sent_tokenize
    except ModuleNotFoundError:
        raise RuntimeError(
            "nltk is required for sentence tokenization in the chunk gate; "
            "install it with `pip install nltk` (also `nltk download punkt`)."
        ) from None

    if path.suffix != ".md":
        return False

    lines = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line]
    # Drop markdown header lines (# ...).
    content_lines = [line for line in lines if not line.startswith("#")]
    if not content_lines:
        return True

    try:
        sentences = sent_tokenize(" ".join(content_lines))
    except Exception as e:
        logger.warning(f"NLTK sentence tokenization failed for {path.name}: {e}")
        return False

    return len(sentences) < 2


def process_article(
    path: Path,
    gating_prompt: str,
    evaluator: LLMEvaluator,
    model: str = "",
    num_ctx: int = 0,
    think: bool = False,
) -> bool:
    """Gate a single article artifact. Returns True if accepted.

    Faithful port of ``reference/chunk_gate.py::process_article``:
    * ``_introduction`` in the filename → fast accept.
    * insufficient sentences → fast reject.
    * otherwise, read the full (truncated) content and ask the gate model.
    """
    if "_introduction" in path.name:
        logger.debug(f"Accept (introduction fast-path): {path.name}")
        return True

    if has_insufficient_sentences(path):
        logger.debug(f"Reject (insufficient sentences): {path.name}")
        return False

    text = read_full_content(path)
    if not text:
        return False

    # ``LLMEvaluator.evaluate`` re-assembles the <ARTICLE> block, applies retry +
    # backoff, and returns a boolean on its own. The ``gate_files`` caller only
    # needs that; the old ``Client.chat`` signature is dropped.
    return evaluator.evaluate(text=text, gating_prompt=gating_prompt)


def move_article(path: Path, decision: bool, source_dir: Path) -> None:
    """Move an article artifact into ``accept/`` or ``reject/`` under source_dir."""
    dest_dir = source_dir / ("accept" if decision else "reject")
    dest_dir.mkdir(exist_ok=True)
    shutil.move(str(path), str(dest_dir / path.name))


def gate_files(
    source_dir: Path,
    gating_prompt: str,
    config: Config,
    evaluator: Client,
    on_event: Callable[[ProgressEvent], None] | None = None,
    interrupt: Callable[[], None] = None,
) -> None:
    """Gate every ``*.md`` artifact in ``source_dir``, moving accept/reject.

    Event-emitting runner for the chunk gate (Gate 2).
    """
    files = sorted(source_dir.glob("*.md"))
    logger.info(f"Chunk-gating {len(files)} file(s) from {source_dir.name}")

    for i, path in enumerate(files, 1):
        if interrupt is not None:
            interrupt()
        decision = process_article(path, gating_prompt, evaluator)
        move_article(path, decision, source_dir)
        kind = "accept" if decision else "reject"
        msg = f"[{kind.upper()}] chunk {i}/{len(files)}: {path.name}"
        if on_event:
            on_event(ProgressEvent(stage="gate", index=i, total=len(files), kind=kind, message=msg))
