#!/usr/bin/env python3
"""
Gate-filter articles using an LLM via Ollama API (Structured Outputs Version).

This script reads articles, extracts their full body content,
and uses the centralized Ollama gateway to classify each article
strictly into either 'accept' or 'reject' directory.
"""

import argparse
import logging
import os
import re
import shutil
import time
from pathlib import Path

import nltk
from ollama_gateway import PipelineConfig, LLMEvaluator

class Color:
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Cap on words read from a single article before gating. Kept configurable so
# a longer gate window can be requested without code edits; the value only bounds
# the *prompt* sent to the gate, not what is archived.
GATE_MAX_WORDS = int(os.getenv("GATE_MAX_WORDS", "2048"))

def read_full_content(path: Path) -> str:
    """Reads all file lines without truncation, filtering out disruptive artifacts."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise IOError(f"Failed to read {path}: {e}")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    valid_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("{{") or stripped.startswith("{|") or stripped.endswith("|}"):
            continue
        if stripped.startswith("[[File:") or stripped.startswith("[[Image:"):
            continue
        if stripped.startswith("="):
            continue
        valid_lines.append(stripped)

    full_clean_text = " ".join(valid_lines) if valid_lines else " ".join(text.split())
    full_clean_text = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]]+)\]\]', r'\1', full_clean_text)
    words = full_clean_text.split()
    if len(words) > GATE_MAX_WORDS:
        logger.warning(
            f"{path.name}: article truncated from {len(words)} to {GATE_MAX_WORDS} words for gating "
            f"(see GATE_MAX_WORDS); the full article is still archived."
        )
    return " ".join(words[:GATE_MAX_WORDS])

def has_insufficient_sentences(path: Path) -> bool:
    """Returns True if an .md file has fewer than 2 sentences, ignoring markdown headers."""
    if path.suffix.lower() != '.md':
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False

    clean_lines = [line for line in text.splitlines() if not line.strip().startswith("#")]
    clean_text = " ".join(clean_lines).strip()

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    return len(nltk.sent_tokenize(clean_text)) < 2

def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def process_article(article_path: Path, gating_prompt: str, evaluator: LLMEvaluator) -> str:
    """
    Process one article: read full content window, evaluate via centralized gateway.
    Returns 'accept' or 'reject'.
    """
    if "_introduction" in article_path.name:
        return "accept"

    if has_insufficient_sentences(article_path):
        return "reject"

    try:
        text = read_full_content(article_path)
    except IOError as e:
        logger.warning(f"Skipping {article_path.name}: {e}")
        return "reject"

    # Delegate to gateway; native schema enforcement removes manual prompt injection
    accepted = evaluator.evaluate(text, gating_prompt)
    return "accept" if accepted else "reject"

def move_article(article_path: Path, decision: str, source_dir: Path) -> Path:
    dest_dir = source_dir / decision
    ensure_directory(dest_dir)
    dest_path = dest_dir / article_path.name
    shutil.move(str(article_path), str(dest_path))
    return dest_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate-filter articles strictly into accept/reject folders using an LLM via Ollama."
    )
    parser.add_argument("base_dir", type=Path, help="Directory containing the gating prompt file.")
    parser.add_argument("source_dir", type=Path, help="Directory containing articles to process.")
    parser.add_argument("--prompt-file", default="gating_prompt.txt", help="Name of the gating prompt file.")
    parser.add_argument("--dry-run", action="store_true", help="Do not move files; just simulate processing.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    prompt_path = args.base_dir / args.prompt_file
    if not prompt_path.exists():
        raise FileNotFoundError(f"Gating prompt not found: {prompt_path}")

    gating_prompt = prompt_path.read_text(encoding="utf-8")
    articles = sorted(list(args.source_dir.glob("*.txt")) + list(args.source_dir.glob("*.md")))
    total = len(articles)
    logger.info(f"Found {total} articles to process.")

    # Initialize centralized configuration and evaluator
    pipeline_config = PipelineConfig()
    evaluator = LLMEvaluator(pipeline_config)
    start_time = time.time()

    try:
        for idx, article_path in enumerate(articles, start=1):
            decision = process_article(article_path, gating_prompt, evaluator)
            tag = f"{Color.CYAN}ACCEPT{Color.RESET}" if decision == "accept" else f"{Color.MAGENTA}REJECT{Color.RESET}"

            elapsed = time.time() - start_time
            if elapsed > 0:
                items_per_sec = idx / elapsed
                eta_seconds = (total - idx) / items_per_sec
                eta_str = f"{int(eta_seconds)}s" if eta_seconds < 60 else f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
            else:
                eta_str = "--"

            logger.info(f"[{idx}/{total}] {tag} (ETA: {eta_str}) -> {article_path.name}")

            if not args.dry_run:
                move_article(article_path, decision, args.source_dir)
    finally:
        evaluator.close()

if __name__ == "__main__":
    main()