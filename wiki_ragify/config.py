"""Centralized configuration — the single place every tunable lives.

All values come from ``.env`` (via ``python-dotenv``), falling back to the defaults below.
There is deliberately no hardcoded infrastructure anywhere else in the package: the UI,
the crawler, the gate, the embed client, and the retriever all read from one ``Config``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

# Load .env once at package import. load_dotenv() is idempotent — it won't clobber any
# variable already present in the real environment.
load_dotenv()


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw if raw and raw.strip() else default


@dataclass(frozen=True)
class Config:
    """Every tunable the pipeline needs, resolved from ``.env``.

    A frozen dataclass so a loaded config can be passed freely to modules without
    risking silent mutation of shared settings mid-run.
    """

    # --- Ollama connection ---
    ollama_url: str = field(default_factory=lambda: _get_str("OLLAMA_URL", "http://localhost:11434"))

    # --- Gating model (article gate 1 + chunk gate 2) ---
    gate_model: str = field(default_factory=lambda: _get_str("GATE_MODEL_NAME", "qwen3-30b-dataset"))

    # --- Embedding model ---
    embed_model: str = field(default_factory=lambda: _get_str("EMBED_MODEL", "snowflake-arctic-embed2:568m"))
    embed_dim: int = field(default_factory=lambda: _get_int("EMBED_DIM", 1024))

    # --- LLM options ---
    gate_temperature: float = field(default_factory=lambda: _get_float("TEMPERATURE", 0.0))
    num_ctx: int = field(default_factory=lambda: _get_int("NUM_CTX", 8192))
    gate_timeout: float = field(default_factory=lambda: _get_float("GATE_TIMEOUT", 60.0))

    # --- Lightweight LLM for relevance filtering at retrieval time ---
    lightweight_model: str = field(default_factory=lambda: _get_str("LIGHTWEIGHT_LLM_MODEL", "llama3.2:3b"))

    # --- Wikipedia crawler ---
    wiki_lang: str = field(default_factory=lambda: _get_str("WIKI_LANG", "en"))
    wiki_user_agent: str = field(
        default_factory=lambda: _get_str(
            "WIKI_USER_AGENT", "WikiLLMGateCrawler/1.0 (mailto:proximableu@gmail.com)"
        )
    )
    min_text_length: int = field(default_factory=lambda: _get_int("MIN_TEXT_LENGTH", 1500))
    max_title_length: int = field(default_factory=lambda: _get_int("MAX_TITLE_LENGTH", 200))
    rate_limit_delay: float = field(default_factory=lambda: _get_float("RATE_LIMIT_DELAY", 0.35))
    request_timeout: int = field(default_factory=lambda: _get_int("REQUEST_TIMEOUT", 30))

    # --- Splitter ---
    max_tokens: int = field(default_factory=lambda: _get_int("SPLITTER_MAX_TOKENS", 2048))
    max_chars: int = field(default_factory=lambda: _get_int("SPLITTER_MAX_CHARS", 3000))
    tokenizer_model: str = field(default_factory=lambda: _get_str("SPLITTER_TOKENIZER_MODEL", "gpt-4o-mini"))

    # --- Chunk gate ---
    gate_max_words: int = field(default_factory=lambda: _get_int("GATE_MAX_WORDS", 2048))

    # --- Deduplication ---
    dedup_jaccard_threshold: float = field(default_factory=lambda: _get_float("DEDUP_JACCARD_THRESHOLD", 0.65))
    num_perm_ingestion: int = field(
        default_factory=lambda: _get_int("NUM_PERM_INGESTION", _get_int("NUM_PERM", 256))
    )
    num_perm_retrieval: int = field(
        default_factory=lambda: _get_int("NUM_PERM_RETRIEVAL", _get_int("NUM_PERM", 128))
    )

    # --- SQLite-vec HNSW params (retrieval) ---
    hnsw_m: int = field(default_factory=lambda: _get_int("HNSW_M", 32))
    hnsw_ef_construction: int = field(default_factory=lambda: _get_int("HNSW_EF_CONSTRUCTION", 200))
    hnsw_ef_search: int = field(default_factory=lambda: _get_int("HNSW_EF_SEARCH", 100))

    # --- Pipeline behaviour knobs (v2 additions) ---
    embed_batch_size: int = field(default_factory=lambda: _get_int("MAX_BATCH", 16))
    top_k: int = field(default_factory=lambda: _get_int("TOP_K", 16))
    min_passages: int = field(default_factory=lambda: _get_int("MIN_PASSEGES", 16))

    # --- Directory layout names (kept in sync with the old orchestrator) ---
    output_dirname: str = field(default_factory=lambda: "output")
    chunks_dirname: str = field(default_factory=lambda: "chunks")
    gate_accept_dirname: str = field(default_factory=lambda: "accept")
    knowledge_dirname: str = field(default_factory=lambda: "knowledge")

    # --- Paths that depend on the chosen output root ---
    output_dir: Path = field(default=None, compare=False)  # set by the caller / TUI

    def with_output_dir(self, output_dir: "os.PathLike[str] | str") -> "Config":
        """Return a new Config with ``output_dir`` bound (frozen dataclass => copy)."""
        return replace(self, output_dir=Path(output_dir))
