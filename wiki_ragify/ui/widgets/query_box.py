"""Query widget — retrieve top-k relevant passages (FR-8).

:class:`QueryBox` is the Textual adapter: a footer ``Input`` plus a screen overlay that shows
whatever :func:`run_query` returns. :func:`run_query` is the pure, UI-independent retrieval
chain (open the sqlite-vec store, ANN search, relevance filter, embed the query); it returns
``(query, passages)`` and is the part covered by unit tests.

Embedding + the LLM relevance model need Ollama, so :func:`run_query` imports its heavy
dependencies lazily and lets callers handle their absence gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from textual.message import Message
from textual.widgets import Input, Static

from ...config import Config

QueryResult = Tuple[str, List[tuple[float, str, str]]]


def run_query(
    query: str,
    config: Config,
    db: Optional[Path] = None,
    top_k: Optional[int] = None,
    min_passages: Optional[int] = None,
    embed_client=None,
    chat_client=None,
) -> QueryResult:
    """Run a retrieval against a knowledge base and return ``(query, passages)``.

    This is the pure, UI-independent retrieval pipeline: open the store, ANN search, relevance
    filter, then embed the query. ``top_k`` (ANN top-k) and ``min_passages`` (relevance floor)
    are optional so the caller can let the config's defaults apply.
    """
    from ...retrieval import filter_relevant_passages, load_db, search_top_k
    from ...llm.embed import EmbedClient

    if not query.strip():
        return (query, [])

    db_path = Path(db) if db is not None else config.db_path("")
    conn = load_db(db_path, config)
    if embed_client is None:
        embed_client = EmbedClient(config)
    emb = embed_client.embed_query(query)

    top_k = config.top_k if top_k is None else top_k
    min_passages = config.min_passages if min_passages is None else min_passages

    passages = search_top_k(conn, emb, top_k, config=config)
    passages = filter_relevant_passages(
        query, passages, config, min_passages=min_passages, chat_client=chat_client
    )
    return (query, passages)


@dataclass
class QueryRow:
    """One rendered result row for the overlay table."""

    rank: int
    score: float
    filename: str
    text: str

    def formatted(self) -> str:
        snippet = self.text.replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:297] + "..."
        return f"{self.rank}. {self.score:.3f}  {self.filename}\n{snippet}"


def build_rows(query: str, passages: List[tuple[float, str, str]]) -> list[QueryRow]:
    """Turn ``(query, passages)`` into ordered display rows (pure)."""
    return [
        QueryRow(rank=i, score=score, filename=filename, text=text)
        for i, (score, filename, text) in enumerate(passages, start=1)
    ]


class Submitted(Message):
    """Textual message carrying the query string on Enter."""

    def __init__(self, query: str) -> None:
        super().__init__()
        self.query = query


class QueryBox(Static):
    """A footer input that posts :class:`Submitted` on Enter.

    The app subscribes to :class:`Submitted` and runs :func:`run_query` off the main
    thread, then renders the results in an overlay. Composes a single :class:`Input`
    (FR-8): the widget's job is to collect the query string — retrieval runs elsewhere.
    """

    CSS_PATH = "styles.tcss"

    def __init__(self) -> None:
        super().__init__(id="query-input")

    def compose(self) -> "ComposeResult":
        yield Input(placeholder="Query the knowledge base…")

    def on_input_submitted(self, message: Input.Submitted) -> None:  # pragma: no cover
        self.post_message(Submitted(message.value))

    async def on_mount(self) -> None:  # pragma: no cover - Textual lifecycle
        self.query_one(Input).focus()
