"""Vector store — a sqlite-vec-backed knowledge base.

Faithful port of ``load_db`` from ``reference/retriever.py``: opens a SQLite
connection, loads the ``sqlite_vec`` extension, and ensures the schema exists.
The ``(documents, vec_documents)`` table pair and the HNSW parameters are the
same constants the old ``build_db_vec.py`` / ``retriever.py`` used.
"""

from __future__ import annotations

import logging
import os
import sqlite3

from ..config import Config

logger = logging.getLogger(__name__)


def load_db(db_path: str | None = None, config: Config | None = None) -> sqlite3.Connection:
    """Open the knowledge-base database, loading ``sqlite_vec`` and ensuring schema.

    Accepts either a path or a :class:`~wiki_ragify.config.Config` (used to derive
    the db path and embedding dimension). Mirrors ``reference/retriever.py::load_db``.
    """
    from sqlite_vec import load as _load_vec  # imported lazily: only needed at runtime

    if db_path is None and config is not None:
        # Default: the knowledge db lives at the output root, named from the seed
        # page. This mirrors where ``build_db`` writes, so a run that builds a db at
        # output/Autistic-supremacism.db can always reopen the same file on read.
        db_path = config.db_path("")

    db_path = str(db_path)
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    try:
        _load_vec(conn)
    except sqlite3.OperationalError as e:
        logger.error(f"Failed to load sqlite_vec extension: {e}")
        raise

    dim = config.embed_dim if config is not None else 1024
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            text TEXT
        );
        """
    )
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_documents USING vec0(
            id INTEGER PRIMARY KEY,
            embedding float[{dim}] distance_metric=cosine
        );
        """
    )
    logger.info(f"Database loaded: {db_path}")
    return conn
