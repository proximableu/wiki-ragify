"""Knowledge-base builder — read, deduplicate, embed, and upsert (Phase 3 of the
old ``build_db_vec.py``). This is the optional "step 6": it turns the gated
``*.md`` artifacts into the sqlite-vec vector store.

Faithful port of ``reference/build_db_vec.py::process_files``:

* Phase 1 — read every ``*.md`` into memory.
* Phase 2 — upfront MinHash LSH dedup (threshold 0.98); duplicates are moved out
  to an ``embedded/`` side-dir immediately, matching the reference.
* Phase 3 — embed unique files in batches and upsert into the DB. The upsert
  preserves ID stability: if the filename already exists, the text is updated and
  the old row re-inserted; otherwise a fresh row is created. Committed files are
  archived to ``embedded/`` afterward.

The one deliberate change: embeddings come through an injectable
:class:`~wiki_ragify.llm.embed.EmbedClient` (defaulting to a live one built from
``config``) instead of a module-global ``ollama.Client``, so ingestion can be
driven and tested without a running Ollama server.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Callable

from ..config import Config
from ..llm.embed import EmbedClient
from ..pipeline.events import ProgressEvent
from .dedup import index_documents

logger = logging.getLogger(__name__)


def build_db(
    source_dir: Path,
    db_path: Path,
    config: Config,
    embed_client: EmbedClient | None = None,
    on_event: Callable[[ProgressEvent], None] | None = None,
) -> None:
    """Ingest ``*.md`` files from ``source_dir`` into the sqlite-vec store at ``db_path``.

    Reads, deduplicates, embeds in batches of ``config.embed_batch_size``, and
    upserts into ``documents`` / ``vec_documents``. No-op (with a warning) when the
    source directory has no ``*.md`` files — matching the reference.

    ``db_path`` defaults to ``config.knowledge_dir/knowledge.db``.
    """
    from ..retrieval.store import load_db

    if embed_client is None:
        embed_client = EmbedClient(config)

    source_dir = Path(source_dir)
    db_path = Path(db_path) if db_path is not None else config.knowledge_dir / "knowledge.db"

    md_files = sorted(source_dir.glob("*.md"))
    if not md_files:
        logger.warning(f"No *.md files found in {source_dir}")
        if on_event:
            on_event(ProgressEvent(stage="ingest", kind="info", message=f"No *.md files in {source_dir}"))
        return

    logger.info(f"Found {len(md_files)} .md file(s) to ingest")
    if on_event:
        on_event(ProgressEvent(stage="ingest", total=len(md_files), kind="info", message="Reading files"))

    # Phase 1: read all files into memory.
    file_data = []  # (Path, text)
    for file_path in md_files:
        try:
            file_data.append((file_path, file_path.read_text(encoding="utf-8", errors="ignore")))
        except OSError as e:
            logger.error(f"Error reading {file_path}: {e}")

    # Phase 2: upfront dedup (0.98). Duplicates are reported and moved aside.
    embedded_dir = source_dir / "embedded"
    embedded_dir.mkdir(exist_ok=True)

    def _report_dedup(ident: str) -> None:
        logger.debug(f"Deduplicated at ingestion: skipping {ident}")

    kept = index_documents(
        [(str(p), text) for p, text in file_data],
        on_dedup=_report_dedup,
    )

    # Re-attach the original Path objects to the kept (identifier, text) pairs.
    path_by_name = {p.name: p for p in file_data}
    unique_file_data = [(path_by_name[ident], text) for ident, text in kept]
    logger.info(
        f"Deduplication complete. {len(unique_file_data)} / {len(file_data)} documents are unique."
    )
    if on_event:
        on_event(
            ProgressEvent(
                stage="ingest",
                total=len(file_data),
                index=len(unique_file_data),
                kind="tick",
                message=f"Dedup: {len(unique_file_data)}/{len(file_data)} unique",
            )
        )

    conn = load_db(db_path, config)
    cur = conn.cursor()

    # Phase 3: batch embed + upsert.
    batch_size = config.embed_batch_size
    processed_count = 0

    for start in range(0, len(unique_file_data), batch_size):
        batch = unique_file_data[start : start + batch_size]
        texts = [text for _, text in batch]

        try:
            embeddings = embed_client.embed_batch(texts)
        except Exception as e:  # embed_batch raises RuntimeError on unrecoverable failure
            logger.error(f"Embedding failed for batch starting at index {start}: {e}")
            continue

        successfully_processed_files = []
        for (file_path, text), emb in zip(batch, embeddings):
            emb_json = json.dumps(emb)
            try:
                # Preserve ID stability: an existing filename is updated in place
                # rather than creating a new row.
                cur.execute("SELECT id FROM documents WHERE filename = ?", (file_path.name,))
                existing_row = cur.fetchone()
                if existing_row:
                    doc_id = existing_row[0]
                    cur.execute("UPDATE documents SET text = ? WHERE id = ?", (text, doc_id))
                    cur.execute("DELETE FROM vec_documents WHERE id = ?", (doc_id,))
                else:
                    cur.execute(
                        "INSERT INTO documents (filename, text) VALUES (?, ?)", (file_path.name, text)
                    )
                    doc_id = cur.lastrowid

                cur.execute(
                    "INSERT INTO vec_documents (id, embedding) VALUES (?, ?)", (doc_id, emb_json)
                )
                successfully_processed_files.append(file_path)
                processed_count += 1
            except Exception as e:  # sqlite3.Error and vec0-specific errors alike
                logger.error(f"Database insertion failed for {file_path.name}: {e}")

        conn.commit()

        # Archive successfully embedded and committed files.
        for file_path in successfully_processed_files:
            shutil.move(str(file_path), str(embedded_dir / file_path.name))

        if on_event:
            on_event(
                ProgressEvent(
                    stage="ingest",
                    total=len(unique_file_data),
                    index=min(start + batch_size, len(unique_file_data)),
                    kind="tick",
                    message=f"Embedding batch {start // batch_size + 1}",
                )
            )

    conn.close()
    logger.info(f"Ingestion complete. Processed {processed_count} unique documents.")
    if on_event:
        on_event(ProgressEvent(stage="ingest", kind="stage_done", message=f"Ingested {processed_count} documents"))
