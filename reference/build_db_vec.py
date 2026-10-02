#!/usr/bin/env python3
import os
import sys
import json
import logging
import sqlite3
from pathlib import Path
import shutil
from typing import List, Tuple

import ollama
from dotenv import load_dotenv
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

load_dotenv()

logger = logging.getLogger("ingestion")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "snowflake-arctic-embed2:568m")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
# Increased to 256 for a much tighter, more accurate 0.98 resolution grid
# Prefer the ingest-specific knob; fall back to a bare NUM_PERM so existing
# .env files that only set the generic name still take effect.
NUM_PERM = int(os.getenv("NUM_PERM_INGESTION", os.getenv("NUM_PERM", "256")))

_ollama_client = ollama.Client(host=OLLAMA_URL)


def _tokenize(text: str) -> List[str]:
    """Tokenizes text into word 3-grams (shingles) to preserve local context."""
    import re
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = [token for token in text.split() if token]

    # Create word 3-grams (shingles) to prevent bag-of-words false positives
    if len(words) >= 3:
        return [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    return words


def process_files(source_dir: str, db_path: str, batch_size: int = 16) -> None:
    """Ingest markdown files, deduplicate upfront, embed unique files in batches, and store in SQLite."""
    from retriever import load_db
    conn = load_db(db_path)
    cur = conn.cursor()

    source_path = Path(source_dir)
    embedded_dir = source_path / "embedded"
    embedded_dir.mkdir(exist_ok=True)

    txt_files = list(source_path.glob("*.md"))
    if not txt_files:
        logger.warning(f"No .md files found in {source_dir}")
        return

    logger.info(f"Found {len(txt_files)} text files to process")

    # Phase 1: Read all files into memory
    file_data: List[Tuple[Path, str]] = []
    for file_path in tqdm(txt_files, desc="Reading files"):
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            file_data.append((file_path, text))
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")

    # Phase 2: Upfront Deduplication (Fast CPU-bound operation)
    lsh = MinHashLSH(threshold=0.98, num_perm=NUM_PERM)
    unique_file_data: List[Tuple[Path, str]] = []

    for file_path, text in tqdm(file_data, desc="Deduplicating upfront"):
        tokens = _tokenize(text)
        if not tokens:
            continue

        m = MinHash(num_perm=NUM_PERM)
        for token in tokens:
            m.update(token.encode("utf-8"))

        if lsh.query(m):
            logger.debug(f"Deduplicated at ingestion: skipping '{file_path.name}'")
            # Move duplicates out immediately
            dest = embedded_dir / file_path.name
            shutil.move(str(file_path), str(dest))
            continue

        lsh.insert(file_path.name, m)
        unique_file_data.append((file_path, text))

    logger.info(f"Deduplication complete. {len(unique_file_data)} / {len(file_data)} documents are unique.")

    # Phase 3: Batch Embedding & DB Insertion (Only processing unique documents)
    processed_count = 0

    for i in tqdm(range(0, len(unique_file_data), batch_size), desc="Embedding & Inserting batches"):
        batch = unique_file_data[i:i + batch_size]
        texts = [text for _, text in batch]

        try:
            response = _ollama_client.embed(model=EMBED_MODEL, input=texts)
            embeddings = response.get("embeddings", [])
        except Exception as e:
            logger.error(f"Embedding failed for batch starting at index {i}: {e}")
            continue

        successfully_processed_files = []

        for (file_path, text), emb in zip(batch, embeddings):
            emb_json = json.dumps(emb)

            try:
                # Check if document already exists to preserve ID stability
                cur.execute("SELECT id FROM documents WHERE filename = ?", (file_path.name,))
                existing_row = cur.fetchone()

                if existing_row:
                    doc_id = existing_row[0]
                    cur.execute("UPDATE documents SET text = ? WHERE id = ?", (text, doc_id))
                    cur.execute("DELETE FROM vec_documents WHERE id = ?", (doc_id,))
                else:
                    cur.execute("INSERT INTO documents (filename, text) VALUES (?, ?)", (file_path.name, text))
                    doc_id = cur.lastrowid

                cur.execute("INSERT INTO vec_documents (id, embedding) VALUES (?, ?)", (doc_id, emb_json))
                successfully_processed_files.append(file_path)
                processed_count += 1
            except sqlite3.Error as e:
                logger.error(f"Database insertion failed for {file_path.name}: {e}")

        conn.commit()

        # Archive successfully embedded and committed files
        for file_path in successfully_processed_files:
            dest = embedded_dir / file_path.name
            shutil.move(str(file_path), str(dest))

    conn.close()
    logger.info(f"Ingestion complete. Processed {processed_count} unique documents.")


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: build_db_vec.py <source_dir> <db_path>")
        sys.exit(1)

    source_dir = sys.argv[1]
    db_path = sys.argv[2]
    process_files(source_dir, db_path)


if __name__ == "__main__":
    main()