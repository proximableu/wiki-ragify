#!/usr/bin/env python3
import os
import json
import logging
import sqlite3
import re
from typing import List, Tuple, Optional
import sqlite_vec
import sys

import numpy as np
import ollama
from dotenv import load_dotenv
from datasketch import MinHash, MinHashLSH
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger("retriever")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "snowflake-arctic-embed2:568m")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
DEDUP_JACCARD_THRESHOLD = float(os.getenv("DEDUP_JACCARD_THRESHOLD", "0.65"))
# Prefer the retrieval-specific knob; fall back to a bare NUM_PERM so existing
# .env files that only set the generic name still take effect.
NUM_PERM = int(os.getenv("NUM_PERM_RETRIEVAL", os.getenv("NUM_PERM", "128")))
HNSW_M = int(os.getenv("HNSW_M", "32"))
HNSW_EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", "200"))
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "100"))
LIGHTWEIGHT_LLM_MODEL = os.getenv("LIGHTWEIGHT_LLM_MODEL", "llama3.2:3b")



_ollama_client = ollama.Client(host=OLLAMA_URL)

class RelevanceFilter(BaseModel):
    relevant_indices: list[int] = Field(description="Indices of passages directly relevant to the query")

def load_db(db_path: str) -> sqlite3.Connection:
    """Initialize SQLite connection, load sqlite-vec, and ensure schema exists."""
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    try:
        import sqlite_vec
        sqlite_vec.load(conn)
    except sqlite3.OperationalError as e:
        logger.error(f"Failed to load sqlite_vec extension: {e}")
        raise

    conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE,
                text TEXT
            );
        """)

    conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_documents USING vec0(
                id INTEGER PRIMARY KEY,
                embedding float[{EMBED_DIM}] distance_metric=cosine
            );
        """)

    logger.info(f"Database loaded: {db_path}")
    return conn


def embed_query(text: str) -> List[float]:
    """Generate embedding vector for a query string using Ollama."""
    try:
        response = _ollama_client.embed(model=EMBED_MODEL, input=text)
        embeddings = response.get("embeddings", [])
        if not embeddings or not isinstance(embeddings[0], list):
            raise ValueError("Invalid embedding structure from Ollama")
        return [float(x) for x in embeddings[0]]
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise


def _tokenize(text: str) -> List[str]:
    """Normalize and tokenize text for MinHash computation."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [token for token in text.split() if token]


def search_top_k(
        conn: sqlite3.Connection,
        query_emb: List[float],
        top_k: int,
        preliminary_top_n: int = 600,
) -> List[Tuple[float, str, str]]:
    """Execute ANN search, join metadata, and apply retrieval-time deduplication."""
    if len(query_emb) != EMBED_DIM:
        raise ValueError(f"Dimension mismatch: expected {EMBED_DIM}, got {len(query_emb)}")

    cur = conn.cursor()
    try:
        # sqlite-vec requires MATCH ? for the vector and AND k = ? for the neighbor count.
        # Removing LIMIT prevents planner conflicts during JOIN operations.
        cur.execute(
            """
            SELECT v.id, v.distance, d.filename, d.text 
            FROM vec_documents v
            JOIN documents d ON v.id = d.id
            WHERE v.embedding MATCH ? AND v.k = ?
            ORDER BY v.distance
            """,
            (json.dumps(query_emb), preliminary_top_n)
        )
        rows = cur.fetchall()
    except sqlite3.Error as e:
        logger.error(f"ANN query failed: {e}")
        raise

    if not rows:
        return []

    logger.info(f"Retrieved {len(rows)} candidates from ANN index")

    # Retrieval-time deduplication using MinHashLSH
    lsh = MinHashLSH(threshold=DEDUP_JACCARD_THRESHOLD, num_perm=NUM_PERM)
    deduped: List[Tuple[float, str, str]] = []

    for doc_id, distance, filename, text in rows:

        tokens = _tokenize(text)
        if not tokens:
            continue

        m = MinHash(num_perm=NUM_PERM)
        for token in tokens:
            m.update(token.encode("utf-8"))

        if lsh.query(m):
            logger.debug(f"Deduplicated (MinHash): skipping '{filename}'")
            continue

        lsh.insert(f"{doc_id}_{distance:.4f}", m)
        # Store negative distance as proxy for similarity for sorting purposes
        deduped.append((-distance, filename, text))

        if len(deduped) >= top_k:
            break

    logger.info(f"Final deduplicated top-{len(deduped)} passages returned")
    return deduped

def deduplicate_passages(
        passages: List[Tuple[float, str, str]],
        top_k: int
) -> List[Tuple[float, str, str]]:
    """
    Unified deduplication for merged candidate pools.
    Applies MinHashLSH to ensure diversity across multiple retrieval queries.
    """
    if not passages:
        return []

    # Sort by similarity score to prioritize high-relevance passages
    sorted_passages = sorted(passages, key=lambda x: x[0], reverse=True)

    lsh = MinHashLSH(threshold=DEDUP_JACCARD_THRESHOLD, num_perm=NUM_PERM)
    unique_passages: List[Tuple[float, str, str]] = []

    for score, filename, text in sorted_passages:
        tokens = _tokenize(text)
        if not tokens:
            continue

        m = MinHash(num_perm=NUM_PERM)
        for token in tokens:
            m.update(token.encode("utf-8"))

        # Skip if similar to already selected passages
        if lsh.query(m):
            continue

        lsh.insert(f"{filename}_{score:.4f}", m)
        unique_passages.append((score, filename, text))

        if len(unique_passages) >= top_k:
            break

    logger.info(f"Unified deduplication: {len(passages)} -> {len(unique_passages)} passages")
    return unique_passages


def filter_relevant_passages(query: str, passages: List[Tuple[float, str, str]], min_passages: int = 16) -> List[Tuple[float, str, str]]:
    """Trim irrelevant passages using structured LLM outputs with deterministic fallback."""
    if not passages:
        return []

    # Sort by similarity score to ensure fallback returns highest-confidence matches
    sorted_passages = sorted(passages, key=lambda x: x[0], reverse=True)
    # FIX: Truncate snippets to 500 chars to reduce token load on the lightweight routing model
    snippets = [f"[{i}] {text[:500].replace(chr(10), ' ')}" for i, (_, _, text) in enumerate(sorted_passages)]

    prompt = (
        f"You are an inclusive RAG context retrieval assistant evaluating passages for the query: \"{query}\"\n\n"
        f"INCLUSION CRITERIA:\n"
        f"- Keep any passage that directly answers, provides useful background context, explores adjacent concepts, or offers alternative viewpoints/debunking details related to the query.\n"
        f"- Err on the side of caution: if a passage is even partially relevant or helps build a complete picture, include it.\n\n"
        f"EXCLUSION CRITERIA:\n"
        f"- Only exclude a passage if it is completely off-topic, entirely corrupted, or shares zero conceptual overlap with the query.\n\n"
        f"Passages to evaluate:\n{chr(10).join(snippets)}\n\n"
        f"Output JSON only."
    )

    try:
        response = _ollama_client.chat(
            model=LIGHTWEIGHT_LLM_MODEL,
            messages=[
                # Inject a system instruction establishing the native reasoning environment
                {"role": "system", "content": "You are a precise routing filter. Reasoning effort: low."},
                {"role": "user", "content": prompt}
            ],
            format=RelevanceFilter.model_json_schema(),
            options={
                "temperature": 0.0,
                "seed": 42  # Guarantees identical choices on identical runs
            }
        )
        data = RelevanceFilter.model_validate_json(response["message"]["content"])
        valid_indices = set(data.relevant_indices)

        filtered = [p for i, p in enumerate(sorted_passages) if i in valid_indices]

        # FIX: Safety net to guarantee minimum context volume (min_passages)
        # If LLM filters too aggressively, append highest-scoring remaining passages
        if len(filtered) < min_passages:
            logger.warning(f"Relevance filter returned {len(filtered)} passages. Padding to {min_passages}.")
            remaining = [p for i, p in enumerate(sorted_passages) if i not in valid_indices]
            filtered.extend(remaining[:min_passages - len(filtered)])

        logger.info(f"Relevance filter: {len(passages)} -> {len(filtered)} passages")
        return filtered
    except Exception as e:
        logger.warning(f"Relevance filtering failed ({e}), bypassing filter")
        return sorted_passages[:min_passages]