"""ANN search over the vector store.

Faithful port of ``search_top_k`` from ``reference/retriever.py``: runs the
sqlite-vec nearest-neighbour query with ``MATCH ? AND k = ?`` (no LIMIT, which
the old code found conflicts with the JOIN), then de-duplicates the candidates
on CPU with a MinHash LSH (0.65 threshold) before returning the top-k passages.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from datasketch import MinHash, MinHashLSH

from ..config import Config
from ..ingestion.dedup import RETRIEVAL, deduplicate_passages

logger = logging.getLogger(__name__)

# sqlite-vec returns a wide net first; dedup collapses it down to the real top-k.
_PRELIMINARY_TOP_N = 600


def search_top_k(
    conn,
    query_emb: List[float],
    top_k: int,
    preliminary_top_n: int = _PRELIMINARY_TOP_N,
    dedup: RETRIEVAL = RETRIEVAL,
    config: Optional[Config] = None,
) -> List[tuple[float, str, str]]:
    """ANN search over ``vec_documents``, de-duplicated to the top-k passages.

    Returns ``(similarity, filename, text)`` tuples sorted by descending
    similarity, each deduplicated against the others via MinHash LSH. A
    ``dimension mismatch`` raises ValueError, matching the reference.
    """
    dim = config.embed_dim if config is not None else len(query_emb)
    if len(query_emb) != dim:
        raise ValueError(f"Dimension mismatch: expected {dim}, got {len(query_emb)}")

    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT v.id, v.distance, d.filename, d.text
            FROM vec_documents v
            JOIN documents d ON v.id = d.id
            WHERE v.embedding MATCH ? AND v.k = ?
            ORDER BY v.distance
            """,
            (json.dumps(query_emb), preliminary_top_n),
        )
        rows = cur.fetchall()
    except Exception as e:  # sqlite3.Error and vec0-specific errors alike
        logger.error(f"ANN query failed: {e}")
        raise

    if not rows:
        return []

    # sqlite-vec gives a wide net; dedup collapses it down to the real top-k.
    passages = [(-distance, filename, text) for _, distance, filename, text in rows]
    deduped = deduplicate_passages(
        passages,
        dedup=dedup,
        limit=top_k,
        on_dedup=lambda ident: logger.debug(f"Deduplicated (MinHash): skipping {ident}"),
    )
    logger.info(f"Retrieved {len(deduped)} deduplicated passages after ANN search")
    return deduped
