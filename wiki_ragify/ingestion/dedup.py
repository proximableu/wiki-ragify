"""Content deduplication via MinHash LSH.

Faithful port of the shingle-tokeniser and MinHash LSH helper from
``reference/build_db_vec.py`` and ``reference/retriever.py``. Two thresholds are
used across the pipeline — a tight 0.98 for deduplicating documents at ingestion,
and a looser 0.65 for deduplicating candidate passages at retrieval. Both share
the same 3-word shingling so they speak the same similarity language.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from datasketch import MinHash, MinHashLSH

# 3-word shingles keep local word order, so two short documents that share a
# phrase still hash close together rather than only scoring on bag-of-words overlap.
_SHINGLE_RE = re.compile(r"[^a-z0-9\s]+", re.UNICODE)


def shingle_tokens(text: str, num_words: int = 3) -> list[str]:
    """Tokenise ``text`` into ``num_words``-word shingles.

    Lowercases, removes non-alphanumeric characters, then slides a window of
    ``num_words`` words. Documents shorter than the window fall back to single
    words so they still hash. Mirrors ``reference/*_vec.py::_tokenize``.
    """
    text = text.lower().strip()
    text = _SHINGLE_RE.sub(" ", text)
    words = [w for w in text.split() if w]
    if len(words) >= num_words:
        return [" ".join(words[i : i + num_words]) for i in range(len(words) - num_words + 1)]
    return words


def _to_minhash(tokens: list[str], num_perm: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for token in tokens:
        m.update(token.encode("utf-8"))
    return m


@dataclass(frozen=True)
class DedupConfig:
    """MinHash LSH parameters for a single stage of the pipeline."""

    threshold: float
    num_perm: int


# Document-level dedup (ingestion, 0.98) and passage-level dedup (retrieval, 0.65).
INGEST = DedupConfig(threshold=0.98, num_perm=256)
RETRIEVAL = DedupConfig(threshold=0.65, num_perm=128)


def index_documents(
    texts: list[tuple[str, str]],
    dedup: DedupConfig | None = None,
    on_dedup: Callable[[str], None] | None = None,
) -> list[tuple[str, str]]:
    """Return documents that are not near-duplicates of each other.

    ``texts`` is a list of ``(identifier, text)`` pairs. Returns the subset kept
    after MinHash LSH deduplication (threshold 0.98). Files whose token set is
    empty are dropped, matching the reference (which skips empty shingle sets).
    Order is preserved.
    """
    dedup = dedup or INGEST
    lsh = MinHashLSH(threshold=dedup.threshold, num_perm=dedup.num_perm)
    kept: list[tuple[str, str]] = []

    for ident, text in texts:
        tokens = shingle_tokens(text)
        if not tokens:
            continue
        m = _to_minhash(tokens, dedup.num_perm)
        if lsh.query(m):
            if on_dedup:
                on_dedup(ident)
            continue
        lsh.insert(ident, m)
        kept.append((ident, text))

    return kept


def deduplicate_passages(
    passages: list[tuple[str, str]],
    dedup: DedupConfig | None = None,
    limit: int | None = None,
    on_dedup: Callable[[str], None] | None = None,
) -> list[tuple[str, str]]:
    """De-duplicate a merged candidate pool of ``(identifier, text)`` passages.

    Sorts by descending similarity score first (passages carry a leading score),
    then drops any passage within the dedup threshold of one already selected.
    This mirrors the reference's ``deduplicate_passages`` for unifying candidate
    pools across multiple queries.
    """
    dedup = dedup or RETRIEVAL

    sorted_passages = sorted(passages, key=lambda p: p[0], reverse=True)
    lsh = MinHashLSH(threshold=dedup.threshold, num_perm=dedup.num_perm)
    unique: list[tuple[str, str]] = []

    for ident, text in sorted_passages:
        tokens = shingle_tokens(text)
        if not tokens:
            continue
        m = _to_minhash(tokens, dedup.num_perm)
        if lsh.query(m):
            if on_dedup:
                on_dedup(ident)
            continue
        lsh.insert(ident, m)
        unique.append((ident, text))
        if limit is not None and len(unique) >= limit:
            break

    return unique
