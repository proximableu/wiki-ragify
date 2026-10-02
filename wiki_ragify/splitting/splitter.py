"""Text splitter — moves the reference implementation verbatim into a library module.

The pure functions (section detection + chunking) are untouched. Only the CLI wiring is
dropped; the runner drives ``process_dir`` instead.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import List, Optional, Tuple

# --- Config ---
BAD_END_HEADERS = {"references", "see also", "further reading", "external links"}
# Explicit Wiki-heading format (== Heading ==)
WIKI_HEADER_RE = re.compile(r"^==\s*(.*?)\s*==\s*$", flags=re.MULTILINE)
LINE_SPLIT_RE = re.compile(r"\r?\n")

# --- Tokenizer fallback ---
try:
    import tiktoken

    def get_token_count(text: str, model: str = "gpt-4o-mini"):
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))

    TOKENIZER_AVAILABLE = True
except Exception:
    TOKENIZER_AVAILABLE = False


def clean_filename(name: str) -> str:
    """Clean a heading to a safe filename."""
    s = name.strip().lower()
    s = re.sub(r"[^\w\-\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:120]


def save_unique_file(out_dir: Path, base_name: str, content: str) -> Path:
    """Write without overwriting: append _2, _3 … as needed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate = out_dir / f"{base_name}.md"
    i = 2
    while candidate.exists():
        candidate = out_dir / f"{base_name}_{i}.md"
        i += 1
    candidate.write_text(content, encoding="utf-8")
    return candidate


def is_potential_bare_header(line: str, next_line: Optional[str]) -> bool:
    """Heuristic for detecting a 'bare' heading (a short line without trailing
    punctuation, followed by an empty line or an uppercase-starting line)."""
    if not line:
        return False
    l = line.strip()
    if len(l) < 2 or len(l) > 80:
        return False
    if l.endswith(".") or l.endswith("?") or l.endswith("!"):
        return False
    if ":" in l and not l.count(" "):
        return False
    if next_line is None:
        return False
    nl = next_line.strip()
    if nl == "":
        return True
    if nl[0].isupper():
        return True
    return False


def remove_trailing_sections(text: str) -> str:
    """Remove text after bad trailing sections (References, See Also, etc.)."""
    # Explicit '== ... ==' headers first
    for m in WIKI_HEADER_RE.finditer(text):
        header = m.group(1).strip().lower()
        if header in BAD_END_HEADERS:
            return text[: m.start()]
    # Fallback: lines that are just 'References' etc.
    for line in LINE_SPLIT_RE.split(text):
        if line.strip().lower() in BAD_END_HEADERS:
            return text
    return text


def find_sections(text: str) -> List[Tuple[str, str]]:
    """Return a list of (title, body) sections in order.

    Handles both '== Heading ==' and bare headings. Always starts with 'Introduction'
    for any text preceding the first heading.
    """
    text = text.replace("\r\n", "\n")
    text = html.unescape(text)
    text = remove_trailing_sections(text)
    lines = LINE_SPLIT_RE.split(text)

    sections: List[Tuple[str, str]] = []
    current_title = "Introduction"
    buffer_lines: List[str] = []

    def flush():
        body = "\n".join(buffer_lines).strip()
        if body:
            sections.append((current_title, body))
        buffer_lines.clear()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        # 1) explicit wiki header
        m = WIKI_HEADER_RE.match(line)
        if m:
            flush()
            current_title = m.group(1).strip()
            i += 1
            continue

        # 2) bare header check (current line + next)
        next_line = lines[i + 1] if i + 1 < n else None
        if is_potential_bare_header(line, next_line):
            flush()
            current_title = line.strip()
            # Skip possible empty line after header
            if next_line is not None and next_line.strip() == "":
                i += 2
            else:
                i += 1
            continue

        # otherwise a normal body line
        buffer_lines.append(line)
        i += 1

    flush()
    return sections


def chunk_section(title: str, body: str, max_tokens: Optional[int] = None,
                  max_chars: Optional[int] = 3000, tokenizer_model: str = "gpt-4o-mini"
                  ) -> List[Tuple[str, str]]:
    """Split a section (title + body) into chunks no larger than max_tokens or max_chars.

    Uses tiktoken for token-accurate chunking when available; otherwise falls back to
    char-based splitting. Returns (sub_title, chunk_text) pairs.
    """
    if max_tokens and TOKENIZER_AVAILABLE:
        # Token-based chunking
        paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
        chunks: List[str] = []
        cur: List[str] = []
        cur_tokens = 0
        for p in paragraphs:
            p_tokens = get_token_count(p, tokenizer_model)
            if p_tokens > max_tokens:
                # Split long paragraphs by sentence
                sentences = re.split(r'(?<=[.!?])\s+', p)
                for s in sentences:
                    s_tokens = get_token_count(s, tokenizer_model)
                    if cur_tokens + s_tokens > max_tokens:
                        if cur:
                            chunks.append("\n\n".join(cur))
                        cur = [s]
                        cur_tokens = s_tokens
                    else:
                        cur.append(s)
                        cur_tokens += s_tokens
                continue

            if cur_tokens + p_tokens > max_tokens:
                if cur:
                    chunks.append("\n\n".join(cur))
                cur = [p]
                cur_tokens = p_tokens
            else:
                cur.append(p)
                cur_tokens += p_tokens
        if cur:
            chunks.append("\n\n".join(cur))
    else:
        # Fallback: char/paragraph-based chunking
        paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
        chunks: List[str] = []
        cur: List[str] = []
        cur_len = 0
        for p in paragraphs:
            p_len = len(p)
            if p_len > max_chars:
                # Break long paragraphs into smaller pieces
                for j in range(0, p_len, max_chars):
                    part = p[j:j + max_chars]
                    if cur:
                        chunks.append("\n\n".join(cur))
                        cur = []
                        cur_len = 0
                    chunks.append(part)
                continue

            if cur_len + p_len + 2 > max_chars:
                if cur:
                    chunks.append("\n\n".join(cur))
                cur = [p]
                cur_len = p_len
            else:
                cur.append(p)
                cur_len += p_len + 2
        if cur:
            chunks.append("\n\n".join(cur))

    results: List[Tuple[str, str]] = []
    for idx, c in enumerate(chunks, start=1):
        sub_title = f"{title} (part {idx}/{len(chunks)})" if len(chunks) > 1 else title
        results.append((sub_title, c))
    return results


def process_file(src_path: Path, out_dir: Path, max_tokens: Optional[int] = None,
                 max_chars: int = 3000, tokenizer_model: str = "gpt-4o-mini"):
    """Split one source file into chunks; return the list of output paths."""
    text = src_path.read_text(encoding="utf-8", errors="ignore")
    sections = find_sections(text)
    base_original = src_path.stem

    saved: List[Path] = []
    for title, body in sections:
        if not title or not title.strip():
            title = "Introduction"
        # Skip very small sections
        if len(body.strip()) < 40:
            continue

        subchunks = chunk_section(title, body, max_tokens=max_tokens, max_chars=max_chars,
                                  tokenizer_model=tokenizer_model)
        for sub_title, chunk_text in subchunks:
            safe = clean_filename(sub_title)
            full_heading = f"{base_original} — {sub_title}"
            md = f"## {full_heading}\n\n{chunk_text}\n"
            out_name = f"{base_original}_{safe}"
            saved.append(save_unique_file(out_dir, out_name, md))
    return saved
