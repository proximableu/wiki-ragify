#!/usr/bin/env python3
"""
wiki_splitter_v2.py

Robust Wikipedia-style text splitter:

- Upptäcker både "== Heading ==" och "bare headings" (en rad utan punkt följd av text).
- Tar bort "References" / "See also" block (konfigurerbart).
- Chunkar per sektion (rubrik + följande stycken) och säkerställer maxstorlek per chunk.
- Stöd för token-baserad chunkning via tiktoken (valfritt). Annars fallback till ungefärlig ord-/tecken-baserad chunkning.
- Skapar säkra, unika filnamn och sparar som Markdown med rubrik.
- CLI: python wiki_splitter_v2.py <source_dir> <out_dir> [--max-tokens ...] [--max-chars ...]

Author: ChatGPT (omskriven för dina behov)
"""

from pathlib import Path
import re
import sys
import argparse
import json
import html
from typing import List, Tuple, Optional

# --- Konfig ---
BAD_END_HEADERS = {"references", "see also", "further reading", "external links"}
# Mönster för explicit Wiki-heading format (== Heading ==)
WIKI_HEADER_RE = re.compile(r"^==\s*(.*?)\s*==\s*$", flags=re.MULTILINE)
# Enkel rad-split
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

# --- Hjälpfunktioner ---


def clean_filename(name: str) -> str:
    """Rensa rubrik till säkert filnamn."""
    s = name.strip().lower()
    s = re.sub(r"[^\w\-\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    # Trimm längd
    return s[:120]


def save_unique_file(out_dir: Path, base_name: str, content: str) -> Path:
    """Spara utan att överskriva: lägg till _2, _3 … vid behov."""
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate = out_dir / f"{base_name}.md"
    i = 2
    while candidate.exists():
        candidate = out_dir / f"{base_name}_{i}.md"
        i += 1
    candidate.write_text(content, encoding="utf-8")
    return candidate


def is_potential_bare_header(line: str, next_line: Optional[str]) -> bool:
    """
    Heuristik för att upptäcka en 'bare' rubrik:
    - Linjen är kort (3..80 tecken)
    - Saknar punkt i slutet
    - Ser inte ut som en mening (flera ord, ingen avslutande punkt)
    - Nästa rad är tom eller börjar med versal (möjligen nytt stycke)
    """
    if not line:
        return False
    l = line.strip()
    if len(l) < 2 or len(l) > 80:
        return False
    # Undvik om det ser ut som en mening
    if l.endswith(".") or l.endswith("?") or l.endswith("!"):
        return False
    # Undvik om det innehåller ':' eller 'http' (ej rubrik)
    if ":" in l and not l.count(" "):
        return False
    if next_line is None:
        return False
    nl = next_line.strip()
    if nl == "":
        return True
    # Om nästa rad börjar med versal och verkar som en mening -> troligen rubrik
    if nl[0].isupper():
        # men om nästa rad är väldigt kort, fortfarande rubrik-troligt
        return True
    return False


def split_to_lines(text: str) -> List[str]:
    return LINE_SPLIT_RE.split(text)


def remove_trailing_sections(text: str) -> str:
    """Ta bort text efter dåliga slutsektioner (References, See also etc.)."""
    # Sök explicit '== ... ==' headers först
    for m in WIKI_HEADER_RE.finditer(text):
        header = m.group(1).strip().lower()
        if header in BAD_END_HEADERS:
            return text[: m.start()]
    # Fallback: sök lines som är enbart 'References' etc.
    lines = split_to_lines(text)
    for idx, line in enumerate(lines):
        if line.strip().lower() in BAD_END_HEADERS:
            return "\n".join(lines[:idx])
    return text


def find_sections(text: str) -> List[Tuple[str, str]]:
    """
    Returnerar lista av (title, body) i ordningsföljd.
    - Hanterar både '== Heading ==' och bare headings.
    - Inleder alltid med "Introduction" om text före första heading finns.
    """
    text = text.replace("\r\n", "\n")
    text = html.unescape(text)
    text = remove_trailing_sections(text)
    lines = split_to_lines(text)

    sections = []
    current_title = "Introduction"
    buffer_lines: List[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        # 1) explicit wiki header
        m = WIKI_HEADER_RE.match(line)
        if m:
            # push current buffer
            body = "\n".join(buffer_lines).strip()
            if body:
                sections.append((current_title, body))
            buffer_lines = []
            current_title = m.group(1).strip()
            i += 1
            continue

        # 2) bare header check (current line + next)
        next_line = lines[i + 1] if i + 1 < n else None
        if is_potential_bare_header(line, next_line):
            body = "\n".join(buffer_lines).strip()
            if body:
                sections.append((current_title, body))
            buffer_lines = []
            current_title = line.strip()
            # skip possible empty line after header
            if next_line is not None and next_line.strip() == "":
                i += 2
            else:
                i += 1
            continue

        # annars - en vanlig body-rad
        buffer_lines.append(line)
        i += 1

    # tail
    tail_body = "\n".join(buffer_lines).strip()
    if tail_body:
        sections.append((current_title, tail_body))

    return sections


def chunk_section(title: str, body: str, max_tokens: Optional[int] = None, max_chars: Optional[int] = 3000,
                  tokenizer_model: str = "gpt-4o-mini") -> List[Tuple[str, str]]:
    """
    Chunka en sektion (title + body) i bitar som inte överskrider max_tokens eller max_chars.
    Om tokeniserare finns, används den; annars fallback till char-based split på stycken.
    Returnerar lista av (sub_title, chunk_text).
    sub_title t.ex. "SectionTitle (part 1/2)"
    """
    if max_tokens and TOKENIZER_AVAILABLE:
        # token-baserad chunkning
        paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
        chunks = []
        cur = []
        cur_tokens = 0
        for p in paragraphs:
            p_tokens = get_token_count(p, tokenizer_model)
            if p_tokens > max_tokens:
                # splitera paragraph further by sentences
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

        results = []
        for idx, c in enumerate(chunks, start=1):
            sub_title = f"{title} (part {idx}/{len(chunks)})" if len(chunks) > 1 else title
            results.append((sub_title, c))
        return results

    # Fallback: char / paragraph-based chunking
    paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks = []
    cur = []
    cur_len = 0
    for p in paragraphs:
        p_len = len(p)
        if p_len > max_chars:
            # bryt ner långt stycke i mindre bitar (brutal men ok)
            for i in range(0, p_len, max_chars):
                part = p[i:i + max_chars]
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

    results = []
    for idx, c in enumerate(chunks, start=1):
        sub_title = f"{title} (part {idx}/{len(chunks)})" if len(chunks) > 1 else title
        results.append((sub_title, c))
    return results


def process_file(src_path: Path, out_dir: Path, max_tokens: Optional[int], max_chars: int, tokenizer_model: str):
    text = src_path.read_text(encoding="utf-8", errors="ignore")
    sections = find_sections(text)
    base_original = src_path.stem

    for title, body in sections:
        # renodla title
        if not title or not title.strip():
            title = "Introduction"
        # hoppa över väldigt små sektioner
        if len(body.strip()) < 40:
            # men vi kan spara som anteckning i fil "base_meta" om önskas - för nu skip
            continue

        # chunka
        subchunks = chunk_section(title, body, max_tokens=max_tokens, max_chars=max_chars,
                                  tokenizer_model=tokenizer_model)
        for sub_title, chunk_text in subchunks:
            safe = clean_filename(sub_title)
            full_heading = f"{base_original} — {sub_title}"
            md = f"## {full_heading}\n\n{chunk_text}\n"
            out_name = f"{base_original}_{safe}"
            saved = save_unique_file(out_dir, out_name, md)
            print(f"Saved: {saved.name}")


def parse_args():
    p = argparse.ArgumentParser(description="Robust Wikipedia-style splitter with bare-header detection.")
    p.add_argument("src_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--max-tokens", type=int, default=None,
                   help="Max tokens per chunk (requires tiktoken). If omitted, char-based chunking used.")
    p.add_argument("--max-chars", type=int, default=3000,
                   help="Max characters per chunk (fallback if tokens not used).")
    p.add_argument("--tokenizer-model", type=str, default="gpt-4o-mini",
                   help="Model name for tiktoken encoding (only relevant if --max-tokens is set).")
    return p.parse_args()


def main():
    args = parse_args()
    src_dir: Path = args.src_dir
    out_dir: Path = args.out_dir
    max_tokens = args.max_tokens
    max_chars = args.max_chars
    tokenizer_model = args.tokenizer_model

    if not src_dir.exists():
        print("Source directory not found:", src_dir)
        sys.exit(2)

    in_files = [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in (".txt", ".md")]
    if not in_files:
        print("No .txt/.md files found in", src_dir)
        sys.exit(0)

    if max_tokens and not TOKENIZER_AVAILABLE:
        print("Warning: --max-tokens requested but tiktoken not available. Falling back to --max-chars.")
        max_tokens = None

    print(f"Processing {len(in_files)} files -> output: {out_dir}")
    for f in in_files:
        print("Processing:", f.name)
        try:
            process_file(f, out_dir, max_tokens=max_tokens, max_chars=max_chars, tokenizer_model=tokenizer_model)
        except Exception as e:
            print(f"Error processing {f.name}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
