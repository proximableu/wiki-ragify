"""Offline unit tests for the splitter — pure functions, no network/LLM."""

from wiki_ragify.splitting import (
    find_sections,
    chunk_section,
    clean_filename,
    remove_trailing_sections,
    process_file,
)


def test_clean_filename():
    # Non-word chars (like = / &) are stripped; spaces collapse to single _
    assert clean_filename("== Hello World ==") == "_hello_world_"
    assert clean_filename("A/B & C") == "ab_c"


def test_find_sections_explicit():
    text = "Intro paragraph.\n\n== Background ==\nSome background text.\n\n== Conclusions ==\nThe end."
    sections = find_sections(text)
    titles = [t for t, _ in sections]
    assert titles == ["Introduction", "Background", "Conclusions"]
    assert "Some background text." in dict(sections)["Background"]


def test_find_sections_removes_references():
    text = "Real content.\n\n== References ==\n[1] A citation"
    sections = find_sections(text)
    assert [t for t, _ in sections] == ["Introduction"]


def test_chunk_section_max_chars():
    # 3000-char default cap; body of 5000 chars of 'x' -> multiple chunks
    body = "x" * 5000
    chunks = chunk_section("Title", body, max_tokens=None, max_chars=1000)
    assert len(chunks) > 1
    # Each chunk individually respects the cap
    assert all(len(c) <= 1000 for _, c in chunks)
    # The concatenated chunk bodies sum to the original length (content not lost)
    total = sum(len(c) for _, c in chunks)
    assert total == 5000


def test_process_file_writes(tmp_path):
    src = tmp_path / "sample.txt"
    body = "Intro paragraph that is long enough to survive the minimum length filter.\n\n"
    body += "== Section One ==\n\n" + "A sufficiently long paragraph of content that will not be dropped " \
            "by the splitter's short-section filter.\n"
    src.write_text(body, encoding="utf-8")
    out = tmp_path / "chunks"
    saved = process_file(src, out, max_tokens=None, max_chars=3000)
    assert len(saved) >= 1
    assert all(p.suffix == ".md" for p in saved)
    # First chunk should contain the intro
    content = (saved[0]).read_text(encoding="utf-8")
    assert "Intro paragraph" in content
