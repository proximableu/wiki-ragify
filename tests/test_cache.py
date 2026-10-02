"""Offline unit tests for the crawler Cache (pure Python, no network/LLM needed)."""

from wiki_ragify.crawler.cache import Cache


def test_round_trip(tmp_path):
    p = tmp_path / "download_cache.json"
    c = Cache()
    c.add_accepted("Artificial intelligence", "Artificial_intelligence.txt")
    c.add_rejected("List of things")
    c.save(p)
    assert p.exists()

    loaded = Cache.load(p)
    assert loaded.accepted == {"Artificial intelligence": "Artificial_intelligence.txt"}
    assert loaded.rejected == ["List of things"]
    assert loaded.titles == ["Artificial intelligence"]


def test_missing_returns_fresh(tmp_path):
    c = Cache.load(tmp_path / "nope.json")
    assert c.accepted == {}
    assert c.rejected == []


def test_corrupt_returns_fresh(tmp_path):
    p = tmp_path / "cache.json"
    p.write_text("{not valid json", encoding="utf-8")
    c = Cache.load(p)
    assert c.accepted == {}
    assert c.rejected == []


def test_dedupe_rejected(tmp_path):
    p = tmp_path / "cache.json"
    c = Cache()
    c.add_rejected("Dup")
    c.add_rejected("Dup")
    c.save(p)
    assert Cache.load(p).rejected == ["Dup"]


def test_save_creates_parent_dirs(tmp_path):
    p = tmp_path / "nested" / "deeper" / "cache.json"
    c = Cache()
    c.add_accepted("X", "X.txt")
    c.save(p)
    assert Cache.load(p).accepted == {"X": "X.txt"}
