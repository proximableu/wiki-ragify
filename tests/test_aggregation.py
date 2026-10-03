"""Offline tests for ``wiki_ragify.ui.aggregation``.

Statistics folding is pure logic with no UI and no wall-clock dependence (an explicit
``now_s`` is passed in), so it is unit-tested directly against synthetic events.
"""

from wiki_ragify.pipeline.events import ProgressEvent
from wiki_ragify.ui.aggregation import Statistics, format_duration


def _progress(stage, kind, message=""):
    return ProgressEvent(stage=stage, index=0, total=None, kind=kind, message=message, ts=0.0)


def test_crawl_counts_page_accept_and_reject():
    stats = Statistics()
    stats.update_stats(_progress("crawl", "accept", "A"), now_s=10.0)
    stats.update_stats(_progress("crawl", "reject", "B"), now_s=20.0)
    stats.update_stats(_progress("crawl", "accept", "C"), now_s=30.0)
    snap = stats.snapshot()
    assert snap.pages_crawled == 3
    assert snap.accepted == 2
    assert snap.rejected == 1
    # skip/tick events do not count as a page crawled
    stats.update_stats(_progress("crawl", "tick"), now_s=31.0)
    assert stats.snapshot().pages_crawled == 3


def test_gate_counts_chunk_accept_and_reject():
    stats = Statistics()
    stats.update_stats(_progress("gate", "accept"), now_s=0.0)
    stats.update_stats(_progress("gate", "reject"), now_s=0.0)
    stats.update_stats(_progress("gate", "accept"), now_s=0.0)
    snap = stats.snapshot()
    assert snap.gate2_accept == 2
    assert snap.gate2_reject == 1


def test_split_and_ingest_counts():
    stats = Statistics()
    stats.update_stats(_progress("split", "tick"), now_s=0.0)
    # ingest reports counts via messages, not accept/skip kinds.
    stats.update_stats(
        _progress("ingest", "tick", "Dedup: 8/10 unique"), now_s=0.0
    )
    stats.update_stats(
        _progress("ingest", "stage_done", "Ingested 8 documents"), now_s=0.0
    )
    snap = stats.snapshot()
    assert snap.chunks == 1
    assert snap.dedup_removed == 2
    assert snap.db_rows == 8


def test_elapsed_seconds_is_delta_from_first_event():
    stats = Statistics()
    stats.update_stats(_progress("crawl", "accept"), now_s=12.0)
    stats.update_stats(_progress("crawl", "reject"), now_s=42.0)
    assert stats.snapshot().elapsed_s == 30.0


def test_crawl_rate_is_per_minute():
    stats = Statistics()
    # 2 pages crawled over 2 minutes = 1 page/min.
    stats.update_stats(_progress("crawl", "accept"), now_s=0.0)
    stats.update_stats(_progress("crawl", "reject"), now_s=120.0)
    snap = stats.snapshot()
    assert snap.crawl_rate == 1.0


def test_rate_is_zero_before_any_time_elapses():
    stats = Statistics()
    stats.update_stats(_progress("crawl", "accept"), now_s=0.0)
    snap = stats.snapshot()
    assert snap.elapsed_s == 0.0
    assert snap.crawl_rate == 0.0


def test_ordered_rows_labels_values():
    stats = Statistics()
    stats.update_stats(_progress("crawl", "accept"), now_s=0.0)
    rows = stats.snapshot().ordered_rows()
    keys = [metric for _, metric, _ in rows]
    # crawl is first in display order
    assert keys[0] == "pages_crawled"
    labels = {metric: label for label, metric, _ in rows}
    assert labels["pages_crawled"] == "pages crawled"


def test_format_duration_drops_leading_groups():
    assert format_duration(5) == "5s"
    assert format_duration(65) == "1m 5s"
    assert format_duration(3661) == "1h 1m 1s"
    assert format_duration(-10) == "0s"
