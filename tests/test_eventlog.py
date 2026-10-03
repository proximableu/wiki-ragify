"""Offline tests for ``wiki_ragify.ui.eventlog``.

The event log is pure folding logic (wrap at ``max``, newest-first ordering, style
classification) with no UI, so it is unit-tested directly against synthetic events.
"""

from wiki_ragify.pipeline.events import ProgressEvent
from wiki_ragify.ui.eventlog import EventLog, LogRow


def _event(stage, kind, message, index=None, total=None):
    return ProgressEvent(stage=stage, index=index or 0, total=total, kind=kind, message=message)


def _log(*events):
    log = EventLog()
    for ev in events:
        log.append(ev)
    return log


def test_classify_maps_kind_to_style():
    assert EventLog.classify("accept") == "accent2"
    assert EventLog.classify("reject") == "error"
    assert EventLog.classify("skip") == "subtle"
    assert EventLog.classify("warn") == "warning"
    # unknown kind falls back to neutral info
    assert EventLog.classify("weird") == "primary"


def test_append_keeps_newest_first_and_wraps_at_max():
    log = EventLog(max=3)
    for i in range(10):
        log.append(_event("crawl", "tick", f"item {i}"))
    rows = log.rows()  # newest-first
    assert len(rows) == 3
    assert rows[0].text == "item 9"
    assert rows[-1].text == "item 7"


def test_append_composes_index_total_prefix():
    log = _log(_event("gate", "accept", "Biology", index=42, total=63))
    assert log.rows()[0].text == "[42/63] Biology"
    # no index/total -> bare message
    log2 = _log(_event("crawl", "skip", "already saved"))
    assert log2.rows()[0].text == "already saved"


def test_append_records_style_and_index():
    log = _log(
        _event("gate", "accept", "A", index=1, total=2),
        _event("gate", "reject", "B", index=2, total=2),
    )
    rows = log.rows()
    assert rows[0] == LogRow("[2/2] B", "error", 2)
    assert rows[1] == LogRow("[1/2] A", "accent2", 1)


def test_clear_removes_all_rows():
    log = _log(_event("crawl", "tick", "x"))
    log.clear()
    assert log.rows() == []


def test_classify_is_constant():
    # guard against accidental refactor: stage_done is a distinct bright style
    assert EventLog.classify("stage_done") == "success"
