"""Unit tests for the pure progress-banner helpers (FR-3).

The stage banner (bar + percentage + ETA/rate) is factored into pure functions so the
display math — which used to live inline in the Textual widget — is testable offline.
"""

from __future__ import annotations

from wiki_ragify.ui.widgets.progress_panel import (
    eta_str,
    progress_bar,
    rate_str,
    render_banner,
)


def test_progress_bar_edges():
    assert progress_bar(0.0) == "░" * 20
    assert progress_bar(1.0) == "█" * 20
    assert progress_bar(0.5, width=10) == "█" * 5 + "░" * 5


def test_progress_bar_clamps():
    assert progress_bar(-1.0) == "░" * 20
    assert progress_bar(2.0) == "█" * 20


def test_rate_str():
    assert rate_str(0) == "0.0/s"
    assert rate_str(-1) == "0.0/s"
    assert rate_str(0.5) == "0.5/s"


def test_eta_str_unknown():
    assert eta_str(None, 1.0) == "--:--"
    assert eta_str(10, 0) == "--:--"


def test_eta_str_formats():
    assert eta_str(60, 1.0) == "1m 00s"  # 60s, seconds zero-padded
    assert eta_str(30, 1.0) == "30s"     # under a minute
    assert eta_str(0, 2.0) == "0s"


def test_render_banner_live_stage():
    out = render_banner("crawl", index=42, total=63, phases_done=1, phases_total=6, rate=0.5)
    lines = out.splitlines()
    assert "Stage · crawl" in lines[0]
    assert "67%" in out                   # 42/63 rounds to 67%
    assert "ETA" in out and "/s" in out
    # bar portion (bar width 20 + "  " 2 + "67%" 3) precedes the ETA suffix.
    assert len(lines[1].split("ETA")[0].strip()) == 25


def test_render_banner_phase_view():
    out = render_banner(None, None, None, phases_done=2, phases_total=6, rate=0.0)
    assert "2/6 phases" in out
    assert "ETA --:--" in out


def test_render_banner_idle_first_phase():
    out = render_banner(None, None, None, phases_done=0, phases_total=6, rate=0.0)
    assert "0/6 phases" in out
    assert out.splitlines()[1].startswith("░" * 20)
