"""Fan-out test for the app event handlers (the fix for the empty stats panel).

The runner emits a single stream of :class:`ProgressEvent`s. The progress panel
renders it; the stats panel *folds* the same stream into its rolling metrics, but
only if the app routes each event to both. This test pins that contract: feeding a
:class:`ProgressEvent` through ``app._on_event`` must advance the stats accumulator
(as the already-tested ``aggregation`` module expects) *and* land in the progress
panel — proving the fan-out that Bug B/C relied on is wired correctly.

Textual has no async pytest plugin here, so each test is a sync function that runs
``run_test`` inside ``anyio.run`` (the Textual event loop is an anyio backend).
"""

from __future__ import annotations

from pathlib import Path

import anyio

from wiki_ragify.pipeline.events import ProgressEvent, StatEvent
from wiki_ragify.ui.app import WikiRagifyApp
from wiki_ragify.ui.widgets.progress_panel import ProgressPanel
from wiki_ragify.ui.widgets.stats_panel import StatsPanel


def _run(coro_factory) -> None:
    """Drive an async Textual test in a synchronous test function.

    ``anyio.run`` calls its first argument as a coroutine factory, so we pass a
    zero-arg lambda that returns the coroutine rather than the coroutine itself.
    """
    anyio.run(lambda: coro_factory())


def _make_app(tmp_path: Path) -> WikiRagifyApp:
    from wiki_ragify.config import Config

    return WikiRagifyApp(root=tmp_path, config=Config(output_dir=tmp_path))


async def _fanout_into_stats(app: WikiRagifyApp) -> None:
    async with app.run_test() as pilot:
        app._on_event(
            ProgressEvent(stage="crawl", kind="accept", index=1, total=3, message="X")
        )
        await pilot.pause()
        stats = app.query_one("#stats-panel", StatsPanel)
        snap = stats.stats.snapshot()
        assert snap.pages_crawled == 1
        assert snap.accepted == 1


async def _also_updates_progress(app: WikiRagifyApp) -> None:
    async with app.run_test() as pilot:
        app._on_event(ProgressEvent(stage="gate", kind="reject", message="bad chunk"))
        await pilot.pause()

        panel = app.query_one("#progress-panel", ProgressPanel)
        rows = panel._log.render_snapshot_rows()
        assert rows and rows[-1][0] == "bad chunk"
        # And the stats panel must have folded the same reject event too.
        stats = app.query_one("#stats-panel", StatsPanel)
        assert stats.stats.snapshot().gate2_reject == 1


async def _on_stat_still_folds(app: WikiRagifyApp) -> None:
    async with app.run_test() as pilot:
        app._on_stat(StatEvent(name="embeddings", value=7))
        await pilot.pause()
        stats = app.query_one("#stats-panel", StatsPanel)
        assert stats.stats.snapshot().embeddings == 7


def test_on_event_fans_into_stats_panel(tmp_path: Path) -> None:
    _run(lambda: _fanout_into_stats(_make_app(tmp_path)))


def test_on_event_leaves_progress_panel_also_updated(tmp_path: Path) -> None:
    _run(lambda: _also_updates_progress(_make_app(tmp_path)))


def test_on_stat_still_folds(tmp_path: Path) -> None:
    _run(lambda: _on_stat_still_folds(_make_app(tmp_path)))
