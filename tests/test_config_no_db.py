"""Test the "do not create database" checkbox in the CONFIG modal.

The checkbox toggles ``PipelineConfig.ingest``: off (default) builds the vector DB as
phase 6, on skips it. Verifies the value flows from the Switch, through ``ConfigResult``
and ``_on_config_result``, into the ``PipelineConfig`` that RUN builds.

Textual has no async pytest plugin here, so the test runs ``run_test`` inside
``anyio.run`` via a zero-arg coroutine factory (see ``test_app_event_fanout.py``).
"""

from __future__ import annotations

from pathlib import Path

import anyio
import pytest

from wiki_ragify.config import Config
from wiki_ragify.ui.app import WikiRagifyApp, ConfigScreen
from textual.widgets import Switch


def _make_app(tmp_path: Path) -> WikiRagifyApp:
    return WikiRagifyApp(root=tmp_path, config=Config(output_dir=tmp_path))


async def _open_config(app: WikiRagifyApp, pilot) -> None:
    await pilot.pause()
    app.push_screen(ConfigScreen(output_root=str(app.root), config=app.config))
    await pilot.pause()


def test_default_creates_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            await _open_config(app, pilot)
            # Switch defaults to off -> ingest/DB creation enabled.
            assert app.screen.query_one("#no-db", Switch).value is False
            app.screen.query_one("#start-page").value = "Seed"
            app.screen.query_one("#prompt-path").value = str(tmp_path / "prompt.txt")
            app.screen.query_one("#output-dir").value = str(tmp_path)
            (tmp_path / "prompt.txt").write_text("gate it")
            app.screen.query_one("#confirm").press()
            for _ in range(3):
                await pilot.pause()

            assert app._config_create_db is True
            assert app._make_pipeline().ingest is True

    anyio.run(scenario)


def test_checkbox_skips_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        app = _make_app(tmp_path)
        async with app.run_test() as pilot:
            await _open_config(app, pilot)
            await pilot.pause()
            app.screen.query_one("#no-db", Switch).value = True
            app.screen.query_one("#start-page").value = "Seed"
            app.screen.query_one("#prompt-path").value = str(tmp_path / "prompt.txt")
            app.screen.query_one("#output-dir").value = str(tmp_path)
            (tmp_path / "prompt.txt").write_text("gate it")
            app.screen.query_one("#confirm").press()
            for _ in range(3):
                await pilot.pause()

            assert app._config_create_db is False
            assert app._make_pipeline().ingest is False

    anyio.run(scenario)
