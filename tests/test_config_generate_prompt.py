"""Test for the CONFIG modal "Generate" prompt button (DESIGN change-request).

The button synthesizes a gating prompt from two new fields (target topic / explicitly
exclude) via ``prompt_gen.generate_gating_prompt`` and writes the result to disk,
prefilling the Gating prompt path field when it was left empty. Generation is network
I/O, so it runs on a worker thread; the result is posted back and the notification is
updated on the main thread.

Textual has no async pytest plugin here, so each test runs ``run_test`` inside
``anyio.run`` via a zero-arg coroutine factory (see ``test_app_event_fanout.py``).
"""

from __future__ import annotations

from pathlib import Path

import anyio
import pytest

from wiki_ragify.config import Config
from wiki_ragify.gating import prompt_gen
from wiki_ragify.ui.app import WikiRagifyApp, ConfigScreen


def _make_app(tmp_path: Path) -> WikiRagifyApp:
    return WikiRagifyApp(root=tmp_path, config=Config(output_dir=tmp_path))


def _run(coro_factory) -> None:
    anyio.run(coro_factory)


def _generate(app: WikiRagifyApp, monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive the Generate button and await the worker result, with generation faked."""
    monkeypatch.setattr(prompt_gen, "DEFAULT_TEMPLATE", prompt_gen.DEFAULT_TEMPLATE)
    monkeypatch.setattr(prompt_gen, "_generate", lambda *a, **k: "GENERATED PROMPT")
    app.screen.query_one("#generate-prompt").press()
    for _ in range(10):
        app.screen._eventlog  # no-op; real awaits below


async def _test_generate_writes_and_prefills(tmp_path, monkeypatch):
    app = _make_app(tmp_path)
    monkeypatch.setattr(prompt_gen, "DEFAULT_TEMPLATE", prompt_gen.DEFAULT_TEMPLATE)
    monkeypatch.setattr(prompt_gen, "_generate", lambda *a, **k: "GENERATED PROMPT")
    async with app.run_test() as pilot:
        app.push_screen(ConfigScreen(output_root=str(app.root), config=app.config))
        await pilot.pause()
        app.screen.query_one("#target-topic").text = "autistic supremacism"
        app.screen.query_one("#exclude-topic").text = "violence"
        app.screen.query_one("#generate-prompt").press()
        for _ in range(10):
            await pilot.pause()

        dest = tmp_path / "prompts" / "autistic-supremacism.txt"
        assert dest.exists()
        assert dest.read_text(encoding="utf-8") == "GENERATED PROMPT"
        assert app.screen.query_one("#prompt-path").value == str(dest)


async def _test_generate_overwrites(tmp_path, monkeypatch):
    explicit = tmp_path / "my_prompt.txt"
    explicit.write_text("OLD CONTENT", encoding="utf-8")
    app = _make_app(tmp_path)
    monkeypatch.setattr(prompt_gen, "DEFAULT_TEMPLATE", prompt_gen.DEFAULT_TEMPLATE)
    monkeypatch.setattr(prompt_gen, "_generate", lambda *a, **k: "NEW PROMPT")
    async with app.run_test() as pilot:
        app.push_screen(ConfigScreen(output_root=str(app.root), prompt_path=str(explicit), config=app.config))
        await pilot.pause()
        app.screen.query_one("#target-topic").text = "topic"
        app.screen.query_one("#generate-prompt").press()
        for _ in range(10):
            await pilot.pause()
        assert explicit.read_text(encoding="utf-8") == "NEW PROMPT"


async def _test_generate_empty_topic_is_noop(tmp_path, monkeypatch):
    app = _make_app(tmp_path)
    monkeypatch.setattr(prompt_gen, "_generate", lambda *a, **k: "SHOULD NOT RUN")
    async with app.run_test() as pilot:
        app.push_screen(ConfigScreen(output_root=str(app.root), config=app.config))
        await pilot.pause()
        app.screen.query_one("#generate-prompt").press()
        for _ in range(6):
            await pilot.pause()
        warning = app.screen.query_one("#output-warning").content
        assert "required" in str(warning)
        assert not list((tmp_path / "prompts").glob("*.txt"))


def test_generate_writes_file_and_prefills_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _run(lambda: _test_generate_writes_and_prefills(tmp_path, monkeypatch))


def test_generate_overwrites_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _run(lambda: _test_generate_overwrites(tmp_path, monkeypatch))


def test_generate_empty_topic_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _run(lambda: _test_generate_empty_topic_is_noop(tmp_path, monkeypatch))
