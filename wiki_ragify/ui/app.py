"""Wiki Ragify TUI — the application shell (DESIGN §6 + Phase 3.1).

This composes the widgets built in this package into one full-screen Textual app:

* :class:`~wiki_ragify.ui.widgets.folder_tree.FolderTree` (FR-1) — the left folder picker.
* :class:`~wiki_ragify.ui.widgets.progress_panel.ProgressPanel` (FR-3) — the central log.
* :class:`~wiki_ragify.ui.widgets.stats_panel.StatsPanel` (FR-4) — the right stats panel.
* :class:`~wiki_ragify.ui.widgets.query_box.QueryBox` (FR-8) — the footer query box.
* A footer of run controls: RUN / PAUSE / RESUME / STOP (DESIGN §7).
* A config modal that collects the start page + gating prompt before a run begins (FR-1 + §3.7).

The runner (``PipelineRunner``) runs on a Textual worker thread; events it emits flow back
to the UI through the app's ``on_event`` / ``on_stat`` callbacks, which post to the panels.
The app never calls ``os.system`` or spawns subprocesses — the funnel is a pure in-process
call. The run-controls state machine (idle/running/paused/stopped) is small and deliberate so
it can be reasoned about off-screen.

Signal handling (DESIGN §4.1 / FR-6): a signal can only be caught on the main thread, and
the runner lives on a worker thread, so the handler lives here. On SIGINT/SIGTERM the app
posts an internal :class:`Quit` message and :meth:`on_quit` runs the same graceful stop the
STOP button uses — the runner sets its stop event, finishes the current phase, persists the
checkpoint at that durable boundary, and its worker exits. The checkpoint is therefore
always up to date before the process leaves, matching the old orchestrator's behaviour.

Textual 8.2.8 note: there is no built-in prompt dialog (``Prompt`` arrived in Textual 1.0),
so the start-page / prompt entry is a small custom modal screen (:class:`ConfigScreen`) that
posts a :class:`ConfigResult` / :class:`ConfigCancelled` message and lets the app decide what
to do with the confirmation. All button handlers are keyed off the widget ``id`` because
8.2.8 dispatches ``Button.Pressed`` to a single ``on_button_pressed`` on the handler.
"""

from __future__ import annotations

import signal
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult, Screen
from textual.containers import Container
from textual.message import Message
from textual.widgets import Button, DataTable, Header, Input, Static

from ..config import Config, PipelineConfig
from ..pipeline.checkpoint import State, phase_progress
from ..pipeline.runner import PipelineRunner
from ..pipeline.events import ProgressEvent, StatEvent
from .widgets.folder_tree import FolderTree, ProjectInfo, ProjectSelected
from .widgets.progress_panel import ProgressPanel
from .widgets.stats_panel import StatsPanel
from .widgets.query_box import QueryBox, QueryRow, Submitted, build_rows, run_query


# --- Config modal (FR-1 + §3.7) -----------------------------------------------


class ConfigResult(Message):
    """Carries the confirmed start page + prompt path out of :class:`ConfigScreen`."""

    def __init__(self, start_page: str, prompt_path: str) -> None:
        super().__init__()
        self.start_page = start_page
        self.prompt_path = Path(prompt_path)


class ConfigCancelled(Message):
    """Posted when the config modal is dismissed without confirmation (Esc / Cancel)."""


class Quit(Message):
    """Internal signal: run's signal handler posts this so the graceful stop can happen."""


class ConfigScreen(Screen):
    """A small modal that collects the start page and gating prompt for a run.

    The heavy lifting (prompt rendering, project discovery) stays in the library; this screen
    only gathers the two strings and posts a :class:`ConfigResult` on confirmation.
    """

    def __init__(self, start_page: str = "", prompt_path: str = "") -> None:
        super().__init__(id="config-screen")
        self._start_page = start_page
        self._prompt_path = prompt_path

    def compose(self) -> ComposeResult:
        yield Static("Configure run", id="config-title")
        yield Static("Start page (Wikipedia title):")
        yield Input(value=self._start_page, id="start-page")
        yield Static("Gating prompt path:")
        yield Input(
            value=self._prompt_path,
            id="prompt-path",
            placeholder="absolute path to gating_prompt.txt",
        )
        yield Container(Button("Confirm", id="confirm"), Button("Cancel", id="cancel"), id="config-actions")

    async def on_mount(self) -> None:  # pragma: no cover - Textual lifecycle
        self.query_one("#start-page").focus()

    def on_input_submitted(self, message: Input.Submitted) -> None:  # pragma: no cover
        # Either field confirming with Enter submits the modal.
        self._submit()

    def on_button_pressed(self, message: Button.Pressed) -> None:  # pragma: no cover
        button_id = message.button.id or ""
        if button_id == "confirm":
            self._submit()
        elif button_id == "cancel":
            self.post_message(ConfigCancelled())
            self.pop_screen()

    def _submit(self) -> None:  # pragma: no cover - Textual modal
        start_page = self.query_one("#start-page", Input).value.strip()
        prompt_path = self.query_one("#prompt-path", Input).value.strip()
        if not start_page:
            return
        self.post_message(ConfigResult(start_page, prompt_path))
        self.pop_screen()


# --- Query results overlay (FR-8) --------------------------------------------


class QueryDone(Message):
    """Thread-safe carrier for a finished retrieval: results are posted back to the app."""

    def __init__(self, rows: list[QueryRow], message: str) -> None:
        super().__init__()
        self.rows = rows
        self.message = message


class QueryScreen(Screen):
    """Full-screen query results overlay (FR-8)."""

    def __init__(self, rows: list[QueryRow], message: str) -> None:
        super().__init__()
        self.id = "query-overlay"
        self._rows = rows
        self._message = message

    def compose(self) -> ComposeResult:
        yield Static(f"[subtle]{self._message}[/]", id="query-message")
        yield DataTable(id="query-table")
        yield Container(Button("Back", id="query-back"), id="query-back-actions")

    async def on_mount(self) -> None:  # pragma: no cover - Textual overlay
        table = self.query_one("#query-table", DataTable)
        table.add_column("Rank", key="rank")
        table.add_column("Score", key="score")
        table.add_column("File", key="file")
        table.add_column("Text", key="text")
        for row in self._rows:
            snippet = row.text.replace("\n", " ")
            if len(snippet) > 300:
                snippet = snippet[:297] + "..."
            table.add_row(str(row.rank), f"{row.score:.3f}", row.filename, snippet)

    def on_button_pressed(self, message: Button.Pressed) -> None:  # pragma: no cover
        if (message.button.id or "") == "query-back":
            self.pop_screen()


# --- Application shell -------------------------------------------------------


class WikiRagifyApp(App):
    """The full-screen Textual app: folder picker | progress window | stats panel.

    The runner runs on a Textual worker thread; its events are posted to the panels off the
    worker thread. PAUSE / RESUME / STOP map to the runner's primitives. The run controls,
    query box, and status indicator compose into the layout the stylesheet expects.
    """

    TITLE = "Wiki Ragify"

    def __init__(self, root: Path, config: Config) -> None:
        super().__init__()
        self.root = Path(root)
        self._base_config = config
        self.config = config
        self.runner: Optional[PipelineRunner] = None
        self._selected: Optional[ProjectInfo] = None
        # idle | running | paused | stopped
        self._run_state = "idle"

    def install_signal_handlers(self) -> None:
        """Register SIGINT/SIGTERM handlers that trigger a graceful stop + exit.

        Registered from ``on_mount`` (the guaranteed main thread): signals can only be caught
        there, and the runner lives on a worker thread, so we never catch a signal in-process
        on that thread.
        """
        for _sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(_sig, self._on_signal)

    def _on_signal(self, signum, _frame) -> None:  # pragma: no cover - main-thread only
        # Run in the app message loop, never in the signal frame.
        self.post_message(Quit())

    def on_mount(self) -> None:
        self.install_signal_handlers()

    def on_quit(self) -> None:
        """Handle a graceful stop: STOP the runner so it checkpoints, then leave the app.

        Mirrors the STOP button (``stop_run``): the runner sets its stop event, finishes the
        current phase, persists the checkpoint at that durable boundary, and its worker exits.
        Then the app exits. If no run is active, just exit.
        """
        self.stop_run()
        self.exit()

    # --- event fan-out (called by PipelineRunner on the worker thread) ---------

    def _on_event(self, event: ProgressEvent) -> None:
        panel = self.query_one("#progress-panel", ProgressPanel)
        panel.add_event(event)

    def _on_stat(self, event: StatEvent) -> None:
        panel = self.query_one("#stats-panel", StatsPanel)
        panel.feed(event)

    # --- run controls state machine -------------------------------------------

    @property
    def run_state(self) -> str:
        return self._run_state

    def _begin_run(self, pipeline: PipelineConfig) -> None:
        """Construct and start a PipelineRunner on an exclusive worker thread."""
        self.runner = PipelineRunner(
            config=self.config,
            pipeline=pipeline,
            on_event=self._on_event,
            on_stat=self._on_stat,
        )
        self._run_state = "running"
        self._refresh_status_bar()
        self.app.run_worker(self._drive, exclusive=True)

    def _drive(self) -> None:
        assert self.runner is not None
        self.runner.run()
        self._run_state = "stopped"
        self._refresh_status_bar()

    def toggle_pause(self) -> None:
        if self.runner is None:
            return
        if self._run_state == "running":
            self.runner.pause()
            self._run_state = "paused"
        elif self._run_state == "paused":
            self.runner.resume()
            self._run_state = "running"
        self._refresh_status_bar()

    def stop_run(self) -> None:
        if self.runner is not None:
            self.runner.stop()
        self._run_state = "stopped"
        self._refresh_status_bar()

    def _refresh_status_bar(self) -> None:
        labels = {
            "idle": "IDLE",
            "running": "RUNNING",
            "paused": "PAUSED",
            "stopped": "STOPPED",
        }
        indicator = self.query_one("#status-indicator", Static)
        indicator.content = labels[self._run_state]
        # Colour the bar via a state class (styles.tcss: .running/.paused/.stopped).
        indicator.remove_class("running", "paused", "stopped")
        indicator.add_class(self._run_state)

    # --- folder selection (FR-1) ----------------------------------------------

    def on_project_selected(self, message: ProjectSelected) -> None:
        project = message.project
        self._selected = project
        # Rebind config.output_dir so checkpoints, artifacts, and the query store all
        # resolve against the chosen project directory.
        self.config = self._base_config.with_output_dir(project.project_dir)
        state = State(project.state_file)
        done, total = phase_progress(state)
        self.query_one("#progress-panel", ProgressPanel).set_state(state)
        self._run_state = "idle"
        self._refresh_status_bar()

    # --- run controls (footer) ------------------------------------------------

    def on_button_pressed(self, message: Button.Pressed) -> None:  # pragma: no cover
        button_id = message.button.id or ""
        if button_id == "run-button":
            if self.runner is not None and self._run_state in ("running", "paused"):
                return
            if self._selected is None:
                self.query_one("#progress-panel", ProgressPanel).add_event(
                    ProgressEvent(
                        stage="crawl",
                        kind="warn",
                        message="Select a project in the folder picker first",
                    )
                )
                return
            self.push_screen(ConfigScreen(), callback=self._on_config_result)
        elif button_id == "pause-button":
            self.toggle_pause()
        elif button_id == "stop-button":
            self.stop_run()

    def _on_config_result(self, message: ConfigResult | ConfigCancelled) -> None:
        if isinstance(message, ConfigCancelled):
            return
        assert self._selected is not None
        output_dir = self._selected.project_dir
        cfg = self._base_config.with_output_dir(output_dir)
        pipeline = PipelineConfig(
            start_page=message.start_page,
            prompt_path=message.prompt_path,
            output_dir=output_dir,
            ingest=True,
        )
        # Run against the project-scoped config so checkpoints + query store line up.
        self.config = cfg
        self._begin_run(pipeline)

    # --- query box (FR-8) -----------------------------------------------------

    def on_query_submitted(self, message: Submitted) -> None:
        # Retrieval is blocking (embed + LLM relevance), so it runs on a worker *thread*;
        # the result is posted back to the main thread via QueryDone, and the overlay is
        # pushed there — the DOM is never touched from another thread.
        query = message.query
        self.app.run_worker(lambda: self._run_query(query), thread=True)

    def _run_query(self, query: str) -> None:  # pragma: no cover - needs live Ollama
        try:
            _, passages = run_query(query, self.config)
        except Exception as exc:
            rows, message = [], f"query failed: {exc}"
        else:
            message = f"{len(passages)} relevant passages" if passages else "no relevant passages"
            rows = build_rows(query, passages)
        self.post_message(QueryDone(rows, message))

    def on_query_done(self, message: QueryDone) -> None:
        self.push_screen(QueryScreen(message.rows, message.message))

    def compose(self) -> ComposeResult:
        """Build the full-screen layout: header | status | three panels | footer."""
        yield Header()
        yield Static("IDLE", id="status-indicator")
        yield Container(
            FolderTree(self.root, id="folder-tree"),
            ProgressPanel(),
            StatsPanel(),
            id="main",
        )
        yield Container(
            QueryBox(),
            Container(
                Button("Run", id="run-button"),
                Button("Pause", id="pause-button"),
                Button("Stop", id="stop-button"),
                id="run-controls",
            ),
            id="footer",
        )
