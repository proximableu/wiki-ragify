"""Wiki Ragify TUI — the application shell (DESIGN §6 + Phase 3.1).

This composes the widgets built in this package into one full-screen Textual app:

* :class:`~wiki_ragify.ui.widgets.progress_panel.ProgressPanel` (FR-3) — the central log.
* :class:`~wiki_ragify.ui.widgets.stats_panel.StatsPanel` (FR-4) — the right stats panel.
* :class:`~wiki_ragify.ui.widgets.query_box.QueryBox` (FR-8) — the footer query box.
* A footer of run controls: RUN / PAUSE / RESUME / STOP (DESIGN §7).
* A config modal that collects the start page, gating prompt, and **output directory**
  before a run begins (FR-1 + §3.7). The output directory used to live in a sidebar
  folder picker (the empty root left that paned blank); it now belongs in the config
  modal, which is the only place a project folder can actually be picked in Textual 8.2.8
  (no ``FileBrowser`` / ``DirectoryPicker`` exist here), and where the rest of the run
  parameters are set.

The runner (``PipelineRunner``) runs on a Textual worker thread; events it emits flow back
to the UI through the app's ``on_event`` / ``on_stat`` callbacks, which post to the panels.
The app never calls ``os.system`` or spawns subprocesses — the funnel is a pure in-process
call. The run-controls state machine (idle/running/paused/stopped) is small and deliberate so
it can be reasoned about off-screen.
"""

from __future__ import annotations

import signal
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult, Screen
from textual.containers import Container
from textual.message import Message
from textual.widgets import Button, DataTable, Header, Input, Static

from ..config import Config, PipelineConfig, db_titlepath
from ..pipeline.checkpoint import State
from ..pipeline.runner import PipelineRunner
from ..pipeline.events import ProgressEvent, StatEvent
from .widgets.progress_panel import ProgressPanel
from .widgets.stats_panel import StatsPanel
from .widgets.query_box import QueryBox, QueryRow, Submitted, build_rows, run_query


# --- Config modal (FR-1 + §3.7) -----------------------------------------------


class ConfigResult(Message):
    """Carries the confirmed start page + prompt path + output dir out of ``ConfigScreen``."""

    def __init__(self, start_page: str, prompt_path: str, output_dir: Path) -> None:
        super().__init__()
        self.start_page = start_page
        self.prompt_path = Path(prompt_path)
        self.output_dir = Path(output_dir)


class PromptGenerated(Message):
    """Result posted back from the generate-prompt worker onto the config modal."""

    def __init__(self, path: Path, ok: bool, message: str) -> None:
        super().__init__()
        self.path = path
        self.ok = ok
        self.message = message


class ConfigCancelled(Message):
    """Posted when the config modal is dismissed without confirmation (Esc / Cancel)."""


class Quit(Message):
    """Internal signal: run's signal handler posts this so the graceful stop can happen."""


class ConfigScreen(Screen):
    """A small modal that collects the start page, prompt path, and output directory.

    The heavy lifting (prompt rendering, project discovery) stays in the library; this screen
    only gathers the three strings and posts a :class:`ConfigResult` on confirmation. The output
    directory is validated to be an existing directory at submit time (a non-existent one shows
    an inline warning and does not submit).
    """

    def __init__(
        self,
        start_page: str = "",
        prompt_path: str = "",
        output_root: str = "",
        config: Optional["Config"] = None,
    ) -> None:
        super().__init__(id="config-screen")
        self._start_page = start_page
        self._prompt_path = prompt_path
        self._output_root = Path(output_root)
        self._config = config or Config()

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
        yield Static("Output directory (where this project's data lives):")
        yield Input(value=str(self._output_root), id="output-dir")
        yield Static("", id="output-warning")
        yield Static("Target topic (used to generate a gating prompt):")
        yield Input(id="target-topic", placeholder="e.g. autism acceptance")
        yield Static("Explicitly exclude:")
        yield Input(id="exclude-topic", placeholder="e.g. violence, code")
        yield Button("Generate", id="generate-prompt")
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
            self.app.pop_screen()
        elif button_id == "generate-prompt":
            self._on_generate_pressed()

    def _on_generate_pressed(self) -> None:  # pragma: no cover - needs live Ollama
        """Generate a gating prompt from the topic/exclude fields and write it to disk.

        Runs on a worker thread (network I/O to the gate model); the result is posted
        back onto this modal, where the notification is updated and the prompt-path
        field is prefilled when it was left empty. An empty target topic is a no-op
        with an inline warning.
        """
        topic = self.query_one("#target-topic", Input).value.strip()
        exclude = self.query_one("#exclude-topic", Input).value.strip()
        warn = self.query_one("#output-warning", Static)
        if not topic:
            warn.update("⚠  Target topic is required to generate a prompt")
            return

        def _generate_in_worker() -> None:
            from ..gating import prompt_gen

            output_dir = Path(self.query_one("#output-dir", Input).value.strip()).expanduser()
            dest = self._resolve_destination(
                self.query_one("#prompt-path", Input).value.strip(), output_dir
            )
            try:
                text = prompt_gen.generate_gating_prompt(
                    prompt_gen.DEFAULT_TEMPLATE, topic, exclude, config=self._config
                )
                if not text.strip():
                    warn.update("✗  Generation produced no prompt")
                    self.post_message(
                        PromptGenerated(dest, ok=False, message="Generation produced no prompt")
                    )
                    return
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(text, encoding="utf-8")
            except Exception as exc:  # pragma: no cover - live only
                self.post_message(PromptGenerated(dest, ok=False, message=f"Generation failed: {exc}"))
                return
            self.post_message(PromptGenerated(dest, ok=True, message=f"Prompt saved to {dest}"))

        self.app.run_worker(_generate_in_worker, thread=True, exclusive=True)

    def on_prompt_generated(self, message: PromptGenerated) -> None:  # pragma: no cover
        """Main-thread handler: notify the user and (if the field was empty) prefill it."""
        warn = self.query_one("#output-warning", Static)
        if message.ok:
            warn.update(f"✓  {message.message}")
            prompt_field = self.query_one("#prompt-path", Input)
            if not prompt_field.value.strip():
                prompt_field.value = str(message.path)
        else:
            warn.update(f"✗  {message.message}")

    def _resolve_destination(self, prompt_path: str, output_dir: Path) -> Path:
        """Where to write a generated prompt.

        If the Gating prompt path field is set, write there (overwriting).
        Otherwise fall back to ``<output-dir>/prompts/<topic-slug>.txt`` and let the
        caller prefill that field so the user can confirm the file exists.
        """
        if prompt_path.strip():
            return Path(prompt_path).expanduser()
        stem = db_titlepath(self.query_one("#target-topic", Input).value.strip() or "gate").removesuffix(".db")
        return output_dir / "prompts" / f"{stem}.txt"

    def _submit(self) -> None:  # pragma: no cover - Textual modal
        start_page = self.query_one("#start-page", Input).value.strip()
        prompt_path = self.query_one("#prompt-path", Input).value.strip()
        output_dir = self.query_one("#output-dir", Input).value.strip()
        if not start_page:
            return
        warn = self.query_one("#output-warning", Static)
        # Validate the output directory is an existing directory.
        out_path = Path(output_dir).expanduser()
        if not out_path.exists() or not out_path.is_dir():
            warn.update("⚠  Output directory does not exist")
            return
        # Expand ~ on the prompt path and require the file to exist, so a "~/…" path
        # like the one in the screenshot resolves instead of crashing the runner.
        prompt_p = Path(prompt_path).expanduser()
        if not prompt_p.exists() or not prompt_p.is_file():
            warn.update("⚠  Gating prompt file does not exist")
            return
        warn.update("")
        self.post_message(ConfigResult(start_page, str(prompt_p), out_path))
        self.app.pop_screen()


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
            self.app.pop_screen()


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
        # Raw config from the CONFIG modal (start page + prompt path + output dir),
        # set by CONFIG and consumed by RUN. None until the user has configured a run.
        self._config_start_page: Optional[str] = None
        self._config_prompt_path: Optional[str] = None
        self._config_output_dir: Optional[Path] = None
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
        # 8.2.8 has no `stylesheets` class attribute; load styles.tcss ourselves
        # relative to this module, so the palette (colours on the panels + footer)
        # actually applies (otherwise the progress/stats panels render blank).
        styles_path = Path(__file__).resolve().parent / "styles.tcss"
        self.stylesheet.read(styles_path)
        # The stylesheet docks #query-input left (60%) and #run-controls right, but
        # dock doesn't reserve space for the controls' fixed, natural-width buttons,
        # so they collide with the query box and the quit button overflows a narrow
        # terminal. Drop both dock rules and lay the footer out as a plain horizontal
        # flow — query box (flex: 1) takes the leftover width, controls sit beside it
        # at natural size. Colours still come entirely from the stylesheet.
        self.query_one("#query-input").styles.dock = None
        self.query_one("#query-input").styles.flex = (1, 0, 0)
        run_controls = self.query_one("#run-controls")
        run_controls.styles.dock = None
        run_controls.styles.layout = "horizontal"
        # The stats panel folds events off-thread but only repaints when asked; a
        # timer on the main thread drives that repaint (never touched from the worker
        # thread). The progress panel repaints itself on write, so this is the one
        # widget that would otherwise never re-render.
        self.set_interval(0.25, self._repaint_stats)

    def _repaint_stats(self) -> None:  # pragma: no cover - main-thread only
        panel = self.query_one("#stats-panel", StatsPanel)
        panel.maybe_repaint()

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
        # The stats panel folds the same ProgressEvent stream (crawl accept/reject,
        # split chunks) that the runner already emits as StatEvents; feeding it here is
        # what makes FR-4 live-update while the run crawls.
        self.query_one("#stats-panel", StatsPanel).feed(event)

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
        self.app.run_worker(self._drive, exclusive=True, thread=True)

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

    def resume_run(self) -> None:
        """Explicit RESUME (DESIGN §6): unpause a paused run. A no-op otherwise."""
        if self.runner is None or self._run_state != "paused":
            return
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
        base = labels[self._run_state]
        # Prefix the configured output directory so the bar matches §6's "project: …".
        out = self._config_output_dir if self._config_output_dir else "<no project>"
        indicator.content = f"project: {out}  {base}"
        # Colour the bar via a state class (styles.tcss: .running/.paused/.stopped).
        indicator.remove_class("running", "paused", "stopped")
        indicator.add_class(self._run_state)

    # --- run controls (footer) ------------------------------------------------

    def on_button_pressed(self, message: Button.Pressed) -> None:  # pragma: no cover
        button_id = message.button.id or ""
        if button_id == "run-button":
            if self.runner is not None and self._run_state in ("running", "paused"):
                return
            if self._config_start_page is None or self._config_output_dir is None:
                self.query_one("#progress-panel", ProgressPanel).add_event(
                    ProgressEvent(
                        stage="crawl",
                        kind="warn",
                        message="Run CONFIG first to set start page, prompt, and output dir",
                    )
                )
                return
            self._begin_run(self._make_pipeline())
        elif button_id == "config-button":
            self.push_screen(
                ConfigScreen(
                    start_page=self._config_start_page or "",
                    prompt_path=self._config_prompt_path or "",
                    output_root=str(self.root),
                    config=self.config,
                ),
                callback=self._on_config_result,
            )
        elif button_id == "resume-button":
            self.resume_run()
        elif button_id == "pause-button":
            self.toggle_pause()
        elif button_id == "stop-button":
            self.stop_run()
        elif button_id == "quit-button":
            self.exit()

    def _make_pipeline(self) -> PipelineConfig:
        """Build a PipelineConfig bound to the configured output directory.

        The output dir is captured at CONFIG time (CONFIG and the run parameters are
        now one step — the sidebar picker was removed), so RUN simply wires the three
        stored strings into the runner. Assumes ``self._config_output_dir`` and
        ``self._config_start_page`` were already checked by the caller.
        """
        assert self._config_output_dir is not None
        return PipelineConfig(
            start_page=self._config_start_page,
            prompt_path=self._config_prompt_path,
            output_dir=self._config_output_dir,
            ingest=True,
        )

    def _on_config_result(self, message: ConfigResult | ConfigCancelled) -> None:
        # CONFIG is an explicit, independent entry point that runs before a run begins,
        # so it stores the raw start page, prompt path, and output directory; RUN binds
        # them into a PipelineConfig. A Cancel keeps the current values; only a Confirm
        # replaces them.
        if isinstance(message, ConfigResult):
            self._config_start_page = message.start_page
            self._config_prompt_path = message.prompt_path
            self._config_output_dir = message.output_dir
            self._run_state = "idle"
            # Bind the configured output dir so checkpoints, artifacts, and the query
            # store all resolve against it.
            self.config = self._base_config.with_output_dir(message.output_dir)
            # Show the chosen project's checkpoint in the progress panel so the user can
            # see where a prior run stopped.
            state = State(message.output_dir / "pipeline_state.json")
            self.query_one("#progress-panel", ProgressPanel).set_state(state)
            self._refresh_status_bar()

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
        """Build the full-screen layout: header | status | progress + stats | footer.

        The output directory is chosen in the CONFIG modal (the sidebar picker was
        removed — Textual 8.2.8 has no file-browser widget), so the main region is
        just the progress window and the rolling stats panel. The footer's horizontal
        flow (query box + six run controls) is tuned in ``on_mount`` after the
        stylesheet loads, so the dock rules there don't crowd the natural-width buttons.
        """
        yield Header()
        yield Static("IDLE", id="status-indicator")
        yield Container(ProgressPanel(), StatsPanel(), id="main")

        yield Container(
            QueryBox(),
            Container(
                Button("Config", id="config-button"),
                Button("Run", id="run-button"),
                Button("Pause", id="pause-button"),
                Button("Resume", id="resume-button"),
                Button("Stop", id="stop-button"),
                Button("Quit", id="quit-button"),
                id="run-controls",
            ),
            id="footer",
        )
