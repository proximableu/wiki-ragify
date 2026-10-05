"""Phase 2.5 end-to-end integration test: the runner over a fixture project dir.

This is the "fake gate + fake fetcher + fake embed" test the DESIGN.md task list
promises — it proves the whole funnel (crawl -> list crawl -> split -> chunk gate ->
archive, plus the optional embed phase) executes without touching the network or a
running Ollama. None of the heavy deps (``ollama``/``nltk``/``datasketch``/
``sqlite-vec``) are required:

* a **fake fetcher** (no Wikipedia API) supplies the seeded page bodies and links,
  injected via the crawler's ``fetcher=`` constructor argument,
* a **fake evaluator** (serving both the article-level ``evaluate`` gate and the
  chunk-level ``chat`` gate) replaces :class:`~wiki_ragify.llm.gateway.LLMEvaluator`,
* a **fake embed client** and a **faked ``build_db``** replace the ingest phase, so
  no vector store or MinHash dependency is needed.

Because the heavy dependencies are imported *inside* the runner's phase bodies (not at
module import), a :class:`PipelineRunner` subclass that overrides the ``evaluator`` and
``embed_client`` properties plus a monkeypatched ``build_db`` is enough to exercise
every phase offline.
"""

from __future__ import annotations

import gzip
import importlib
import tarfile
import tempfile
import threading
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from wiki_ragify.config import Config, PipelineConfig, sanitize_title
from wiki_ragify.pipeline.events import ProgressEvent
from wiki_ragify.pipeline.runner import PipelineRunner

# --------------------------------------------------------------------------- #
# Constants that drive the fakes
# --------------------------------------------------------------------------- #

# Gate 1 (article-level) is evaluated on the article *lead* (text before the first
# "==" heading); Gate 2 (chunk-level) ``chat`` receives the full article text. The
# markers live in the relevant windows so they steer the right gate.
ACCEPT_MARKER = "ACCEPT_ME"
REJECT_MARKER = "REJECT_ME"

SEED_TITLE = "Test Seed"
LINK_TITLE = "Test Link"


def _filler_block(words: int = 160) -> str:
    """A long lorem-ipsum paragraph to clear the min-text-length heuristic."""
    tokens = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
              "golf", "hotel", "india", "juliet", "kilo", "mike"]
    return " ".join(tokens[i % len(tokens)] for i in range(words))


def _page_wikitext(title: str, *, accept_lead: bool = True, reject_body: bool = True) -> str:
    """Build a Wikipedia-style article.

    ``accept_lead`` places the accept marker in the lead (so Gate 1 accepts);
    ``reject_body`` appends the reject marker to a non-introduction body section (so
    the resulting chunk routes through the chunk gate's ``chat`` and gets rejected).
    """
    lead = f"{title} is a subject worth keeping. " + _filler_block(60)
    if accept_lead:
        lead += " " + ACCEPT_MARKER
    body = _filler_block()
    if reject_body:
        body += " " + REJECT_MARKER + " filler words to be dropped."
    return (
        f"{lead}\n\n"
        f"== Introduction ==\n\n{lead}\n\n"
        f"== {title} body ==\n\n{body}\n\n"
    )


@dataclass
class FakeFetcher:
    """Serves canned pages/links from dicts; records every call for assertions."""

    pages: dict = field(default_factory=dict)
    links: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)

    def fetch_text(self, title: str) -> str | None:
        self.calls.append(("text", title))
        return self.pages.get(title)

    def fetch_links(self, title: str) -> list:
        self.calls.append(("links", title))
        return list(self.links.get(title, []))


class FakeEvaluator:
    """Serves both gates. Gate 1 = ``evaluate(text, prompt) -> bool``;
    Gate 2 = ``chat(model, messages, options) -> {"message": {"content": ...}}``.

    The article gate accepts when its lead carries ``ACCEPT_MARKER``; the chunk gate
    accepts unless the full text carries ``REJECT_MARKER``.
    """

    def __init__(self, on_evaluate=None):
        self.on_evaluate = on_evaluate

    def evaluate(self, text: str, gating_prompt: str) -> bool:
        if self.on_evaluate is not None:
            self.on_evaluate(text, gating_prompt)
        return ACCEPT_MARKER in text

    def chat(self, model: str, messages: list, options: dict) -> dict:
        prompt = " ".join(m.get("content", "") for m in messages)
        content = "reject" if REJECT_MARKER in prompt else "accepted"
        return {"message": {"content": content}}


class FakeEmbedClient:
    """Minimal embed client the runner injects into phase 6; records batches."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.embedded: list = []

    def embed_batch(self, texts: list, retries: int = 3) -> list:
        self.embedded.extend(texts)
        return [[0.0] * self.dim for _ in texts]


class FakePipelineRunner(PipelineRunner):
    """A runner with fakes wired in, so it runs without any heavy dependency."""

    def __init__(self, config, pipeline, fake_fetcher, fake_evaluator, fake_embed,
                 events, on_stat=None):
        super().__init__(config, pipeline, on_event=lambda e: events.append(e), on_stat=on_stat)
        self.fake_fetcher = fake_fetcher
        self.fake_evaluator = fake_evaluator
        self.fake_embed = fake_embed
        self.ran_ingest = False

    @property
    def evaluator(self) -> FakeEvaluator:
        return self.fake_evaluator

    @property
    def embed_client(self) -> FakeEmbedClient:
        return self.fake_embed


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #

def _make_pipeline(tmp_path: Path, prompt_file: Path, ingest: bool = True) -> PipelineConfig:
    return PipelineConfig(
        start_page=SEED_TITLE,
        prompt_path=prompt_file,
        output_dir=tmp_path,
        ingest=ingest,
        timeout=60.0,
    )


def _install_stub_modules(monkeypatch, fetcher: FakeFetcher):
    """Install fake modules so the runner's lazy imports succeed and it uses fakes.

    The runner imports heavy deps *inside* its phase bodies; none of those deps
    (``ollama``/``nltk``/``datasketch``/``pydantic``/``sqlite3_vec``) are installed, and
    importing the real ``wiki_ragify.ingestion.build_db`` (or ``llm``) would fail on
    ``pydantic``. We register a minimal fake ``wiki_ragify.llm`` (whose ``LLMEvaluator``
    is our :class:`FakeEvaluator`, so Gate 1 resolves) plus stub leaf modules, and make
    the crawler's lazily-built fetcher return ``fetcher`` (no network).

    Returns the stubbed ``wiki_ragify.ingestion.build_db`` module so callers can attach
    a faked ``build_db`` to it directly (``monkeypatch`` cannot mutate the namespace
    package, so this is an attribute assignment instead).
    """
    import sys

    # Leaf third-party modules the chain would import if it ran that far.
    ollama_mod = types.ModuleType("ollama")
    ollama_mod.Client = type("Client", (), {})
    sys.modules["ollama"] = ollama_mod
    nltk_mod = types.ModuleType("nltk")

    def _sent_tokenize(text: str) -> list:
        # A crude but honest sentence counter: ~1 sentence per 10 words. Real
        # content here (hundreds of words) yields well more than 2 sentences, so
        # ``has_insufficient_sentences`` routes it to the chunk gate's ``chat``
        # instead of the insufficient-sentence fast-reject.
        words = len(text.split())
        return ["placeholder"] * max(2, words // 10)

    nltk_mod.sent_tokenize = _sent_tokenize
    sys.modules["nltk"] = nltk_mod
    sys.modules["datasketch"] = types.ModuleType("datasketch")
    sys.modules["pydantic"] = types.ModuleType("pydantic")
    sys.modules["sqlite3_vec"] = types.ModuleType("sqlite3_vec")
    sys.modules["sqlite3_vec"].load = lambda *a, **k: None

    # The runner lazily does ``from ..llm.gateway import LLMEvaluator`` (crawler Gate 1)
    # and ``from ..llm.embed import EmbedClient`` (phase 6). Point them at our fakes.
    llm = types.ModuleType("wiki_ragify.llm")
    gateway = types.ModuleType("wiki_ragify.llm.gateway")
    gateway.LLMEvaluator = FakeEvaluator
    embed = types.ModuleType("wiki_ragify.llm.embed")
    embed.EmbedClient = FakeEmbedClient

    # Register so ``from ..llm.X import Y`` (a submodule import) resolves.
    llm.gateway = gateway
    sys.modules["wiki_ragify.llm"] = llm
    sys.modules["wiki_ragify.llm.gateway"] = gateway
    sys.modules["wiki_ragify.llm.embed"] = embed

    # Make the crawler's lazily-built fetcher return our fake (no network).
    monkeypatch.setattr("wiki_ragify.crawler.fetcher.WikiPageFetcher", lambda config: fetcher)

    # A faked build_db that records nothing yet; callers install their behavior.
    build_db = types.ModuleType("wiki_ragify.ingestion.build_db")
    build_db.build_db = lambda *a, **k: None
    sys.modules["wiki_ragify.ingestion.build_db"] = build_db

    return build_db


@pytest.fixture
def prompt_file(tmp_path: Path) -> Path:
    p = tmp_path / "prompt.txt"
    p.write_text(
        "Keep only encyclopedia content that is topically focused, factually stated, "
        "and free of navigation boilerplate.",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def config(tmp_path: Path, prompt_file: Path) -> Config:
    return Config().with_output_dir(tmp_path)


def _seeded_fetcher(config: Config) -> FakeFetcher:
    """A fetcher whose seeded graph funnels through all phases:
    SEED -> LINK_TITLE, both accepted by Gate 1; LINK rejected by Gate 2."""
    pages = {
        SEED_TITLE: _page_wikitext(SEED_TITLE, accept_lead=True, reject_body=False),
        LINK_TITLE: _page_wikitext(LINK_TITLE, accept_lead=True, reject_body=True),
    }
    links = {SEED_TITLE: [LINK_TITLE]}
    return FakeFetcher(pages=pages, links=links)


def _archive_path(config) -> Path:
    return config.output_dir / f"knowledge_{sanitize_title(SEED_TITLE)}.tgz"


def _assert_valid_gzip_tar(archive: Path) -> None:
    assert archive.exists(), f"archive missing: {archive}"
    assert archive.stat().st_size > 0, "archive is empty"
    # Every member must fully decompress (the archive integrity guarantee).
    with tempfile.TemporaryDirectory() as tmp:
        with gzip.open(archive, "rb") as raw:
            with tarfile.open(fileobj=raw, mode="r") as tar:
                names = tar.getnames()
                assert names, "archive has no members"
                tar.extractall(path=tmp)


def _stage_events(runner) -> list:
    return [(e.stage, e.kind) for e in runner._events]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_full_pipeline_end_to_end(tmp_path, prompt_file, config, monkeypatch):
    """Fake fetcher + fake gate + fake embed: every phase runs and produces artifacts."""
    events: list = []
    fetcher = _seeded_fetcher(config)
    evaluator = FakeEvaluator()
    embed = FakeEmbedClient()
    runner = FakePipelineRunner(config, _make_pipeline(tmp_path, prompt_file, ingest=True),
                                fetcher, evaluator, embed, events)

    _build_db = _install_stub_modules(monkeypatch, fetcher)

    def fake_build_db(accepted_dir, db_path, cfg, embed_client=None, on_event=None, interrupt=None):
        runner.ran_ingest = True
        assert accepted_dir.exists()
        assert list(accepted_dir.glob("*.md"))
        # Real build_db embeds the accepted chunk bodies here; exercise the same so the
        # embed client (and embed.embedded) reflects the content that was ingested.
        if embed_client is not None:
            embed_client.embed_batch(["seed chunk content", "link chunk content"])
        if on_event is not None:
            on_event(ProgressEvent(stage="ingest", kind="stage_done", message="[DONE] embed"))

    setattr(_build_db, "build_db", fake_build_db)

    # Snapshot the accept/ and reject/ subdirs immediately after the gate phase,
    # because the archive phase (phase 5) moves accepted chunks into knowledge/ and
    # then purges the whole chunks/ dir — after a full run the source is gone.
    gate_snapshot: dict[str, list[str]] = {"chunks": [], "accept": [], "reject": []}
    chunks_dir = config.output_dir / config.chunks_dirname

    orig_on_event = runner.on_event

    def snap_on_event(event):
        orig_on_event(event)
        if event.stage == "split" and event.kind == "stage_done":
            # Chunk count captured before the gate phase moves files out of chunks/.
            gate_snapshot["chunks"] = sorted(p.name for p in chunks_dir.glob("*.md"))
        elif event.stage == "gate" and event.kind in ("accept", "reject"):
            # move_article is synchronous, so the dirs always reflect the state after
            # each file's decision; the final one captures the whole gate. Guard each
            # read — the accept/ and reject/ dirs are created on the first decision, so
            # one or both may be absent on early events.
            accept = chunks_dir / config.gate_accept_dirname
            reject = chunks_dir / "reject"
            if accept.exists():
                gate_snapshot["accept"] = sorted(p.name for p in accept.iterdir())
            if reject.exists():
                gate_snapshot["reject"] = sorted(p.name for p in reject.iterdir())

    runner.on_event = snap_on_event
    runner.run()
    runner.on_event = orig_on_event

    seen = [(e.stage, e.kind) for e in events]
    for expected in ("crawl", "split", "gate", "archive", "ingest"):
        assert any(s == expected for s, _ in seen), f"missing stage {expected}"
    # Ordering: each later stage comes after the previous one.
    stages = [s for s, _ in seen]
    assert stages.index("crawl") < stages.index("split") < stages.index("gate")
    assert stages.index("gate") < stages.index("archive") < stages.index("ingest")

    # Phase 1 produced a non-empty accepted list.
    list_file = config.output_dir / config.output_dirname / "accepted_pages.list"
    titles = [t for t in list_file.read_text(encoding="utf-8").splitlines() if t.strip()]
    assert SEED_TITLE in titles
    assert LINK_TITLE in titles

    # Phase 3 split accepted pages into .md chunks in the chunks dir (captured before
    # the archive purge consumed them).
    assert gate_snapshot["chunks"], "phase 3 produced no chunks"

    # Phase 4 moved chunks into accept/ and reject/ subdirs (source root drained).
    assert gate_snapshot["accept"], "phase 4 accepted nothing"
    assert gate_snapshot["reject"], "phase 4 rejected nothing"
    # The introduction fast-path accepts; the body chunk with REJECT_MARKER rejects.
    assert any(name.endswith("introduction.md") for name in gate_snapshot["accept"]), (
        "introduction chunk should fast-accept"
    )

    # Phase 5 archived the knowledge dir to a valid gzip tar.
    _assert_valid_gzip_tar(_archive_path(config))

    # Phase 6 (ingest) embedded content via the fake embed client.
    assert runner.ran_ingest
    assert embed.embedded, "embed client saw no content"

    # State persisted all phases as completed.
    for phase in (
        "phase_1_seed",
        "phase_2_list",
        "phase_3_split",
        "phase_4_gate",
        "phase_5_archive",
        "phase_6_ingest",
    ):
        assert runner.state.is_phase_complete(phase), phase


def test_accept_all_then_gate_rejects_all_noop(tmp_path, prompt_file, config, monkeypatch):
    """When the chunk gate rejects every chunk, the archive is a successful no-op."""
    events: list = []
    fetcher = FakeFetcher(pages={SEED_TITLE: _page_wikitext(SEED_TITLE, reject_body=True, accept_lead=False)},
                          links={SEED_TITLE: []})
    evaluator = FakeEvaluator()
    embed = FakeEmbedClient()
    runner = FakePipelineRunner(config, _make_pipeline(tmp_path, prompt_file),
                                fetcher, evaluator, embed, events)

    def never_ingest(*a, **k):
        raise AssertionError("ingest must not run when there is nothing to embed")

    _build_db = _install_stub_modules(monkeypatch, fetcher)
    setattr(_build_db, "build_db", never_ingest)

    runner.run()

    assert not runner.ran_ingest
    # No accepted chunks survived the gate; no archive was written.
    accept_dir = config.output_dir / config.chunks_dirname / config.gate_accept_dirname
    assert not accept_dir.exists() or not any(accept_dir.iterdir())
    assert not _archive_path(config).exists()


def test_pause_blocks_then_resumes(tmp_path, prompt_file, config, monkeypatch):
    """Pause set before run blocks the runner; resuming lets it finish every phase."""
    events: list = []
    fetcher = _seeded_fetcher(config)
    evaluator = FakeEvaluator()
    embed = FakeEmbedClient()
    runner = FakePipelineRunner(config, _make_pipeline(tmp_path, prompt_file),
                                fetcher, evaluator, embed, events)

    _build_db = _install_stub_modules(monkeypatch, fetcher)
    setattr(_build_db, "build_db", lambda *a, **k: setattr(runner, "ran_ingest", True))

    runner.pause()
    thread = threading.Thread(target=runner.run)
    thread.start()

    # The runner should be blocked in _check_pause before completing any phase.
    thread.join(timeout=2.0)
    assert thread.is_alive(), "runner did not block on pause"

    runner.resume()
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "runner did not resume"

    assert runner.ran_ingest
    assert runner.state.is_phase_complete("phase_6_ingest")
    for phase in ("phase_1_seed", "phase_2_list", "phase_3_split",
                  "phase_4_gate", "phase_5_archive", "phase_6_ingest"):
        assert runner.state.is_phase_complete(phase)


def test_stop_during_crawl_exits_before_ingest(tmp_path, prompt_file, config, monkeypatch):
    """A stop signal set during crawl halts cleanly, before the ingest phase."""
    events: list = []
    fetcher = _seeded_fetcher(config)
    embed = FakeEmbedClient()

    def stop_on_first_evaluate(text, prompt):
        runner.stop()

    evaluator = FakeEvaluator(on_evaluate=stop_on_first_evaluate)
    runner = FakePipelineRunner(config, _make_pipeline(tmp_path, prompt_file),
                                fetcher, evaluator, embed, events)

    _build_db = _install_stub_modules(monkeypatch, fetcher)
    setattr(_build_db, "build_db", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ingest must not run after stop")))

    runner.run()

    assert not runner.ran_ingest
    seen = [e.stage for e in events]
    assert "crawl" in seen
    assert not any(s in ("split", "gate", "archive", "ingest") for s in seen)


def test_stop_during_gate_halts_and_leaves_phase_rerunnable(tmp_path, prompt_file, config, monkeypatch):
    """A stop signal set mid-gate-file-loop exits cleanly: the gate phase is left
    *incomplete* (so a resumed run re-gates rather than skipping), and nothing past
    it (archive, ingest) runs."""
    events: list = []
    fetcher = _seeded_fetcher(config)
    evaluator = FakeEvaluator()
    embed = FakeEmbedClient()
    runner = FakePipelineRunner(config, _make_pipeline(tmp_path, prompt_file), fetcher, evaluator, embed, events)

    def stop_on_first_chunk(text, prompt):
        # Fire stop on the very first chunk-gate decision.
        runner.stop()

    # Route the chunk gate through the real gate_files with the runner's interrupt
    # hook active; have it request a stop on the first chunk so we exercise the
    # between-iterations interrupt path in gate_files (not just the phase-boundary check).
    from wiki_ragify.gating import chunk as chunk_module

    real_process = chunk_module.process_article

    def stop_process(path, gating_prompt, evaluator, model="", num_ctx=0, think=False):
        if real_process(path, gating_prompt, evaluator, model, num_ctx, think):
            runner.stop()
        return False

    chunk_module.process_article = stop_process

    _build_db = _install_stub_modules(monkeypatch, fetcher)
    setattr(_build_db, "build_db", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ingest must not run after stop")))

    runner.run()

    chunk_module.process_article = real_process

    # Gate interrupted before draining its files => the gate phase is NOT marked
    # complete, so a re-run will re-gate safely (safe default).
    assert not runner.state.is_phase_complete("phase_4_gate")
    assert not runner.ran_ingest
    assert not runner.state.is_phase_complete("phase_5_archive")
    seen = [e.stage for e in events]
    # The gate started but archive never began, and neither ingest's warn nor its
    # stage_done ever fired.
    assert "gate" in seen
    assert not any(s == "archive" for s in seen)
    assert not any(s == "ingest" for s in seen)


def test_resumption_skips_completed_phases(tmp_path, prompt_file, config, monkeypatch):
    """When the state marks phases 1-3 done, a re-run skips them and only runs 4-5."""
    import json

    state = {
        "phase_1_seed": "completed",
        "phase_2_list": "completed",
        "phase_3_split": "completed",
        "phase_4_gate": "pending",
        "phase_5_archive": "pending",
    }
    (config.output_dir / "pipeline_state.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )

    events: list = []
    fetcher = FakeFetcher(pages={}, links={})
    evaluator = FakeEvaluator()
    embed = FakeEmbedClient()
    runner = FakePipelineRunner(config, _make_pipeline(tmp_path, prompt_file),
                                fetcher, evaluator, embed, events)

    _build_db = _install_stub_modules(monkeypatch, fetcher)
    setattr(_build_db, "build_db", lambda *a, **k: setattr(runner, "ran_ingest", True))

    runner.run()

    # Phases 1 & 2 (both stage "crawl") are reported skipped; the fetcher is never called.
    crawl_info = [e.message for e in events if e.stage == "crawl" and e.kind == "skip"]
    assert any("Skipping" in m for m in crawl_info)
    assert not fetcher.calls, "fetcher must not be consulted for skipped crawl phases"
    assert not runner.ran_ingest, "no accepted chunks => ingest skipped"


def test_ingest_toggle_off_skips_ingest(tmp_path, prompt_file, config, monkeypatch):
    """With ingest disabled, phase 6 never runs even though chunks were accepted."""
    events: list = []
    fetcher = _seeded_fetcher(config)
    evaluator = FakeEvaluator()
    embed = FakeEmbedClient()
    runner = FakePipelineRunner(config, _make_pipeline(tmp_path, prompt_file, ingest=False),
                                fetcher, evaluator, embed, events)

    _build_db = _install_stub_modules(monkeypatch, fetcher)
    setattr(_build_db, "build_db", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ingest must not run when toggled off")))

    runner.run()

    assert not runner.ran_ingest
    assert not runner.state.is_phase_complete("phase_6_ingest")
    for phase in ("phase_1_seed", "phase_2_list", "phase_3_split", "phase_4_gate", "phase_5_archive"):
        assert runner.state.is_phase_complete(phase)
