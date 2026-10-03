# Wiki Ragify v2 — TUI Design & Requirements

> Documentation-only. This is the spec for a **contained, homogenous Python package**
> that runs the whole wiki → knowledge base funnel from inside a single **Textual** TUI.
> The 5 scripts in the *old* working tree (`wiki_crawler.py`, `wiki_splitter.py`,
> `chunk_gate.py`, `build_db_vec.py`, `retriever.py`, plus `ollama_gateway.py`) are the
> **reference implementation** — we copy their algorithms verbatim into clean modules and
> rewrite the glue (orchestration + UI) from scratch.

---

## 0. The One-Sentence Goal

A single Python package where a Textual TUI browses a target `output/` folder tree,
then drives crawl → split → gate → archive → embed as **in-process library calls**,
showing a live progress window and rolling statistics, with **pause/resume at any stage**
and no reliance on subprocess glue or a wall of 5 separate CLIs.

---

## 1. The System We're Building (the funnel)

```
        ┌─ Browse output/ folder tree (which project) ─┐
        │                                                │
   OutputDir/                                          │
   ├── project_A/                                       │
   │   ├── output/*.txt                     Gate 1     │  crawl seed(s)
   │   │                       (LLM qwen3-30b)          │  → .txt articles
   │   ├── chunks/*.md      Gate 2                     │
   │   │                     (LLM qwen3-30b)            │  split into chunks
   │   ├── chunks/accept/*.md  →  knowledge/*.md        │
   │   ├── knowledge/  +  knowledge.tgz (integrity)     │  archive accepted
   │   └── *.sqlite      (minhash dedup + embed)         │  ingest into DB
   └── project_B/ …                                             │
        └────────────────────────────────────────────────────────┘
                                                        retrieve / query
                                                        (ANN + relevance filter)
```

The existing orchestrator (`pipeline_orchestrator.py`, present and working in this dir) already
chains the funnel as **5 subprocess stages** — and carries hard-won fixes we re-use: dual-layer
checkpointing (JSON state **and** filesystem artifact validation), a per-command **6h timeout**,
a **flock** lock so two runs can't race, **full-stream archive integrity verification**, a
`sanitize_title` SHA-256 collision guard, `--dry-run` support, and an **empty-input graceful
no-op** on archive. We port *all of that logic* into the new in-process runner (as `pipeline/runner.py`
and `pipeline/archive.py`) and drop the subprocess plumbing.

Two LLM gates, each using **structured JSON output** (Pydantic schema enforced natively by
Ollama, no manual parsing):

- **Gate 1 — article level:** crawler accepts/rejects whole `.txt` pages by their intro
  paragraph. Keeps a `download_cache.json` (accepted→filename, rejected titles) so it is
  resumable and idempotent.
- **Gate 2 — chunk level:** splits `.txt` into section `.md` chunks, then re-gates each
  chunk into `accept/` / `reject/`. Fast-path bypasses: `_introduction` → accept; < 2
  sentences → reject.

Then archive → embed → store, and the **retriever** answers queries against the store.

---

## 2. Design Principles (non-negotiable)

1. **One package, not five scripts.** Everything is importable modules in a package
   `wiki_ragify/`. No script shells out to another script as its primary wiring.
   (The old scripts *can* still run standalone, but the TUI talks to the *library*.)
2. **In-process pipeline, not subprocess soup.** The pipeline is a library object
   (`PipelineRunner`) whose methods (`crawl`, `split`, `gate_chunks`, `archive`, `ingest`)
   return **progress callbacks / streamed events**. The TUI binds directly to those events.
   Blocking network/LLM work runs on a `concurrent.futures.ThreadPoolExecutor` (Ollama is
   synchronous HTTP; this keeps the UI event loop responsive — Textual is async, so we
   offload, we don't `run_in_executor` per-call churn).
3. **Copy algorithms, rewrite interfaces.** `MinHashLSH(threshold=0.98)` for ingest,
   `0.65` for retrieval; word-3-gram shingles; `num_perm` knobs; Pydantic gate schemas;
   `tiktoken`-aware vs char fallback in the splitter — **verbatim logic, new home.**
4. **Config lives in `.env`, nothing else.** Every tunable already in `.env.example` keeps
   its name; the package has a single `Config` object that reads `.env` via `python-dotenv`
   (exactly like the old `ollama_gateway.PipelineConfig`). No hardcoded URLs.
5. **Structured outputs everywhere.** Every LLM use is a Pydantic `BaseModel` +
   `format=<schema>`. Fail-safe routing: retry w/ exponential backoff, then a documented
   safe default (reject for gates; highest-score fallback for retrieval).
6. **Pause / resume is a first-class state machine**, not an afterthought. Every stage
   persists a checkpoint the moment it makes durable progress; the TUI can pause, quit, or
   be relaunched and resume from exactly where it stopped.
7. **Write tests as we go.** `pytest`. The old code had zero tests — v2 ships with them.
   Pure functions (filename sanitizing, section detection, eta math, cache round-trips,
   checkpoint load/save) are unit-tested; LLM-touching functions are tested with a fake
   gate client.

---

## 3. Package Layout

```
wiki_ragify/                      # the new package (new git repo in the new working tree)
├── __init__.py
├── __main__.py                   # python -m wiki_ragify  → launches the TUI
├── config.py                     # Config(dataclass) reading .env; single source of truth
├── logging_setup.py              # logger per module, like the old logging.getLogger(...)
│
├── llm/                          # Ollama surface — ONE place that talks to the gateway
│   ├── __init__.py
│   ├── gateway.py                # LLMEvaluator + GatingResponse (moved verbatim)
│   ├── embed.py                  # batch embed + embed_query, with retry/backoff
│   └── models.py                 # GatingResponse, RelevanceFilter pydantic schemas
│
├── crawler/                      # Gate 1
│   ├── __init__.py
│   ├── fetcher.py                # WikiPageFetcher (text + links, pagination)
│   ├── processor.py              # title_to_filename, extract_first_paragraph, is_bad_page
│   ├── cache.py                  # Cache (load/save download_cache.json)
│   └── runner.py                 # WikiCrawler.crawl() → emits (total, index, title, decision) events
│
├── splitting/
│   ├── __init__.py
│   └── splitter.py               # find_sections + chunk_section (verbatim)
│
├── gating/                       # Gate 2
│   ├── __init__.py
│   └── chunk_gate.py             # has_insufficient_sentences, read_full_content, move_*
│
├── ingestion/
│   ├── __init__.py
│   ├── dedup.py                  # _tokenize (shingles) + MinHashLSH helpers
│   └── build_db.py               # read → dedup → batch embed → upsert SQLite (verbatim)
│
├── retrieval/
│   ├── __init__.py
│   ├── store.py                  # load_db (sqlite + sqlite-vec), schema
│   ├── search.py                 # search_top_k, deduplicate_passages, embed_query
│   └── filter.py                 # filter_relevant_passages
│
├── pipeline/                     # THE orchestrator, now a library object
│   ├── __init__.py
│   ├── events.py                 # ProgressEvent / StatEvent dataclasses (the contract)
│   ├── runner.py                 # PipelineRunner: crawl/split/gate/archive/ingest in-process
│   ├── checkpoint.py             # State: load/save resume JSON, per-stage status
│   └── archive.py                # copy + gzip-tar with full-stream integrity check
│
└── ui/                           # Textual app
    ├── __init__.py
    ├── app.py                    # WikiRagifyApp: screens, comms to runner
    ├── widgets/
    │   ├── __init__.py
    │   ├── folder_tree.py        # browseable output/ folder picker
    │   ├── progress_panel.py     # per-stage progress window + log
    │   ├── stats_panel.py        # rolling statistics (rates, accepted/rejected, totals)
    │   └── query_box.py          # query → top-k passages
    └── styles.tcss               # Textual CSS for the layout
```

Why: this is the shape that lets *one program* own the funnel. `crawler.runner` etc. each
**emit events** the UI binds to; `pipeline.runner` wires them together and owns checkpoint
+ pause. The UI never parses stdout.

---

## 4. Requirements

### 4.1 Functional

- **FR-1 Browse folder.** The app presents a tree of a chosen root `output/` directory
  (default `~/wiki`, or whatever the user picks) so they can navigate into a project, and
  confirm the output directory the run will use. Selecting a folder highlights its stage
  state from the persisted checkpoint.
- **FR-2 Configure a run.** From a config dialog the user sets: start page(s) or
  `--start-list` file, gating-prompt file, target output/project dir, language, and
  optional overrides (max tokens, min text length, batch size). All values fall back to
  `.env`.
- **FR-3 Drive the funnel in-process.** Pressing run starts `PipelineRunner`:
  1. crawl (seed + depth-1 links, gate 1),
  2. *(expand option)* crawl `accepted_pages.list` again (gate 1),
  3. split accepted `.txt` → `.md` chunks,
  4. gate 2 chunks → `accept/` / `reject/`,
  5. archive `accept/` → `knowledge/` + `knowledge.tgz` (integrity-checked),
  6. *(optional — toggle in the config dialog)* ingest → dedup → embed → SQLite store.
     Stopped at archive leaves a complete, archived knowledge base you can re-ingest later
     from its checkpoint without re-crawling.
  Each stage streams progress back to the UI as it goes.
- **FR-4 Live progress window.** A panel per stage shows: stage name, `current/total`,
  a progress bar, rate (`items/s`), ETA, and a live log of decisions (ACCEPT/REJECT per
  item, with title), coloured like the old `chunk_gate.py` output but as list entries.
- **FR-5 Rolling statistics.** A panel that accumulates and re-renders: pages crawled,
  accepted/rejected (gate 1), chunks split, accepted/rejected (gate 2), embeddings produced,
  dedups removed, DB rows, wall-clock elapsed, and per-stage timing. Updates on every
  event, throttled to avoid redundant repaints.
- **FR-6 Pause / stop / resume.**
  - **Pause** a running stage (e.g. the LLM gateway is busy elsewhere); resume continues
    from the same checkpoint.
  - **Stop/quit** persists the checkpoint at the last durable boundary and exits cleanly
    (handle SIGINT/SIGTERM in the runner, like the old code).
  - **Relaunch** reloads the checkpoint and offers "Resume from stage N". Stages already
    `"completed"` are skipped; `"failed"`/`"pending"`/missing re-run.
- **FR-7 Archive integrity.** On archive, the `.tgz` is validated by reading every
  member's full decompressed data (a truncated archive must fail, as in the old fix).
- **FR-8 Retrieve.** A query box lets the user run a natural-language query against the
  built store and see top-k passages (ANN + retrieval dedup + relevance-filtered), with
  source filenames.

### 4.2 Non-functional

- **NF-1 Responsiveness.** The Textual event loop never blocks; all Ollama/network runs
  on a bounded worker thread pool. UI stays interactive during multi-hour crawls.
- **NF-2 Determinism where it matters.** Gate/retrieval LLM calls use `temperature=0.0`
  (and `seed=42` for the relevance filter, as today) so reruns are stable.
- **NF-3 Fail-safe defaults.** Any unrecoverable LLM failure → reject (gates) /
  highest-score fallback (retrieval). Never crash the run.
- **NF-4 Rate-limiting.** HTTP requests to Wikipedia honour `RATE_LIMIT_DELAY` / `REQUEST_TIMEOUT`
  from `.env` (verbatim).
- **NF-5 One lock.** A run acquires a flock lockfile (`.pipeline.lock`) in the output dir so
  two TUIs can't race (the old fix — re-adopt).
- **NF-6 Config-driven.** No tunable in code that isn't in `.env` + `Config`.
- **NF-7 Testable.** Pure logic is unit-tested; LLM I/O is behind a `GateClient`/`EmbedClient`
  protocol so tests inject fakes.

---

## 5. The Event Contract (how the library talks to the UI)

The library emits typed events; the UI consumes them. This is the single contract that
decouples everything.

```python
# pipeline/events.py
@dataclass
class ProgressEvent:
    stage: str                 # "crawl" | "expand" | "split" | "gate_chunks" | "archive" | "ingest"
    index: int                 # 0-based current item
    total: int | None          # None if unknown
    kind: str                  # "tick" | "accept" | "reject" | "skip" | "info" | "stage_done"
    message: str
    ts: float

@dataclass
class StatEvent:
    metric: str                # "pages_crawled", "accepted", "rejected", "chunks", ...
    value: int                 # cumulative
    ts: float
```

- `crawler.runner`, `gating.chunk_gate`, `ingestion.build_db` each accept an optional
  `on_event: Callable[[ProgressEvent], None]` callback (and a `stats` object).
- The UI subscribes; `PipelineRunner` fans the callback to the event loop safely.
- Because Textual is async, `PipelineRunner` exposes `async def run(...)`; blocking
  work runs via `asyncio.get_event_loop().run_in_executor(pool, sync_fn)`.

This replaces the old "subprocess + stdout parsing" model entirely.

---

## 6. Textual UI Layout

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ HEADER  Wiki Ragify v2        [● RUNNING] [❚❚ PAUSE] [■ STOP]    project: …/project_A │
├──────────────────────┬──────────────────────────────┬───────────────────────────┤
│ SIDEBAR              │ PROGRESS WINDOW               │ ROLLING STATISTICS        │
│ 📁 output/           │ Stage 1 · crawl               │ pages crawled   128        │
│   📂 project_A        │ ██████████░░░░░░░  67%        │ accepted     97            │
│     📂 project_B      │ [42/63] ACCEPT  Artificial…   │ rejected      31           │
│       output/         │ [42/63] REJECT  List of …     │ chunks       1204         │
│       chunks/         │ [ 9/63] ACCEPT  Biology…      │ gate2_accept   982         │
│       knowledge/      │ …                              │ gate2_reject   222        │
│       artifacts.sqlite│ ETA  3m 12s                    │ embeddings   1204         │
│                       │                                │ dedup_removed  0           │
│                       │ ──────────────────────────────│ elapsed    14m 03s        │
│                       │  [live log, scrollable]        │ crawl_rate   0.31/s       │
├──────────────────────┴──────────────────────────────┴───────────────────────────┤
│ CONFIG │ RESUME │ PAUSE/RESUME │ STOP │ QUIT  │  query: [_text_________________] [▶] │
└───────────────────────────────────────────────────────────────────────────────────┘
```

- **Sidebar** = folder picker (a `TreeView`/custom tree over the chosen root). Clicking a
  folder loads its checkpoint into the progress panel so you can see where it stopped and
  resume.
- **Progress window** = the live log panel. Each decision is a row (coloured accept/reject/
  skip), newest appended; shows the same `[N/total] ACCEPT title` rhythm as old `chunk_gate.py`.
- **Statistics panel** = rolling metrics, throttled repaint.
- **Query box** in the footer opens a results overlay when it returns passages.
- **Pause/stop controls** map to the runner's pause/stop primitives.

Textual specifics to remember: app is `textual.app.App`; screens for config/query;
`Table`/`Static`/`ListView` for panels; CSS in `styles.tcss`; heavy blocking work via
`self.run_worker(coro, exclusive=...)` and `ThreadPoolExecutor`.

---

## 7. Pause / Resume Model

- **Checkpoint schema** (`checkpoint.py`, keeps the orchestrator's `phase_N_xxx` schema; the
  legacy `steps: {crawl_seed, crawl_list, split, gate_chunks, archive}` shape from the old
  `pipeline_state.json` is normalized onto it on load):
  ```json
  {
    "project_dir": "/abs/path/project_A",
    "start_page": "Artificial intelligence",
    "expand": false,
    "current_step": "phase_4_gate",
    "phases": {
      "phase_1_seed":  "completed",
      "phase_2_list":  "failed",
      "phase_3_split": "pending",
      "phase_4_gate":  "in_progress",
      "phase_5_archive": "pending"
    }
  }
  ```
- A phase flips to `completed` **only after** its durable side effect is flushed
  (e.g. crawl: cache written + `accepted_pages.list` written; split: chunks on disk;
  ingest: committed + files moved to `embedded/`).
- **In_progress** survives a crash; on relaunch it re-runs and continues.
- **Pause** = runner signals a `threading.Event`; each gate/crawl iteration checks it and
  yields, keeping the checkpoint current. **Stop/quit** = set a stop flag, checkpoint at the
  next boundary, exit.
- Phase 5 (archive) re-uses the "check artifact file exists + non-empty, else graceful
  no-op" fix, and the archive integrity check.

---

## 8. Configuration (.env — keep all existing names)

Same keys as `.env.example`; the new `Config` reads them once. New tunables we may add:
`MAX_BATCH` (embedding batch size, default 16), `TOP_K` (retrieval, default 16),
`MIN_PASSEGES` (relevance floor, default 16). Everything stays in `.env`, nothing hardcoded.

---

## 9. Task List (ordered, each item testable on its own)

### Phase 0 — Skeleton & config
- [ ] **0.1** Create the new working tree + git repo; copy `.env.example`, a snapshot of the
      5 old scripts as `reference/` (never edited), and this `DESIGN.md`.
- [ ] **0.2** `config.py` — `Config` dataclass reading `.env`; `logging_setup.py`.
- [ ] **0.3** `llm/models.py` — `GatingResponse`, `RelevanceFilter` (verbatim from the old code).
- [ ] **0.4** `llm/gateway.py` — `LLMEvaluator` + retry/backoff (verbatim, moved). Unit: fake-gate
            test that `evaluate()` returns True/False and retries on failure.

### Phase 1 — Library modules (copy algorithms, add event callbacks)
- [ ] **1.1** `crawler/cache.py` — `Cache` load/save. Test: round-trip via tmp path.
- [ ] **1.2** `crawler/fetcher.py` + `processor.py` — Wikipedia fetch + text processing. Test:
            `is_bad_page`, `title_to_filename`, `extract_first_paragraph` with fixture text.
- [ ] **1.3** `crawler/runner.py` — `crawl()` emitting `ProgressEvent`s, bound to cache for resume.
            Test with a **fake fetcher** (no network).
- [ ] **1.4** `splitting/splitter.py` — `find_sections`, `chunk_section` verbatim. Test:
            `== header` + bare-heading + References-stripping fixtures.
- [ ] **1.5** `gating/chunk_gate.py` — fast-path bypasses + move logic, emitting events.
            Test: bypass decisions, no network needed.
- [ ] **1.6** `ingestion/dedup.py` — shingle tokenize + MinHashLSH helpers. Test: near-duplicate
            collapses to one.
- [ ] **1.7** `ingestion/build_db.py` — read → dedup → batch embed → upsert. Inject the embed
            client so tests run without Ollama.
- [ ] **1.8** `retrieval/store.py` + `search.py` + `filter.py` — sqlite-vec load, ANN, dedup,
            relevance filter. Test with a pre-seeded tiny DB.

### Phase 2 — Orchestrator as library
- [ ] **2.1** `pipeline/events.py` — the event dataclasses (Section 6 contract).
- [ ] **2.2** `pipeline/checkpoint.py` — `State` load/save, `_normalize_state` for legacy schema.
            Test: legacy→new mapping, corrupt-file→all-pending, mark-and-reload.
- [ ] **2.3** `pipeline/archive.py` — copy + gzip-tar + full-stream integrity check. Test:
            good archive passes, truncated archive fails.
- [ ] **2.4** `pipeline/runner.py` — `PipelineRunner` binding 1.1–1.8, checkpoint at every
            durable boundary, pause/stop flags, flock lock, `on_event` fan-out.
- [ ] **2.5** Test the runner with **fake gate + fake fetcher + fake embed** over a fixture
            project dir end-to-end (the integration test that proves the funnel works without
            real Ollama or network).

### Phase 3 — Textual UI
- [x] **3.1** `ui/app.py` scaffold — app shell, screens, comms layer to `PipelineRunner`
            (workers, event subscription).  *(committed; headless mount + nav test passes)*
- [x] **3.2** `ui/widgets/folder_tree.py` — browseable output/ tree; on select, load project
            checkpoint.
- [x] **3.3** `ui/widgets/progress_panel.py` — per-stage bar + live decision log.
- [x] **3.4** `ui/widgets/stats_panel.py` — rolling metrics, throttled repaint.
- [x] **3.5** `ui/widgets/query_box.py` — query → results overlay.
- [x] **3.6** `ui/styles.tcss` — layout + colour palette; header status + pause/stop controls.
- [x] **3.7** Wire config dialog + resume-from-checkpoint flow into the app.

### Phase 4 — Hardening & sign-off
- [x] **4.1** Signal handling (SIGINT/SIGTERM) → graceful checkpoint + exit.  (TUI-level
            handler posts a `Quit` message on the main thread; `on_quit` runs the same
            graceful stop the STOP button uses, so the runner persists its checkpoint.)
- [x] **4.2** Full pytest suite (target the ~48-test parity the old docs promised, minus the
            subprocess-oriented tests that no longer apply).  Full suite passes (45 passed)
            once the runtime deps (requests/pydantic/ollama/…) are installed; two
            `save_prompt` path assertions were corrected to the shared `prompts/` dir.
- [ ] **4.3** Manual end-to-end smoke run with real Ollama on one seed page.
- [x] **4.4** `__main__.py` (`python -m wiki_ragify`) + README.

---

## 10. What "done" looks like

- `python -m wiki_ragify` opens the Textual TUI.
- You browse to a project dir, set the start page + prompt, hit **run**.
- The progress window and statistics panel live-update while crawl → split → gate → archive →
  embed all run **in-process** in the same program — no subprocess glue, no 5 CLIs.
- You **pause** mid-crawl, quit; relaunch, and **resume** from the same checkpoint.
- The query box retrieves from the finished store.
- Full pytest suite passes; config is 100% in `.env`.

## 11. Future / Change Requests

- **LLM-generated gating prompt from a topic.** ✅ DONE. `wiki_ragify/gating/prompt_gen.py`
  renders `example/make_a_topic_gate.txt` (with `{{target_topic | textarea}}` and
  `{{exclude | textarea}}` placeholders) and drives the *same gate model* the crawler uses
  to emit a topic-specific `gating_prompt.txt`. The rendered prompt is written under
  `output/prompts/` next to a small manifest recording the seeds used, so a run is traceable
  back to its prompt. `wiki_ragify/gating/make_gate_prompt.py` is the thin CLI wrapper — a
  **pre-run
  step, not part of the crawler**: it writes a prompt you then point
  `smoke_run.py --prompt ...` (or the TUI) at. Iterative refinement is built in — with
  `--refine`, the prompt from the previous pass is picked up automatically and fed back with
  the instruction. The generation call uses plain text (no `GatingResponse` JSON schema) with
  the gate's temperature/`num_ctx`/timeout, and can raise `num_ctx` to fit the larger
  template. The prompt-authoring turn stays human-led (a topic seeds it) while letting a
  topic produce it.
