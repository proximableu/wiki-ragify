# Wiki Ragify v2

A contained, TUI-driven Wikipedia → knowledge-base pipeline. It crawls a Wikipedia
seed page, LLM-gates the articles, splits them into chunks, archives what's accepted,
embeds into a local `sqlite-vec` store, and lets you **retrieve** the most relevant
passages from inside the app.

Everything runs **in-process** — there is no subprocess spawning or `os.system`. The
heavy lifting is a library (`wiki_ragify/…`) and the front door is a
[Textual](https://textual.textualize.com/) multi-screen TUI.

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Header                                                                     │
├────────────┬───────────────────────────┬─────────────────┬─────────────────┤
│            │                           │                 │                 │
│ Folder     │  Progress log             │  Stats panel    │  (rolling)      │
│ picker     │  [N/total] ACCEPT title   │  score / bytes …│                 │
│ (FR-1)     │  [N/total] REJECT reason  │                 │                 │
│            │                           │                 │                 │
├────────────┴───────────────────────────┴─────────────────┴─────────────────┤
│ [ Query the knowledge base… ]        [ Run ] [ Pause ] [ Stop ]             │
└───────────────────────────────────────────────────────────────────────────┘
```

## How it works

The pipeline is a funnel of phases, each checkpointed so a run can resume where it left
off (DESIGN §2.5, §6.3):

1. **Crawl** — fetch one seed Wikipedia article and its links (depth=1).
2. **Gate (article)** — an LLM decides to accept/reject/skip each article.
3. **Split** — break each article into chunks.
4. **Gate (chunk)** — fast-path bypasses + an LLM chunk gate.
5. **Archive** — gzip-tar accepted articles with a full-stream integrity check.
6. **Ingest** — dedup (MinHashLSH) → batch-embed → upsert into `sqlite-vec`.
7. **Retrieve** — ANN search → LLM relevance filter, surfaced in the TUI (FR-8).

## Requirements

* Python 3.11+
* [Ollama](https://ollama.com/) reachable at the URL you configure below.
* The models named in `.env.example` (gating, embedding, light relevance model).

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .            # main deps
pip install -e ".[vector,tokens,dev]"   # vector store + tokenizer + pytest
```

Dependencies live in `pyproject.toml` (`[project.dependencies]` and the
`[project.optional-dependencies]` groups for `vector`, `tokens`, `dev`).

### Configure

```bash
cp .env.example .env
$EDITOR .env
```

The two values that almost always need changing:

| Variable          | Meaning                                                        |
| ----------------- | -------------------------------------------------------------- |
| `OLLAMA_URL`      | Where Ollama is listening, e.g. `http://192.168.0.14:11434`    |
| `GATE_MODEL_NAME` | The gating model (article + chunk gate).                        |

All other tunables (model names, dimensions, thresholds, splitter sizes, HNSW params) are
documented in `.env.example`. **Configure through these environment variables only** — the
`wiki_ragify` package never hardcodes infrastructure, and the real secrets live in
`reference/.env`, which you must **not** touch.

## Running

### The TUI (recommended)

```bash
python -m wiki_ragify
# or, if installed:  wiki-ragify
```

This opens the full-screen app. The output root defaults to `~/wiki`; create a project
under it (a folder like `~/wiki/Autistic-supremacism`), then:

1. Pick a project in the **folder picker** (left). Its checkpoint loads into the progress
   window.
2. Hit **Run** → confirm the start page + gating prompt path in the config modal.
3. Watch the decision log and live stats. **Pause** / **Stop** as needed.

### Retrieving

Once a project has been ingested, type a question in the footer input and press Enter.
Retrieval runs on a worker thread (it's blocking — embed + LLM relevance) and the
top-k passages appear in an overlay table.

## Tests

Offline unit tests cover the pure helpers (folding, event logging, config, splitter, the
pipeline runner with fake gate/fetch/embed):

```bash
pytest
```

> The suite's collection will error on modules that need `requests` / `pydantic`
> (`test_cache`, `test_gateway_decision`, `test_prompt_gen`) unless those packages are
> installed — that's a dependency gap in your venv, not a code defect. Install them to
> get full parity.

To exercise the live paths (real crawl + gate + embed + retrieve) you'll need Ollama
running and a small seed — see `example/`.

## Layout

```
wiki_ragify/
├── __main__.py              # python -m wiki_ragify
├── config.py                # one Config dataclass, read from .env
├── crawler/ splitter/ gating/ ingestion/ retrieval/   # the funnel (library)
├── pipeline/                # checkpoint, archive, lock, in-process runner
├── llm/                     # gateway, embed, models
└── ui/                      # Textual app + widgets (FR-1..8)
    ├── app.py               # app shell, screens, run-controls state machine
    ├── styles.tcss          # layout + color palette
    └── widgets/             # folder_tree, progress_panel, stats_panel, query_box
```

## Security note

The real `.env` with live credentials lives under `reference/`. **Never edit files in
`reference/`.** The new working tree carries only `.env.example` — supply real values
through environment variables (a local `.env`) and never commit secrets.
