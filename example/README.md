# Example: end-to-end smoke run

`smoke_run.py` runs the **real** `wiki_ragify` pipeline end-to-end against a
**real remote Ollama server** — no mocking, no test doubles. It exercises the
whole funnel:

```
crawl "Autistic supremacism"
   → Gate 1 (article-level)
   → crawl the article's outbound links (each gated by the same model)
   → Gate 2 (chunk-level) on accepted articles
   → archive accepted content to a gzip-tar
   → (optional) embed + build a knowledge DB
```

Nothing in `wiki_ragify/` is modified for this run. The only things you point at
the remote server are **environment variables**, which `Config` reads. This is
deliberately a demonstration script, not a unit test — it hits the network and an
LLM server, so run it when you want an honest sanity check of the real system.

## What it wires

| Value | Source | Set via |
|-------|--------|---------|
| Ollama endpoint | `Config.ollama_url` | `OLLAMA_URL` |
| Gate 1 model (article gate) | `Config.gate_model` | `GATE_MODEL_NAME` |
| Gate 2 model (chunk gate) | `Config.gate_model` | `GATE_MODEL_NAME` |
| Embed model (ingest phase) | `Config.embed_model` | `EMBED_MODEL` |
| Seed page | `PipelineConfig.start_page` | `--start-page` (default `Autistic supremacism`) |
| Gating prompt | `PipelineConfig.prompt_path` | `--prompt` (default `./example/gating_prompt.txt`) |

## Prerequisites

* A remote Ollama serving on `http://192.168.0.14:11434` with both models loaded:
  * gate model — `qwen3-30b-dataset:latest` (matches the original `.env.example`)
  * embed model — `snowflake-arctic-embed2:568m`
* The project's Python dependencies installed (`pip install -r requirements.txt`),
  in a Python that has the repo importable.

## Run it

```bash
# Gate 1 + Gate 2, archive only (no DB build) — fastest:
export OLLAMA_URL=http://192.168.0.14:11434
export GATE_MODEL_NAME=qwen3-30b-dataset
export EMBED_MODEL=snowflake-arctic-embed2:568m

python example/smoke_run.py

# Also embed + build a knowledge DB in the knowledge/ dir:
python example/smoke_run.py                 # ingest is on by default
python example/smoke_run.py --no-ingest     # archive only, skip DB
```

Each event prints one compact line, e.g.:

```
[crawl/tick/1] === 1/1: Autistic supremacism ===
[crawl/reject] Autistic supremacism      # Gate 1 decision
[crawl/tick/link/1] [LINK] 16p11.2 deletion syndrome
[gate/tick/...]                          # Gate 1 + Gate 2 decisions on the article / its chunks
```

Artifacts land under `./output` (git-ignored): `accepted_pages.list`, the
`.tgz` archive, and — with ingest — the SQLite knowledge DB.

## Note on latency and a gating bug this smoke test caught

`qwen3-30b-dataset` is a large model; under CPU inference each gate call is slow.
The package's default `gate_timeout` (60s) was too short and every call tripped
`ReadTimeout`, after which the safe-default **rejects**. So set
`GATE_TIMEOUT=600` (as above) so the seed page isn't rejected by a stall.

This end-to-end run is also how the gate's **decision-matching** bug was caught.
The model returns `{"decision": "ACCEPT"}` (uppercase) through Ollama's JSON mode,
but the original gate compared against the lowercase literal `"accepted"`:

```python
# wiki_ragify/llm/gateway.py
return result.decision == "accepted"   # model emits "ACCEPT"
```

So a genuine acceptance silently fell through to the safe-default reject. Both
`gemma4:e4b` and `qwen3-30b-dataset` were probed and both emit the uppercase
`"ACCEPT"`. The fix normalizes the verdict before comparing — lowercase, then take
the first four letters — so `accept`/`accepted`/`accepts` all collapse to `"acce"`
and `reject`/`rejected`/`rejects` to `"reje"`:

```python
decision = str(result.decision).strip().lower()[:4]
if decision == "acce":
    return True
if decision == "reje":
    return False
```

An unknown/missing verdict still defaults to reject, and every verdict path is
covered by `tests/test_gateway_decision.py` (fakes the Ollama client, so it runs
offline).
