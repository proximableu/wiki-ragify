"""End-to-end smoke test for the wiki_ragify pipeline against a *local* Ollama server.

This is NOT a unit test — it is a real funnel that touches the network and a real
LLM server, so run it deliberately (it writes under ``output_dir`` and, with
``--ingest``, builds a knowledge DB):

    # Gate 1 + Gate 2 with the local gate model, archive only (no DB build):
    python example/smoke_run.py

    # ...and also embed + build a knowledge DB in the knowledge/ dir:
    python example/smoke_run.py --ingest

Environment overrides (all optional; sensible defaults are used otherwise):

    OLLAMA_URL          Ollama HTTP endpoint the *package* talks to (default
                        http://localhost:11434). Point this at the local server.
    GATE_MODEL_NAME     Model used for both gates (default qwen3-30b-dataset).
    EMBED_MODEL         Embedding model for the ingest phase (default
                        snowflake-arctic-embed2:568m).

The seed page, gating prompt file, and output root are chosen here; the pipeline
code itself is left untouched. See ./README.md for the writeup.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Make the package importable when run from the repo root as ``python example/smoke_run.py``.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from wiki_ragify.config import Config, PipelineConfig  # noqa: E402
from wiki_ragify.logging_setup import setup_logging  # noqa: E402
from wiki_ragify.pipeline.runner import PipelineRunner  # noqa: E402


def _print_event(event) -> None:
    """on_event callback: render a compact, single-line summary of each pipeline tick."""
    stage = getattr(event, "stage", "?")
    kind = getattr(event, "kind", "tick")
    index = getattr(event, "index", 0)
    total = getattr(event, "total", None)
    msg = getattr(event, "message", "")
    tail = f"/{total}" if total else ""
    print(f"[{stage}/{kind}{tail}] {msg}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start-page", default="Autistic supremacism", help="Wikipedia title to crawl.")
    parser.add_argument("--prompt", default=_REPO_ROOT / "example" / "gating_prompt.txt", help="Path to the gating prompt file (default: example/gating_prompt.txt).")
    parser.add_argument("--output-dir", default=_REPO_ROOT / "output", help="Root dir for crawl/archive/DB.")
    parser.add_argument("--timeout", type=float, default=6 * 60 * 60, help="Wall-clock cap in seconds.")
    parser.add_argument("--no-ingest", dest="ingest", action="store_false", help="Archive only; skip DB build.")
    parser.add_argument("--no-network", action="store_true", help="Fail fast if OLLAMA_URL is unreachable before starting.")
    return parser


def _check_ollama(url: str) -> None:
    """Best-effort reachability probe; only fails the run if --no-network is set."""
    import urllib.request

    try:
        req = urllib.request.Request(url.rstrip("/") + "/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            tags = resp.read().decode("utf-8", "replace")
        print(f"Ollama reachable at {url}; models in reply:\n  {tags[:400]}", flush=True)
    except Exception as exc:  # noqa: BLE001 - report and let caller decide
        print(f"WARNING: could not reach Ollama at {url}: {exc}", flush=True)


def main() -> int:
    args = build_parser().parse_args()

    # Setup logging so pipeline gate DEBUG output (raw model answers) surfaces.
    # Set LOGLEVEL=DEBUG to see every raw gate response on stderr.
    setup_logging(level=logging.DEBUG if os.getenv("LOGLEVEL", "").upper() == "DEBUG" else logging.INFO)

    prompt_path = Path(args.prompt).expanduser().resolve()
    if not prompt_path.exists():
        print(f"ERROR: gating prompt not found: {prompt_path}", flush=True)
        return 2

    if args.no_network:
        _check_ollama(Config.ollama_url)

    config = Config().with_output_dir(args.output_dir)
    pipeline = PipelineConfig(
        start_page=args.start_page,
        prompt_path=prompt_path,
        output_dir=args.output_dir,
        ingest=args.ingest,
        timeout=args.timeout,
    )

    print(
        "=== wiki_ragify smoke run ===",
        f"gate model : {config.gate_model}",
        f"embed model: {config.embed_model}",
        f"ollama url : {config.ollama_url}",
        f"start page : {pipeline.start_page}",
        f"prompt     : {prompt_path}",
        f"output dir : {args.output_dir}",
        f"ingest     : {pipeline.ingest}",
        sep="\n",
        flush=True,
    )

    runner = PipelineRunner(config, pipeline, on_event=_print_event)
    try:
        runner.run()
    except KeyboardInterrupt:
        print("\nInterrupted; runner exited cleanly at the next durable boundary.", flush=True)
        return 130

    print(
        "=== smoke run finished ===",
        f"accepted list : {args.output_dir / 'accepted_pages.list'}",
        f"archive       : {args.output_dir / pipeline.archive_name}",
        sep="\n",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
