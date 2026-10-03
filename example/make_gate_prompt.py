"""Generate a topic-specific gating prompt using the same gate model the crawler uses.

This is the thin CLI wrapper around ``wiki_ragify.gating.prompt_gen``. It is **not**
part of the crawler — it is a pre-run step that renders a ``gating_prompt.txt`` you then
point ``example/smoke_run.py --prompt ...`` (or the TUI) at. No crawler code is touched.

Run it (env vars are read by ``Config``, exactly as for the smoke run):

    export OLLAMA_URL=http://192.168.0.14:11434
    export GATE_MODEL_NAME=qwen3-30b-dataset

    # From the repo root, using the bundled template (topic + exclusions):
    python example/make_gate_prompt.py "autistic supremacism" \\
        --exclude "self-harm, violence, instructions"

    # Iterative refinement — the prompt written by the previous pass is picked up
    # automatically and fed back with the instruction:
    python example/make_gate_prompt.py "autistic supremacism" \\
        --refine "make the rejection boundary narrower for borderline articles"

    # Use a custom template instead of the bundled one:
    python example/make_gate_prompt.py "topic" --template ./my_template.txt

The rendered prompt is written under ``output/prompts/`` (git-ignored) alongside a small
manifest recording the seeds used, so a run is traceable back to its prompt.

With ``--print``, the first ~20 lines of the generated prompt are echoed to stdout.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from wiki_ragify.config import Config  # noqa: E402
from wiki_ragify.gating import prompt_gen  # noqa: E402
from wiki_ragify.logging_setup import setup_logging  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a topic-specific gating prompt with the gate model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("topic", help="Short description of the target topic.")
    parser.add_argument("--exclude", default="", help="Aspects to exclude from gating.")
    parser.add_argument(
        "--template",
        default=None,
        help="Path to a custom template (default: bundled example/make_a_topic_gate.txt).",
    )
    parser.add_argument(
        "--refine",
        default=None,
        help="One-line instruction fed to the model to refine the prompt written by the "
        "previous pass (iterative refinement). Auto-uses the prior generated prompt file.",
    )
    parser.add_argument(
        "--output-dir",
        default=_REPO_ROOT / "output",
        help="Root dir for artifacts (default: ./output).",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help="Override num_ctx for the generation call (raise past the gate default).",
    )
    parser.add_argument(
        "--print",
        dest="show",
        action="store_true",
        help="Echo the first ~20 lines of the generated prompt.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    setup_logging(
        level=logging.DEBUG if os.getenv("LOGLEVEL", "").upper() == "DEBUG" else logging.INFO
    )

    template_path = Path(args.template).expanduser().resolve() if args.template else prompt_gen.DEFAULT_TEMPLATE
    if not template_path.exists():
        print(f"ERROR: template not found: {template_path}", flush=True)
        return 2

    config = Config().with_output_dir(args.output_dir)

    print(
        "=== make_gate_prompt ===",
        f"gate model : {config.gate_model}",
        f"ollama url : {config.ollama_url}",
        f"topic      : {args.topic}",
        f"exclude    : {args.exclude or '<none>'}",
        f"template   : {template_path}",
        sep="\n",
        flush=True,
    )

    # Iterative refinement: if a prior pass already wrote this topic's prompt, feed it back.
    prompts_dir = config.output_dir / "prompts"
    stem = prompt_gen.db_titlepath(args.topic).removesuffix(".db")
    prior = prompts_dir / f"{stem}.txt"

    if args.refine is not None and prior.exists():
        prompt_text = prompt_gen.refine_gating_prompt(
            prior.read_text(encoding="utf-8"),
            args.refine,
            config=config,
            num_ctx=args.num_ctx,
        )
        print(f"Refining prior prompt with instruction: {args.refine}", flush=True)
    else:
        prompt_text = prompt_gen.generate_gating_prompt(
            template_text=template_path.read_text(encoding="utf-8"),
            topic=args.topic,
            exclude=args.exclude,
            config=config,
            num_ctx=args.num_ctx,
        )
        if not prompt_text:
            print("ERROR: generation returned an empty prompt (see logs).", flush=True)
            return 1

    saved = prompt_gen.save_prompt(prompt_text, args.topic, args.exclude, output_dir=config.output_dir)
    print(f"Prompt {'refined' if (args.refine and prior.exists()) else 'generated'}; written to {saved}", flush=True)

    if args.show:
        lines = [ln for ln in prompt_text.splitlines()][:20]
        print("----- generated prompt (first 20 lines) -----", flush=True)
        print("\n".join(lines), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
