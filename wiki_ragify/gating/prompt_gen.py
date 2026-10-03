"""Topic-gating-prompt generator — synthesize a gating prompt from a topic description.

This is the "future / change request" from DESIGN.md §11: instead of hand-writing
``example/gating_prompt.txt`` for every run, the **gate model** renders a
topic-specific gating prompt from a short description of the target topic and a list
of aspects to exclude. The rendered prompt is then fed to the *same* gate the crawler
uses — so generation is a pre-run step that never touches the crawler itself.

The generator is deliberately a thin wrapper over the already-configured Ollama
client (reusing ``LLMEvaluator``), not a new dependency or a re-implementation of the
connection plumbing. Key choices:

* **Plain text, no JSON schema.** The goal here is *producing* a gating prompt, not
  classifying a chunk, so we don't apply :class:`GatingResponse`. A single ``user``
  message is all the model needs (the template already tells it to return only text).
* **Same options as the gate.** Temperature / ``num_ctx`` / ``think`` come from
  ``LLMEvaluator.options`` (which is sourced from ``Config``), so generation behaves
  like the gate would. ``num_ctx`` can be raised: the rendered prompt is larger than a
  typical gate prompt and must fit.
* **Reproducible artifacts.** Each generation writes the rendered prompt to a stable
  filename and a small JSON manifest recording the seeds used, so a run is traceable
  back to the exact prompt that produced it.
* **Iterative refinement.** The prompt starts from the default template; callers can
  refine it by feeding the previous attempt back with an extra instruction (e.g. "make
  the rejection boundary narrower"), which is what makes the tool advanced rather than
  a one-shot.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from ..config import Config, db_titlepath
from ..llm.gateway import LLMEvaluator

logger = logging.getLogger(__name__)

# Placeholders in ``make_a_topic_gate.txt``. The trailing ``| textarea`` mirrors the
# original prompt's intent (human-editable fields) and is stripped here — the template
# receives fully-substituted plain text.
_PLACEHOLDER = re.compile(r"\{\{\s*(target_topic|exclude)\s*\|\s*textarea\s*\}\}")

DEFAULT_TEMPLATE = Path(__file__).resolve().parent.parent / "example" / "make_a_topic_gate.txt"


def render_template(template_text: str, topic: str, exclude: str) -> str:
    """Substitute the topic / exclude placeholders in the template.

    Handles every placeholder form in the template regardless of whitespace. Unknown
    placeholders are left untouched (they'll surface to the user as a missing value).
    """
    return _PLACEHOLDER.sub(
        lambda m: topic if m.group(1) == "target_topic" else exclude,
        template_text,
    )


def generate_gating_prompt(
    template_text: str,
    topic: str,
    exclude: str = "",
    *,
    config: Config,
    evaluator: Optional[LLMEvaluator] = None,
    retries: int = 3,
    num_ctx: Optional[int] = None,
) -> str:
    """Ask the gate model to synthesize a gating prompt for ``topic``.

    Uses the **same gate model and connection** as the crawler — a plain ``chat`` call
    with no JSON schema — and returns the raw prompt text. Returns the empty string if
    every attempt fails, so callers can treat it as a benign no-op.
    """
    prompt_text = render_template(template_text, topic, exclude)
    return _generate(config, prompt_text, evaluator, retries, num_ctx)


def refine_gating_prompt(
    previous_prompt: str,
    instruction: str,
    *,
    config: Config,
    evaluator: Optional[LLMEvaluator] = None,
    retries: int = 3,
    num_ctx: Optional[int] = None,
) -> str:
    """Refine an existing generated prompt with a one-line user instruction.

    Feeds ``{previous_prompt}\n\nINSTRUCTION: {instruction}`` back to the gate model.
    An empty ``previous_prompt`` (e.g. the first generation failed) falls back to
    ``INSTRUCTION: {instruction}`` alone — a reasonable fallback that still drives the
    model toward the desired prompt.

    Deliberately does **not** run ``render_template``: there is no seed prompt to place
    into placeholders here, and ``instruction`` may itself contain ``{{ }}`` characters.
    """
    body = (
        f"{previous_prompt}\n\nINSTRUCTION: {instruction}"
        if previous_prompt
        else f"INSTRUCTION: {instruction}"
    )
    return _generate(config, body, evaluator, retries, num_ctx)


def _generate(
    config: Config,
    prompt_text: str,
    evaluator: Optional[LLMEvaluator],
    retries: int,
    num_ctx: Optional[int],
) -> str:
    """Low-level: send ``prompt_text`` to the gate model with retry + backoff."""
    evaluator = evaluator or LLMEvaluator(config)

    options = dict(evaluator.options)
    if num_ctx is not None:
        options["num_ctx"] = num_ctx

    last_error: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        try:
            response = evaluator.client.chat(
                model=evaluator.model,
                messages=[{"role": "user", "content": prompt_text}],
                options=options,
            )
            content = response.message.content
            logger.debug(f"Raw gate-model prompt generation: {content!r}")
            return content
        except Exception as e:  # noqa: BLE001 - fall back to a benign no-op prompt
            last_error = e
            delay = min(0.5 * (2 ** (attempt - 1)), 5.0)
            logger.warning(
                f"Prompt generation attempt {attempt}/{retries} failed ({type(e).__name__}), "
                f"retrying in {delay:.1f}s: {e}"
            )
    logger.warning(f"Prompt generation failed after {retries} attempts: {last_error}")
    return ""


def save_prompt(
    prompt_text: str,
    topic: str,
    exclude: str,
    *,
    output_dir: Path,
) -> Path:
    """Write the rendered/generated prompt and a small manifest into ``output_dir``.

    Returns the path to the ``.txt`` prompt (the manifest sits beside it as
    ``<prompt-stem>.manifest.json``). Filenames reuse ``db_titlepath`` so the prompt and
    its knowledge DB share a recognizable slug. Writes nothing on an empty ``prompt_text``.
    """
    output_dir = Path(output_dir)
    prompt_dir = output_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    if not prompt_text.strip():
        logger.warning("Empty prompt — nothing written.")
        return prompt_dir / ""

    stem = db_titlepath(topic).removesuffix(".db")
    prompt_path = prompt_dir / f"{stem}.txt"
    manifest_path = prompt_dir / f"{stem}.manifest.json"

    prompt_path.write_text(prompt_text, encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {"topic": topic, "exclude": exclude, "prompt_file": prompt_path.name},
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote gating prompt to %s (manifest: %s)", prompt_path, manifest_path)
    return prompt_path
