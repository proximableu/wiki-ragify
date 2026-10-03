"""Offline tests for ``wiki_ragify.gating.prompt_gen``.

The generator renders the topic-gate template and sends it to the gate model, which
produces a ``gating_prompt.txt`` for a run to consume. The Ollama HTTP client is faked
so no server is needed; the render/placeholder and refinement plumbing are exercised
deterministically, stubbing the actual generation call.
"""

from __future__ import annotations

import json

import pytest

from wiki_ragify.config import Config
from wiki_ragify.gating import prompt_gen

TEMPLATE = """TOPIC: {{target_topic | textarea}}
EXCLUDE: {{exclude | textarea}}
BODY: keep me
"""


def test_render_template_substitutes_both_placeholders():
    out = prompt_gen.render_template(TEMPLATE, "autistic supremacism", "violence, code")
    assert "autistic supremacism" in out
    assert "violence, code" in out
    assert "keep me" in out
    assert "{{" not in out  # placeholders gone


def test_render_template_strips_whitespace_variants():
    template = "T: {{  target_topic|textarea  }} E: {{exclude|textarea}}"
    assert prompt_gen.render_template(template, "topic X", "ex Y") == "T: topic X E: ex Y"


def test_render_template_leaves_unknown_placeholders_untouched():
    out = prompt_gen.render_template("A {{target_topic | textarea}} B {{missing | textarea}}", "T", "E")
    assert "B {{missing | textarea}}" in out


def test_generate_sends_rendered_prompt_to_model(monkeypatch):
    seen = {}

    def fake_generate(config, prompt_text, *a, **k):
        seen["text"] = prompt_text
        return "GENERATED PROMPT"

    monkeypatch.setattr(prompt_gen, "_generate", fake_generate)
    out = prompt_gen.generate_gating_prompt(TEMPLATE, "autistic supremacism", "violence", config=Config())
    assert out == "GENERATED PROMPT"
    assert "autistic supremacism" in seen["text"]
    assert "violence" in seen["text"]
    assert "{{" not in seen["text"]  # the model got the rendered text, not the raw template


def test_refine_bypasses_template_and_prefixes_instruction(monkeypatch):
    seen = {}

    def fake_generate(config, prompt_text, *a, **k):
        seen["text"] = prompt_text
        return "REFINED"

    monkeypatch.setattr(prompt_gen, "_generate", fake_generate)
    out = prompt_gen.refine_gating_prompt("OLD PROMPT", "narrower boundary", config=Config())
    assert out == "REFINED"
    assert seen["text"] == "OLD PROMPT\n\nINSTRUCTION: narrower boundary"


def test_refine_empty_previous_falls_back_to_instruction_only(monkeypatch):
    seen = {}

    def fake_generate(config, prompt_text, *a, **k):
        seen["text"] = prompt_text
        return "REFINED"

    monkeypatch.setattr(prompt_gen, "_generate", fake_generate)
    out = prompt_gen.refine_gating_prompt("", "narrower boundary", config=Config())
    assert out == "REFINED"
    assert seen["text"] == "INSTRUCTION: narrower boundary"


def test_save_prompt_writes_txt_and_manifest(tmp_path):
    prompt = "TOP: X\n---"
    path = prompt_gen.save_prompt(prompt, "Autistic supremacism", "violence", output_dir=tmp_path)
    assert path.name == "autistic-supremacism.txt"
    assert path.read_text() == prompt
    manifest = tmp_path / "autistic-supremacism.manifest.json"
    assert manifest.exists()
    data = json.loads(manifest.read_text())
    assert data["topic"] == "Autistic supremacism"
    assert data["exclude"] == "violence"
    assert data["prompt_file"] == "autistic-supremacism.txt"


def test_save_prompt_writes_nothing_on_empty(tmp_path):
    path = prompt_gen.save_prompt("", "Some topic", "", output_dir=tmp_path)
    assert not path.exists()
    # no prompt file or manifest is written for an empty prompt
    assert list((tmp_path / "prompts").glob("*.txt")) == []
    assert list((tmp_path / "prompts").glob("*.manifest.json")) == []
