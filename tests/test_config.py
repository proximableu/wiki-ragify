"""Offline tests for centralized config."""

from wiki_ragify.config import Config


def test_defaults():
    c = Config()
    assert c.ollama_url == "http://localhost:11434"
    assert c.gate_model == "qwen3-30b-dataset"
    assert c.embed_dim == 1024
    assert c.max_tokens == 2048
    assert c.knowledge_dirname == "knowledge"


def test_with_output_dir_is_immutable_copy():
    base = Config()
    copy = base.with_output_dir("/tmp/x")
    assert copy.output_dir == __import__("pathlib").Path("/tmp/x")
    # Original unchanged
    assert base.output_dir is None
