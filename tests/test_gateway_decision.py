"""Offline tests for the gate decision normalization in ``LLMEvaluate.evaluate``.

The live model returns JSON via Ollama's structured-output mode, and different
models (or prompts) vary the exact token. The gate must route every accepted
wording to True and every rejected wording to False, regardless of case, with
an unknown verdict defaulting to the safe reject.

The Ollama HTTP client is faked so no server is needed.
"""

from wiki_ragify.config import Config
from wiki_ragify.llm.gateway import LLMEvaluator


def _make_evaluator(responses: list[str]) -> LLMEvaluator:
    """Build a real evaluator whose Ollama client returns each JSON response in order.

    The final response is repeated if more attempts are made than responses given.
    """
    ev = LLMEvaluator(Config())
    responses = list(responses)

    class _Resp:
        def __init__(self, content: str):
            self.message = type("M", (), {"content": content})()

    class _Client:
        def __init__(self, *a, **k):
            pass

        idx = 0

        def chat(self, **kwargs):
            i = min(_Client.idx, len(responses) - 1)
            _Client.idx += 1
            return _Resp(responses[i])

    ev.client = _Client()
    return ev


def _evaluate(ev: LLMEvaluator, retries: int = 1) -> bool:
    return ev.evaluate("some text", "a gating prompt", retries=retries)


def test_accepts_all_accepted_wordings():
    for json_text in ['{"decision": "accepted"}', '{"decision": "ACCEPT"}',
                      '{"decision": "Accepted"}', '{"decision": "accept"}',
                      '{"decision": "ACCEPTS"}', '{"decision": "accepts"}',
                      '{"decision": " accepted "}', '{"decision": "ACCEPTED"}']:
        assert _evaluate(_make_evaluator([json_text])), json_text


def test_rejects_all_rejected_wordings():
    for json_text in ['{"decision": "rejected"}', '{"decision": "REJECT"}',
                      '{"decision": "Rejected"}', '{"decision": "reject"}',
                      '{"decision": "REJECTS"}', '{"decision": "rejects"}']:
        assert not _evaluate(_make_evaluator([json_text])), json_text


def test_unknown_verdict_defaults_to_reject():
    for json_text in ['{"decision": "maybe"}', '{"decision": ""}', '{"decision": "meh"}']:
        assert not _evaluate(_make_evaluator([json_text])), json_text


def test_retry_uses_last_attempt_decision():
    """A transient failure is retried; the final (successful) attempt decides."""
    ev = _make_evaluator(["boom", '{"decision": "ACCEPT"}'])
    # retries=2 -> 2 total attempts: first raises, second ACCEPT wins.
    assert _evaluate(ev, retries=2)  # boom -> ACCEPT


def test_all_attempts_fail_defaults_reject():
    """If every attempt errors, the safe default is reject (retries not exhausted)."""
    ev = _make_evaluator(["boom"])  # only one response; all attempts raise
    assert not _evaluate(ev, retries=3)
