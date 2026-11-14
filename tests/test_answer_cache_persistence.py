"""The saved answer cache: survives restarts, keeps no student wording, drops stale answers."""

import json
import time

import pytest

from config import settings
from src.inference import answer_cache
from src.inference.answer_cache import SemanticAnswerCache
from tests.test_agentic import WordEmbedder

SIG = SemanticAnswerCache.signature("5-mark", None, "adaptive")


def response(answer, question="Explain the Rankine cycle."):
    return {"answer": answer, "question": question,
            "decision": {"analysis": {"question": question}},
            "telemetry": {"memory": {"resolved_query": question},
                          "agent_trace": [{"action": "plan", "query": question, "found": 0}]}}


@pytest.fixture
def path(tmp_path):
    return tmp_path / "cache.jsonl"


def test_answers_survive_a_restart(path):
    SemanticAnswerCache(WordEmbedder(), path=path).put("Explain the Rankine cycle.", SIG, response("A"))
    reloaded = SemanticAnswerCache(WordEmbedder(), path=path)
    assert reloaded.get("explain the rankine cycle", SIG)["answer"] == "A"          # exact tier
    assert reloaded.get("The Rankine cycle: explain", SIG)["answer"] == "A"         # semantic tier


def test_answers_saved_by_another_process_are_picked_up(path):
    """The question-bank builder runs beside the app: the app must see new bank answers."""
    app = SemanticAnswerCache(WordEmbedder(), path=path)
    assert app.get("Explain the Rankine cycle.", SIG) is None
    builder = SemanticAnswerCache(WordEmbedder(), path=path)
    time.sleep(0.05)                                     # a distinct file timestamp
    builder.put("Explain the Rankine cycle.", SIG, response("bank"), pinned=True)
    assert app.get("What is the Rankine cycle?", SIG)["answer"] == "bank"


def test_no_student_wording_is_saved(path):
    secret = "Explain the Rankine cycle for Priya in room 204"
    SemanticAnswerCache(WordEmbedder(), path=path).put(secret, SIG, response("A", secret))
    text = path.read_text(encoding="utf-8").lower()
    for word in ("priya", "room", "204", "rankine"):
        assert word not in text, word
    # ...yet the answer is still found after a restart
    assert SemanticAnswerCache(WordEmbedder(), path=path).get(secret, SIG)["answer"] == "A"


def test_question_bank_entries_are_pinned(path):
    cache = SemanticAnswerCache(WordEmbedder(), path=path, max_entries=1, ttl_hours=1)
    cache.put("Explain the Rankine cycle.", SIG, response("bank"), pinned=True)
    cache.put("Advantages of AC motor", SIG, response("a"))
    cache.put("Advantages of DC motor", SIG, response("b"))             # evicts "AC", never the bank
    cache.put("Explain the Rankine cycle.", SIG, response("student"))    # cannot overwrite the bank
    later = time.time() + 30 * 86400
    reloaded = SemanticAnswerCache(WordEmbedder(), path=path, max_entries=1, ttl_hours=1)
    original = time.time
    try:
        time.time = lambda: later
        assert reloaded.get("Explain the Rankine cycle.", SIG)["answer"] == "bank"   # never expires
        assert reloaded.get("Advantages of DC motor", SIG) is None                    # expired
    finally:
        time.time = original
    assert "Rankine" in path.read_text(encoding="utf-8")                 # our own bank question is readable


def test_a_new_model_or_index_drops_old_answers(path, monkeypatch):
    SemanticAnswerCache(WordEmbedder(), path=path).put("Explain the Rankine cycle.", SIG, response("A"))
    monkeypatch.setattr(settings, "LLM_SERVED_MODEL", "another-model")
    assert SemanticAnswerCache(WordEmbedder(), path=path).get("Explain the Rankine cycle.", SIG) is None
    monkeypatch.setattr(answer_cache, "CACHE_FORMAT", 999)
    assert SemanticAnswerCache(WordEmbedder(), path=path).stats()["entries"] == 0


@pytest.mark.parametrize("first,second,same", [
    ("Explain the Rankine cycle.", "What is the Rankine cycle?", True),
    ("Explain the Rankine cycle.", "Tell me about the Rankine cycle", True),
    ("Explain AC and DC motors.", "Compare AC and DC motors.", False),        # different intent
    ("Explain AC and DC motors.", "Difference between AC and DC motors", False),
    ("Explain the Rankine cycle.", "Why is the Rankine cycle used?", False),
    ("What is a transformer?", "What is not a transformer?", False),          # negation counts
    ("Explain Ohm's law.", "Derive Ohm's law.", False),
])
def test_same_words_rule(first, second, same):
    assert (answer_cache.word_set(first) == answer_cache.word_set(second)) is same


def test_a_half_written_line_is_ignored(path):
    SemanticAnswerCache(WordEmbedder(), path=path).put("Explain the Rankine cycle.", SIG, response("A"))
    with path.open("a", encoding="utf-8") as f:
        f.write('{"version": "trunc')                                    # power cut mid-write
    cache = SemanticAnswerCache(WordEmbedder(), path=path)
    assert cache.get("Explain the Rankine cycle.", SIG)["answer"] == "A"
    assert all(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())   # compacted
