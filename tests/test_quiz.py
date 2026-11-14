"""Practice quizzes: only well-formed questions whose answer is in the sources survive."""

import json

import pytest

from config import settings
from src.learning.quiz import QuizMaker, _json_array, validate
from src.llm.backends import GenerationResult

PASSAGES = [
    {"chunk_id": "r1", "article_title": "Rankine cycle", "similarity_score": 0.8,
     "text": "The Rankine cycle has four components: a pump, a boiler, a turbine and a condenser. "
             "Steam expands in the turbine to produce work."},
    {"chunk_id": "c1", "article_title": "Carnot cycle", "similarity_score": 0.7,
     "text": "The Carnot cycle is an idealised reversible cycle with two isothermal and two adiabatic processes."},
]
GOOD = {"question": "Where does steam expand to produce work in the Rankine cycle?",
        "options": ["In the turbine", "In the pump", "In the condenser", "In the feed tank"],
        "answer": 0, "explanation": "Steam expands in the turbine.", "source": 1}


def test_valid_grounded_questions_are_kept():
    kept, dropped = validate([GOOD], PASSAGES)
    assert len(kept) == 1 and dropped == 0 and kept[0].support >= 0.5


@pytest.mark.parametrize("bad", [
    {**GOOD, "options": ["In the turbine", "In the turbine", "x", "y"]},          # duplicate options
    {**GOOD, "options": ["a", "b", "c"]},                                          # three options
    {**GOOD, "answer": 7},                                                         # no such option
    {"question": "Too short?", "options": ["a", "b", "c", "d"], "answer": 0},      # not a real question
    {**GOOD, "options": ["In the flux capacitor", "In the pump", "In the condenser", "In the tank"]},  # unsupported
    {"options": ["a", "b", "c", "d"]},                                             # missing fields
])
def test_bad_questions_are_dropped(bad):
    kept, dropped = validate([bad], PASSAGES)
    assert kept == [] and dropped == 1


def test_wrong_source_number_is_repaired():
    kept, _ = validate([{**GOOD, "source": 2}], PASSAGES)                         # cites Carnot, fact is in Rankine
    assert kept and kept[0].source == 1


def test_json_is_found_in_messy_replies():
    reply = "Here are your questions:\n```json\n[" + json.dumps(GOOD) + "]\n```\nGood luck!"
    assert _json_array("no json here") == []
    assert len(_json_array(reply)) == 1
    assert len(_json_array("[" + json.dumps(GOOD) + ",]")) == 1                   # trailing comma tolerated


class QuizLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def generate(self, messages, **kw):
        self.calls += 1
        return GenerationResult(text=self.reply, model="fake", backend="fake")


class FakeRouter:
    calibration = {"in_domain_threshold": 0.5}


class FakeRetriever:
    def retrieve(self, question, top_k=5, filters=None):
        return [dict(p) for p in PASSAGES]


class FakePipeline:
    def __init__(self, reply):
        self.llm, self.retriever, self.router = QuizLLM(reply), FakeRetriever(), FakeRouter()


def test_quiz_is_made_once_then_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CONTEXT_ENGINEERING", False)
    bad = {**GOOD, "question": "Which is a made-up component of the cycle shown?",
           "options": ["Flux capacitor", "Pump", "Boiler", "Turbine"]}
    pump = {**GOOD, "question": "Which component raises the pressure of the water in the cycle?",
            "options": ["The pump", "The turbine", "The condenser", "The chimney"]}
    carnot = {**GOOD, "question": "Which idealised cycle has two isothermal and two adiabatic processes?",
              "options": ["The Carnot cycle", "The Otto cycle", "The Diesel cycle", "The Brayton cycle"],
              "source": 2}
    pipeline = FakePipeline(json.dumps([GOOD, bad, pump, carnot]))
    maker = QuizMaker(pipeline, path=tmp_path / "quizzes.jsonl")
    quiz = maker.make("Rankine cycle", subject_id="ME003", n=5)
    assert len(quiz.questions) == 3 and quiz.dropped == 1 and len(quiz.sources) == 2
    again = maker.make("rankine cycle", subject_id="ME003", n=5)
    assert pipeline.llm.calls == 1 and again.questions[0]["question"] == GOOD["question"]   # saved quiz reused
    maker.make("Rankine cycle", subject_id="ME003", n=5, fresh=True)
    assert pipeline.llm.calls == 2                                                           # new set on request
