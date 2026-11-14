"""Image-only questions: the vision model reads, the text model reasons (12 Nov fix)."""

from pathlib import Path

import pytest

from config import settings
from src.llm.backends import GenerationResult
from src.pipeline import EduRoutePipeline

IMAGE = Path(__file__).resolve().parent.parent / "evaluation" / "image_eval" / "truth_table.png"
TRANSCRIPT = "| A | B | Y |\n|---|---|---|\n| 0 | 0 | 0 |\n| 0 | 1 | 0 |\n| 1 | 0 | 0 |\n| 1 | 1 | 1 |"


class FakeModel:
    def encode(self, texts, **kw):
        import numpy as np
        return np.ones((len(texts) if isinstance(texts, list) else 1, 384), dtype="float32")


class FakeRetriever:
    version, model, metadata = "fake", FakeModel(), []

    def retrieve(self, question, top_k=5, filters=None):
        return [{"similarity_score": 0.2, "article_title": "Unrelated", "text": "x", "chunk_id": "c"}]


class Recorder:
    def __init__(self, name, reply):
        self.name, self.reply, self.calls = name, reply, []

    def is_available(self):
        return True

    def generate(self, messages, max_new_tokens=512, temperature=0.2, images=None, on_token=None, **kw):
        self.calls.append({"messages": messages, "images": images})
        return GenerationResult(text=self.reply, model=self.name, backend="fake")


@pytest.fixture
def backends():
    return Recorder("text", "This is an AND gate: Y is 1 only when A and B are both 1."), \
        Recorder("vision", TRANSCRIPT)


def ask(text, vision):
    pipeline = EduRoutePipeline(llm_backend=text, vision_backend=vision, retriever=FakeRetriever(), log=False)
    return pipeline.ask("Which logic gate does this truth table represent?",
                        image_bytes=IMAGE.read_bytes(), answer_mode="2-mark")


def test_vision_reads_then_text_model_answers(backends, monkeypatch):
    monkeypatch.setattr(settings, "VISION_TWO_STEP", True)
    text, vision = backends
    response = ask(text, vision)
    assert response.workflow == "vision_llm"
    assert len(vision.calls) == 1 and vision.calls[0]["images"]               # the image went to the vision model
    assert "TABLES" in vision.calls[0]["messages"][-1]["content"]              # asked to transcribe, not answer
    prompt = text.calls[0]["messages"][-1]["content"]
    assert TRANSCRIPT in prompt and "check each row" in prompt                 # text model reasons over the rows
    assert "AND gate" in response.answer and response.visual_description.startswith(TRANSCRIPT)
    # The exact truth-table check is handed to the text model as a checked fact.
    assert "CHECKED FACTS" in prompt and "matches the AND function" in prompt


def test_one_step_mode_still_available(backends, monkeypatch):
    monkeypatch.setattr(settings, "VISION_TWO_STEP", False)
    text, vision = backends
    response = ask(text, vision)
    assert response.workflow == "vision_llm"
    assert not text.calls and response.answer == TRANSCRIPT                    # the vision model answered alone
