"""Answer first, check afterwards: the draft comes back before verification runs."""

import pytest

from config import settings
from tests.test_agentic import CountingLLM, FakeRetriever


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    for flag in ("AGENTIC_RAG", "CONTEXT_ENGINEERING", "CONVERSATION_MEMORY", "ANSWER_CACHE"):
        monkeypatch.setattr(settings, flag, True)
    monkeypatch.setattr(settings, "NLI_MODEL", "")
    monkeypatch.setattr(settings, "LOG_DIR", tmp_path)

    def make(strategy="always_verify", log=False):
        from src.pipeline import EduRoutePipeline
        p = EduRoutePipeline(strategy=strategy, llm_backend=CountingLLM(), vision_backend=None,
                             retriever=FakeRetriever(), log=log)
        p.router.calibration = {**p.router.calibration, "in_domain_threshold": 0.5}
        p.telemetry_path = tmp_path / "telemetry.jsonl"
        return p
    return make


QUESTION = "Explain the Rankine cycle in a steam power plant."


def test_draft_first_then_checks(pipeline):
    p = pipeline()
    draft = p.ask(QUESTION, defer_checks=True)
    assert draft.pending_check and draft.verification is None
    assert draft.answer and draft.sources                                   # readable at once
    assert "draft_ready" in draft.telemetry["stages_ms"]
    final = draft.complete()
    assert not final.pending_check and final.verification is not None       # checked afterwards
    assert final.answer and final.telemetry["total_ms"] >= draft.telemetry["stages_ms"]["draft_ready"]


def test_deferred_and_direct_give_the_same_final_answer(pipeline):
    direct = pipeline().ask(QUESTION)
    deferred = pipeline().ask(QUESTION, defer_checks=True).complete()
    for key in ("answer", "workflow", "cited_sources", "verification", "notices"):
        assert direct.to_dict()[key] == deferred.to_dict()[key], key


def test_nothing_is_cached_or_logged_until_checked(pipeline, tmp_path):
    p = pipeline(strategy="adaptive", log=True)
    draft = p.ask(QUESTION, defer_checks=True)
    assert p.answer_cache.stats()["entries"] == 0
    assert not (tmp_path / "telemetry.jsonl").exists()
    draft.complete()
    assert p.answer_cache.stats()["entries"] == 1
    assert len((tmp_path / "telemetry.jsonl").read_text().splitlines()) == 1     # logged once
