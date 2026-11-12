"""Agentic RAG, context engineering, conversation memory and the answer cache."""

import time

import numpy as np
import pytest

from config import settings
from src.agent import memory
from src.agent.agentic_rag import AgenticRetriever, coverage, merge, plan
from src.inference.answer_cache import SemanticAnswerCache
from src.llm.backends import GenerationResult
from src.llm.prompts import GROUNDING_RULES, build_rag_messages
from src.rag.context import ContextBuilder


def passage(cid, text, score=0.6, title=None):
    return {"chunk_id": cid, "text": text, "similarity_score": score, "article_title": title or cid}


# ---------------------------------------------------------------- planning

@pytest.mark.parametrize("question,expected", [
    ("What is the difference between the Rankine cycle and the Carnot cycle?", ["Rankine cycle", "Carnot cycle"]),
    ("Compare TCP and UDP.", ["TCP", "UDP"]),
    ("AC motor vs DC motor", ["AC motor", "DC motor"]),
    ("Explain the Rankine cycle.", ["Explain the Rankine cycle."]),
])
def test_plan_splits_comparisons(question, expected):
    assert plan(question) == expected


def test_plan_splits_multi_part_questions():
    parts = plan("What is a transformer? Explain its working principle and also explain eddy current losses")
    assert len(parts) == 3


def test_coverage_and_merge():
    ps = [passage("a", "The Rankine cycle uses a boiler and turbine."), passage("b", "Carnot efficiency.")]
    assert coverage("Rankine cycle turbine", ps) == 1.0
    assert coverage("Rankine cycle condenser", ps) == pytest.approx(2 / 3)
    merged = merge([[passage("a", "x"), passage("b", "y")], [passage("c", "z"), passage("a", "x")]], 10)
    assert [p["chunk_id"] for p in merged] == ["a", "c", "b"]        # interleaved, de-duplicated


class Corpus:
    """search() returns passages whose text shares words with the query."""

    def __init__(self, docs):
        self.docs, self.queries = docs, []

    def search(self, query, k):
        self.queries.append(query)
        q = set(query.lower().split())
        scored = [(len(q & set(d["text"].lower().split())), d) for d in self.docs]
        return [{**d, "similarity_score": 0.5 + 0.05 * s} for s, d in sorted(scored, key=lambda x: -x[0]) if s][:k]


DOCS = [passage("r1", "rankine cycle boiler turbine condenser pump steam"),
        passage("c1", "carnot cycle reversible isothermal adiabatic efficiency"),
        passage("x1", "unrelated welding arc electrode")]


def test_agent_retrieves_each_side_of_a_comparison():
    corpus = Corpus(DOCS)
    run = AgenticRetriever(corpus.search, threshold=0.4).run(
        "Difference between the rankine cycle and the carnot cycle", top_k=4)
    ids = {p["chunk_id"] for p in run.passages}
    assert {"r1", "c1"} <= ids
    assert [s.action for s in run.trace][:3] == ["plan", "retrieve", "retrieve"]
    assert run.coverage == 1.0


def test_agent_refines_when_coverage_is_low():
    corpus = Corpus(DOCS)
    probe = [DOCS[0]]                                   # covers rankine, not carnot
    run = AgenticRetriever(corpus.search, threshold=0.4, min_coverage=0.9).run(
        "rankine carnot efficiency", top_k=3, probe=probe)
    assert any(s.action == "refine" for s in run.trace)
    assert "c1" in {p["chunk_id"] for p in run.passages}
    assert "carnot" in corpus.queries[-1]               # searched for what was missing


def test_agent_correction_adds_only_new_relevant_evidence():
    corpus = Corpus(DOCS)
    agent = AgenticRetriever(corpus.search, threshold=0.5)
    new, steps = agent.correct(["the carnot cycle is reversible"], existing=[DOCS[0]])
    assert [p["chunk_id"] for p in new] == ["c1"] and steps[0].action == "correct"
    again, _ = agent.correct(["the carnot cycle is reversible"], existing=[DOCS[0], DOCS[1]])
    assert again == []


# --------------------------------------------------------- context engineering

def test_context_builder_dedupes_compresses_and_budgets():
    long_text = " ".join(f"Sentence {i} talks about welding arcs and electrodes." for i in range(40)) + \
        " The Rankine cycle has a boiler, a turbine, a condenser and a pump."
    ps = [passage("a", long_text, 0.7), passage("a2", long_text, 0.69),
          passage("b", "The Rankine cycle is used in steam power plants. " * 5, 0.6)]
    builder = ContextBuilder(embedder=None)                       # lexical scoring
    selected, stats = builder.build("Rankine cycle condenser turbine", ps, "2-mark", budget=200)
    assert stats.duplicates_removed == 1
    assert stats.tokens_after <= 200 < stats.tokens_before
    assert "condenser" in selected[0]["prompt_text"]              # the relevant sentence survived
    assert selected[0]["text"] == long_text                       # full text kept for verification


def test_stable_prefix_puts_fixed_rules_in_the_system_message():
    ps = [passage("a", "x")]
    plain = build_rag_messages("q?", ps)
    cached = build_rag_messages("q?", ps, stable_prefix=True)
    assert GROUNDING_RULES in cached[0]["content"] and GROUNDING_RULES not in cached[1]["content"]
    assert GROUNDING_RULES in plain[1]["content"]
    assert cached[1]["content"].rstrip().endswith("[2][3].")      # citation reminder still last
    other = build_rag_messages("another question", [passage("b", "y")], stable_prefix=True)
    assert other[0] == cached[0]                                  # identical prefix -> cache reuse


# ---------------------------------------------------------------- memory

HISTORY = [{"question": "Explain the Rankine cycle.", "answer": "The Rankine cycle converts heat into work [1]. "
            "It uses a boiler, turbine, condenser and pump [2]. Steam drives the turbine."}]


@pytest.fixture
def memory_on(monkeypatch):
    monkeypatch.setattr(settings, "CONVERSATION_MEMORY", True)


@pytest.mark.parametrize("question,follow_up", [
    ("What are its advantages?", True), ("Why?", True), ("Give an example", True),
    ("Explain backpropagation in neural networks.", False),
])
def test_follow_ups_are_resolved(memory_on, question, follow_up):
    query, topic = memory.contextualize(question, HISTORY)
    assert (topic is not None) == follow_up
    if follow_up:
        assert "rankine" in query.lower()


def test_memory_is_budgeted_and_summarises_older_turns(memory_on):
    history = [{"question": f"Explain topic number {i} in thermodynamics", "answer": "A long answer. " * 80}
               for i in range(12)]
    text = memory.render(history, budget_tokens=120)
    assert len(text) // 4 <= 220
    assert "Earlier topics" in text and "topic number 11" in text   # newest kept, older summarised
    assert "[1]" not in memory.render(HISTORY)                       # citations stripped


def test_memory_off_keeps_questions_unchanged(monkeypatch):
    monkeypatch.setattr(settings, "CONVERSATION_MEMORY", False)
    assert memory.contextualize("What are its advantages?", HISTORY) == ("What are its advantages?", None)


# ---------------------------------------------------------------- answer cache

class WordEmbedder:
    """Bag-of-words vectors: paraphrases with the same words are near-identical."""
    VOCAB = ["explain", "rankine", "cycle", "ac", "dc", "motor", "advantages", "of", "the", "please", "briefly"]

    def encode(self, texts, **kw):
        out = []
        for t in texts:
            v = np.array([t.lower().count(w) for w in self.VOCAB], dtype=np.float32) + 0.01
            out.append(v / np.linalg.norm(v))
        return np.vstack(out)


def test_answer_cache_exact_and_semantic_hits():
    cache = SemanticAnswerCache(WordEmbedder(), max_entries=10, ttl_hours=1, similarity=0.95)
    sig = SemanticAnswerCache.signature("5-mark", None, "adaptive")
    cache.put("Explain the Rankine cycle.", sig, {"answer": "A"})
    assert cache.get("explain the rankine cycle", sig) == {"answer": "A"}          # exact tier
    assert cache.get("The Rankine cycle: explain", sig) == {"answer": "A"}         # same words, reordered
    assert cache.get("Explain the Rankine cycle.", ("2-mark", (), "adaptive")) is None   # other format


def test_answer_cache_never_confuses_different_topics():
    cache = SemanticAnswerCache(WordEmbedder(), similarity=0.5)        # even with a loose threshold
    sig = SemanticAnswerCache.signature("5-mark", None, "adaptive")
    cache.put("Advantages of AC motor", sig, {"answer": "AC"})
    assert cache.get("Advantages of DC motor", sig) is None             # different content words


def test_answer_cache_is_bounded_and_expires(monkeypatch):
    cache = SemanticAnswerCache(None, max_entries=2, ttl_hours=1)
    sig = ("5-mark", (), "adaptive")
    for q in ("one question here", "two question here", "three question here"):
        cache.put(q, sig, {"q": q})
    assert cache.get("one question here", sig) is None and cache.stats()["entries"] == 2   # LRU evicted
    later = time.time() + 7200
    monkeypatch.setattr(time, "time", lambda: later)
    assert cache.get("three question here", sig) is None                               # expired


# --------------------------------------------------- the pipeline, end to end

class FakeModel(WordEmbedder):
    pass


class FakeRetriever:
    version = "fake"

    def __init__(self):
        self.model, self.metadata, self.queries = FakeModel(), [], []

    def embed_query(self, q):
        return self.model.encode([q])

    def retrieve(self, question, top_k=5, filters=None):
        self.queries.append(question)
        corpus = Corpus([passage("r1", "rankine cycle boiler turbine condenser pump steam power plant", title="Rankine cycle"),
                         passage("c1", "carnot cycle reversible isothermal adiabatic efficiency", title="Carnot cycle")])
        found = corpus.search(question, top_k)
        return [{**p, "similarity_score": 0.75} for p in found] or \
            [{**passage("z", "unrelated text"), "similarity_score": 0.1}]


class CountingLLM:
    name = "fake"

    def __init__(self):
        self.calls = []

    def is_available(self):
        return True

    def health_check(self):
        return True, "fake"

    def generate(self, messages, max_new_tokens=512, temperature=0.2, on_token=None, **kw):
        self.calls.append(messages)
        return GenerationResult(text="The Rankine cycle uses a boiler and a turbine [1].", model="fake", backend="fake")


@pytest.fixture
def all_on(monkeypatch):
    for flag in ("AGENTIC_RAG", "CONTEXT_ENGINEERING", "CONVERSATION_MEMORY", "ANSWER_CACHE"):
        monkeypatch.setattr(settings, flag, True)
    monkeypatch.setattr(settings, "NLI_MODEL", "")


def make_pipeline():
    from src.pipeline import EduRoutePipeline
    llm = CountingLLM()
    p = EduRoutePipeline(llm_backend=llm, vision_backend=None, retriever=FakeRetriever(), log=False)
    p.router.calibration = {**p.router.calibration, "in_domain_threshold": 0.5}
    return p, llm


def test_pipeline_uses_agent_context_and_cache(all_on):
    p, llm = make_pipeline()
    first = p.ask("Explain the Rankine cycle in a steam power plant.", answer_mode="5-mark")
    assert first.workflow == "rag_llm"
    assert first.telemetry["agent_trace"][0]["action"] == "plan"
    assert first.telemetry["context"]["passages_out"] >= 1
    calls = len(llm.calls)
    again = p.ask("explain the rankine cycle in a steam power plant", answer_mode="5-mark")
    assert again.telemetry["cache_hit"] and len(llm.calls) == calls          # no model call
    assert again.answer == first.answer and "stored answer" in again.notices[0]


def test_pipeline_resolves_follow_ups_for_retrieval(all_on):
    p, _ = make_pipeline()
    response = p.ask("What are its advantages?", answer_mode="2-mark", history=HISTORY)
    assert response.telemetry["memory"]["follow_up"]
    assert any("rankine" in q.lower() for q in p.retriever.queries)          # searched with the topic
    assert response.telemetry["cache_hit"] is False                          # follow-ups never cached
