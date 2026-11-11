"""
Fast unit tests (no model downloads, no network).

Run: venv\\Scripts\\python -m pytest -q
"""

import io

import numpy as np
import pytest
from PIL import Image

from src.llm.backends import ExtractiveBackend, LLMUnavailableError, OpenAICompatibleBackend
from src.llm.prompts import ANSWER_MODES, build_rag_messages, format_sources
from src.rag.text_cleaning import clean_article, split_sections
from src.router.query_analysis import analyse_query
from src.router.router import AdaptiveRouter, retrieval_signal
from src.verification.claims import EvidenceMapper, extract_claims, split_sentences
from src.verification.verifier import AnswerVerifier
from src.vision.image_input import ImageValidationError, prepare_image
from src.voice.speech import to_speakable

CALIBRATION = {"in_domain_threshold": 0.5, "strong_threshold": 0.7}


# ------------------------------------------------------------- cleaning

def test_section_cleaner_drops_reference_sections_anywhere():
    text = (
        "Lead paragraph about attention.\n\n== Method ==\nScaled dot-product attention.\n\n"
        "== See also ==\nList of models\n\n== References ==\nVaswani (2017). Retrieved 2024.\n"
        "== External links ==\nGoogle blog"
    )
    sections, dropped = clean_article(text)
    body = " ".join(s["text"] for s in sections)
    assert "Scaled dot-product attention" in body
    assert "List of models" not in body and "Retrieved" not in body
    assert "See also" in dropped["sections"]


def test_section_parser_handles_collapsed_v3_text():
    sections = split_sections("Intro text. == History == Old text. === Early === Earliest.")
    assert [s["heading_path"] for s in sections] == [[], ["History"], ["History", "Early"]]


def test_citation_filter_keeps_prose_with_et_al():
    text = "== History ==\nGraves et al. applied LSTM to speech recognition in 2004 with great success."
    sections, dropped = clean_article(text)
    assert sections and dropped["citation_paragraphs"] == 0


def test_offset_chunking_preserves_original_text():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    from src.rag.build_index_v4 import chunk_section

    try:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2",
                                                  local_files_only=True)
    except OSError:
        pytest.skip("embedding tokenizer not cached")
    text = "Self-attention lets a Transformer weigh tokens. " * 120
    chunks = chunk_section(text, tokenizer)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["text"] in text                 # exact original substring
        assert "##" not in chunk["text"]
        assert chunk["token_count"] <= 220


# ------------------------------------------------------------- routing

class FakeRetriever:
    def __init__(self, score):
        self.score = score

    def retrieve(self, question, top_k=5, filters=None):
        return [{"similarity_score": self.score - 0.01 * i, "article_title": f"A{i}", "text": "x"}
                for i in range(top_k)]


def make_router(score, strategy="adaptive", vision=False, **kw):
    router = AdaptiveRouter(retriever=FakeRetriever(score), strategy=strategy,
                            vision_available=vision, log_decisions=False, **kw)
    router.calibration = dict(CALIBRATION)
    return router


@pytest.mark.parametrize("question,mode,score,expected", [
    ("Explain backpropagation in neural networks.", "5-mark", 0.75, "rag_llm"),
    ("What is overfitting?", "2-mark", 0.6, "direct_llm"),
    ("Explain the Rankine cycle.", "5-mark", 0.3, "direct_llm"),
    ("hello, what can you do?", "conversational", 0.2, "direct_llm"),
    ("According to the syllabus, what is Q-learning?", "2-mark", 0.6, "rag_llm"),
])
def test_adaptive_routing(question, mode, score, expected):
    decision = make_router(score).route(question, answer_mode=mode)
    assert decision.workflow == expected, decision.reasons


def test_out_of_domain_answer_is_flagged_as_ungrounded():
    decision = make_router(0.3).route("Explain the Rankine cycle.")
    assert decision.fallback and not decision.grounded


def test_image_routes_to_hybrid_or_falls_back_without_vision():
    q = "Explain backpropagation in neural networks using the attached notes"
    assert make_router(0.6, vision=True).route(q, has_image=True).workflow == "hybrid_multimodal_rag"
    fallback = make_router(0.6, vision=False).route(q, has_image=True)
    assert fallback.workflow == "rag_llm" and "vision" in fallback.fallback.lower()


def test_questions_about_the_image_itself_go_to_the_vision_model():
    router = make_router(0.6, vision=True)
    for q in ("Which logic gate does this truth table represent?",
              "Explain how the layers in this diagram work",
              "Find the total resistance in the circuit shown"):
        assert router.route(q, has_image=True).workflow == "vision_llm", q
    # An explicit request for sources keeps the knowledge base in the loop.
    assert router.route("According to the sources, what does this circuit do?",
                        has_image=True).workflow == "hybrid_multimodal_rag"


@pytest.mark.parametrize("strategy,workflow,verify", [
    ("direct", "direct_llm", False),
    ("fixed_rag", "rag_llm", False),
    ("always_verify", "rag_llm", True),
    ("modality", "rag_llm", False),
])
def test_baseline_strategies(strategy, workflow, verify):
    decision = make_router(0.7, strategy=strategy).route("Explain gradient descent.")
    assert (decision.workflow, decision.verify) == (workflow, verify)


def test_adaptive_verifies_only_with_triggers():
    simple = make_router(0.75).route("Explain backpropagation.", answer_mode="5-mark")
    assert simple.workflow == "rag_llm" and simple.verify_eligible and not simple.verify
    complex_q = make_router(0.75).route(
        "Compare LSTM and GRU and also explain the vanishing gradient problem.", answer_mode="10-mark")
    assert complex_q.verify and complex_q.verify_triggers


def test_retrieval_signal_strength():
    rows = [{"similarity_score": 0.8, "article_title": "A"}, {"similarity_score": 0.6, "article_title": "B"}]
    assert retrieval_signal(rows, CALIBRATION)["strength"] == "strong"
    rows[0]["similarity_score"] = 0.4
    assert retrieval_signal(rows, CALIBRATION)["in_domain"] is False


def test_query_analysis_signals():
    a = analyse_query("Compare CNN and RNN and also explain pooling in detail for my exam", answer_mode="10-mark")
    assert a.intent == "comparison" and a.complexity >= 3
    assert analyse_query("it?").ambiguous


# --------------------------------------------------------- verification

class KeywordEmbedder:
    """Deterministic stand-in for the sentence-transformer (bag of keywords)."""
    VOCAB = ["gradient", "chain", "rule", "weights", "loss", "layer", "banana", "1986", "1990", "network"]

    def encode(self, texts, **_):
        out = np.array([[t.lower().count(w) for w in self.VOCAB] for t in texts], dtype=np.float32) + 1e-3
        return out / np.linalg.norm(out, axis=1, keepdims=True)


PASSAGES = [{"chunk_id": "c1", "text": "Backpropagation computes the gradient of the loss with respect "
             "to the weights using the chain rule. It was popularised in 1986."}]


def test_claim_extraction_skips_headings_and_meta():
    answer = "## Introduction\nBackpropagation uses the chain rule to compute gradients [1].\nIn summary, it works."
    claims = extract_claims(answer)
    assert [c.text for c in claims] == ["Backpropagation uses the chain rule to compute gradients."]
    assert claims[0].cited_sources == [1]


def test_verifier_flags_unsupported_and_wrong_numbers():
    verifier = AnswerVerifier(KeywordEmbedder())
    answer = ("Backpropagation computes the gradient of the loss using the chain rule [1]. "
              "Bananas are an important layer type in every banana network [1]. "
              "The method was popularised in 1990 [3].")
    report = verifier.verify(answer, PASSAGES)
    labels = [c.label for c in report.claims]
    assert labels[0] == "supported"
    assert labels[1] == "unsupported"
    assert not report.claims[2].numbers_ok
    assert report.invalid_citations == 1
    assert verifier.needs_revision(report)


def test_evidence_cache_is_used_when_complete():
    class Cache:
        def vectors_for(self, chunk_id):
            return np.ones((4, 10), dtype=np.float32) / np.sqrt(10)

    mapper = EvidenceMapper(KeywordEmbedder(), evidence_cache=Cache())
    from src.verification.claims import evidence_windows
    windows = evidence_windows(PASSAGES)
    assert len(windows) == 3  # 2 sentences + 1 pair -> mismatch => recompute
    assert mapper._window_vectors(PASSAGES, windows).shape == (3, 10)


# ------------------------------------------------------------------ llm

def test_prompts_number_sources_and_include_mode_instructions():
    messages = build_rag_messages("What is backprop?", PASSAGES * 2, "10-mark")
    user = messages[-1]["content"]
    assert "[1]" in user and "[2]" in user and "Conclusion" in user
    assert "does not contain enough information" in user
    assert set(ANSWER_MODES) >= {"2-mark", "5-mark", "10-mark"}
    assert format_sources([{"article_title": "T", "section": "S", "text": "x" * 2000}]).count("...") == 1


def test_unreachable_server_raises_and_is_cached():
    backend = OpenAICompatibleBackend("http://127.0.0.1:9/v1", "m")
    with pytest.raises(LLMUnavailableError):
        backend.generate([{"role": "user", "content": "hi"}])
    assert not backend.is_available()  # cached; no second connection attempt


def test_extractive_backend_is_labelled():
    result = ExtractiveBackend().generate([{"role": "user", "content": "q"}],
                                          evidence={"question": "q", "passages": PASSAGES})
    assert not result.is_model_output and "[1]" in result.text


# ------------------------------------------------------- vision / voice

def _png(size=(64, 64), mode="RGBA"):
    buffer = io.BytesIO()
    Image.new(mode, size, (255, 0, 0, 128) if mode == "RGBA" else 128).save(buffer, format="PNG")
    return buffer.getvalue()


def test_image_validation():
    prepared = prepare_image(_png())
    assert prepared.mime == "image/png" and prepared.width == 64
    big = prepare_image(_png((3000, 1000), "RGB"), max_side=1536)
    assert big.resized and max(big.width, big.height) == 1536
    with pytest.raises(ImageValidationError):
        prepare_image(b"%PDF-1.4 not an image")
    with pytest.raises(ImageValidationError):
        prepare_image(_png(), max_bytes=10)


def test_speakable_text():
    spoken = to_speakable("## CNNs vs RNNs\n- A **CNN** uses filters [1], e.g. kernels. See https://x.y")
    assert "[1]" not in spoken and "http" not in spoken and "#" not in spoken
    assert "C N N" in spoken and "for example" in spoken and " ," not in spoken


def test_split_sentences_strips_markdown():
    assert split_sentences("- **Bold** point one. Second point.\n1. Third") == [
        "Bold point one.", "Second point.", "Third"]


# ------------------------------------------------------- multi-branch filters

def test_filters_match_any_branch_or_subject_of_shared_article():
    from src.rag.retrieval import EduRouteRetriever

    retriever = EduRouteRetriever.__new__(EduRouteRetriever)
    retriever.metadata = [
        {"discipline": "EEE", "disciplines": ["ECE", "EEE"], "subject_id": "EEE007",
         "subject_ids": ["ECE003", "EEE007"]},                       # shared article
        {"discipline": "ME", "subject_id": "ME008"},                  # legacy record, no lists
    ]
    assert retriever.allowed_ids({"discipline": "ECE"}).tolist() == [0]
    assert retriever.allowed_ids({"subject_id": "eee007"}).tolist() == [0]
    assert retriever.allowed_ids({"subject_id": "ME008"}).tolist() == [1]
    assert retriever.allowed_ids({"discipline": ["ME", "ECE"]}).tolist() == [0, 1]
    assert retriever.allowed_ids({"discipline": None}) is None


class ScopedRetriever:
    """Weak match inside the subject, strong match in the wider branch."""
    scores = {"subject_id": 0.35, "discipline": 0.76, None: 0.80}

    def retrieve(self, question, top_k=5, filters=None):
        key = next(iter(filters)) if filters else None
        return [{"similarity_score": self.scores[key], "article_title": f"{key}", "text": "x"}]


def _scoped_pipeline():
    from src.pipeline import EduRoutePipeline
    pipeline = object.__new__(EduRoutePipeline)  # no model loading
    pipeline.retriever = ScopedRetriever()
    pipeline.router = make_router(0.5)
    return pipeline


def test_subject_filter_widens_to_branch_when_subject_has_no_match():
    pipeline = _scoped_pipeline()
    subject = {"subject_id": "CHE010"}
    results, scope = pipeline._scoped_retrieve("q", subject, [{"discipline": "CHE"}, None])
    assert scope == {"discipline": "CHE"} and results[0]["similarity_score"] == 0.76
    # Strict mode keeps the filter even when it is weak.
    results, scope = pipeline._scoped_retrieve("q", subject, [])
    assert scope == subject


def test_strong_subject_match_keeps_the_filter():
    pipeline = _scoped_pipeline()
    pipeline.retriever.scores = {**ScopedRetriever.scores, "subject_id": 0.7}
    _, scope = pipeline._scoped_retrieve("q", {"subject_id": "X"}, [None])
    assert scope == {"subject_id": "X"}


def test_wrong_number_counts_as_unsupported_even_when_wording_matches():
    # Regression (Colab run, 8 Nov 2026): a near-copy of the evidence with an
    # invented figure was labelled "partial", so the unsupported claim rate read 0.0.
    verifier = AnswerVerifier(KeywordEmbedder())
    answer = "Backpropagation computes the gradient of the loss with respect to the weights in 1990."
    report = verifier.verify(answer, PASSAGES)
    claim = report.claims[0]
    assert claim.missing_numbers == ["1990"]
    assert claim.label == "unsupported"
    assert report.unsupported_claim_rate == 1.0


MOTOR_PASSAGES = [
    {"chunk_id": "m1", "text": "Backpropagation uses the chain rule on the network weights."},
    {"chunk_id": "m2", "text": "A banana layer was described in 1990."},
]


def test_number_must_come_from_a_relevant_passage():
    # 1990 is in the evidence, but only in an unrelated passage.
    verifier = AnswerVerifier(KeywordEmbedder())
    report = verifier.verify("Backpropagation uses the chain rule on the network weights since 1990.",
                             MOTOR_PASSAGES)
    claim = report.claims[0]
    assert claim.missing_numbers == ["1990"] and claim.label == "unsupported"


class FakeNLI:
    """Contradicts any claim mentioning 'six'; entails claims containing 'chain'."""
    def judge(self, pairs):
        return ["contradiction" if "six" in h else "entailment" if "chain" in h else "neutral"
                for _, h in pairs]


def test_nli_contradiction_counts_only_for_claims_with_figures():
    verifier = AnswerVerifier(KeywordEmbedder(), entailment_fn=FakeNLI())
    answer = ("Backpropagation computes the gradient of the loss using the chain rule [1]. "
              "Backpropagation computes the gradient of the loss for six weights layer. "
              "Backpropagation was popularised for six weights in 1986.")
    report = verifier.verify(answer, PASSAGES)
    first, wordy, figure = report.claims
    assert first.nli == "entailment" and first.label == "supported"
    # No figure: a possible contradiction is recorded but not counted (NLI was
    # unreliable on such claims in the 10 Nov Colab run).
    assert wordy.nli == "possible_contradiction" and wordy.label != "unsupported"
    assert figure.nli == "contradiction" and figure.label == "unsupported"
    assert any("contradicted by source" in i for i in figure.issues)
    assert verifier.needs_revision(report)


def test_fragments_are_not_checked_as_claims():
    answer = ("Backpropagation computes gradients with the chain rule.\n"
              "\\[ \\frac{dy}{dt} = -\\lambda y \\]\n"
              "where y is the amount of the substance at time t.\n"
              "H = -(9/14 x log 9/14) - (5/14 x log 5/14) = 0.940\n"
              "Do not invent source titles or URLs.")
    assert [c.text for c in extract_claims(answer)] == ["Backpropagation computes gradients with the chain rule."]


def test_number_parsing():
    from src.verification.claims import numbers_in
    assert numbers_in("A 1,500 RPM motor draws 0.4 A") == {"1500", "0.4"}
    assert numbers_in("Points (1,1) and (5,5)") == {"1", "5"}          # not 11 and 55
    assert numbers_in("a 32x32 image and a 5 × 5 kernel") == {"32", "5"}
    assert numbers_in("the address 192.168.1.1 is private") == set()  # not a figure
    assert numbers_in("pi is about 3.14.") == {"3.14"}


def test_figures_from_the_question_need_no_source():
    verifier = AnswerVerifier(KeywordEmbedder())
    answer = "The network loss uses the chain rule gradient for 9 weights and 5 layers."
    without = verifier.verify(answer, PASSAGES)
    with_question = verifier.verify(answer, PASSAGES, question="A network has 9 weights and 5 layers.")
    assert without.claims[0].missing_numbers == ["5", "9"]
    assert with_question.claims[0].missing_numbers == []


def test_prompt_echo_is_removed_from_answers():
    from src.llm.prompts import strip_prompt_echo
    answer = ("Answer format:\n"
              "Harry Potter is a novel series [1].\n"
              "- Example rule: take examples from the SOURCES. If they contain no suitable example, "
              "give a short qualitative example without new numbers.\n"
              "Do not pad the answer with unsupported details just to reach the length; a shorter "
              "accurate answer is better.\n"
              "The format of the answer matters less than its accuracy.")
    assert strip_prompt_echo(answer) == ("Harry Potter is a novel series [1].\n"
                                         "The format of the answer matters less than its accuracy.")


def test_prompts_forbid_invented_numerical_examples():
    from src.llm.prompts import build_revision_messages
    rag = build_rag_messages("Explain slip.", PASSAGES, "5-mark")[1]["content"]
    assert "Do not make up numerical worked examples" in rag
    assert rag.rstrip().endswith("[2][3].")  # citation reminder stays last
    rev = build_revision_messages("q", "draft", PASSAGES, ["x 1500 RPM"], "5-mark")[1]["content"]
    assert "never keep the figure" in rev and "Do not add sentences about what the knowledge base" in rev
