"""
Adaptive workflow router (report §8).

Workflows
  direct_llm             question -> LLM
  rag_llm                question -> retrieval -> LLM with sources
  vision_llm             image + question -> vision model -> answer
  hybrid_multimodal_rag  image -> visual extraction; question -> retrieval; combined -> LLM
  (+ selective verification as a flag on any workflow that has evidence)

Strategies (report §22.1 baselines share this code path so comparisons are fair)
  direct         always direct_llm (vision_llm when an image is attached)
  fixed_rag      always rag_llm, never verify
  always_verify  always rag_llm, always verify
  modality       image -> vision_llm, otherwise rag_llm; never verify
  adaptive       the proposed policy below

Adaptive policy (balanced cost preference)
  1. Image attached: hybrid_multimodal_rag if the question also needs domain
     knowledge (in-domain probe or conceptual intent), else vision_llm.
  2. General chat / study-skills: direct_llm.
  3. Explicit request for sources or course material: rag_llm.
  4. Out-of-domain probe (top cosine below the calibrated threshold): direct_llm,
     with a note that the answer is not grounded in the knowledge base.
  5. Simple short answer (2-mark definition, complexity 1): direct_llm.
     (economy: also 5-mark definitions; quality: never - always rag in-domain)
  6. Otherwise: rag_llm.
  Verification runs only when the workflow has evidence and at least one
  trigger fires (report §14.1).

The retrieval probe is one query embedding plus one FAISS search (~15-30 ms on
this CPU) and its results are reused by the RAG workflow, so probing adds
almost no cost when RAG is selected.
"""

import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from config import settings
from src.router.query_analysis import analyse_query
from src.utils.io import append_jsonl, read_json
from src.verification.verifier import verification_triggers

logger = logging.getLogger(__name__)

WORKFLOWS = ("direct_llm", "rag_llm", "vision_llm", "hybrid_multimodal_rag")
STRATEGIES = ("adaptive", "direct", "fixed_rag", "always_verify", "modality")

# Used only if config/router_calibration.json has not been generated yet.
DEFAULT_CALIBRATION = {"in_domain_threshold": 0.40, "strong_threshold": 0.55, "source": "default"}


def load_calibration(version):
    """Calibration fitted on this index; uncalibrated defaults are flagged as such."""
    path = settings.router_calibration_path(version)
    if path.exists():
        data = read_json(path)
        data["source"] = str(path.name)
        return data
    logger.warning("No router calibration for index '%s'; using uncalibrated defaults. "
                   "Run: python -m evaluation.calibrate_router --version %s", version, version)
    return {**DEFAULT_CALIBRATION, "source": f"UNCALIBRATED default (no {path.name})"}


@dataclass
class RoutingDecision:
    workflow: str
    strategy: str
    verify: bool
    verify_eligible: bool
    reasons: list
    components: list
    verify_triggers: list = field(default_factory=list)
    fallback: str | None = None
    grounded: bool = False
    retrieval_signal: dict | None = None
    analysis: dict | None = None
    routing_ms: float = 0.0

    def to_dict(self):
        return asdict(self)


def retrieval_signal(results, calibration):
    """Summarise a retrieval probe. Scores are ranking signals, not probabilities."""
    if not results:
        return {"top_score": 0.0, "strength": "none", "in_domain": False}
    scores = [r["similarity_score"] or 0.0 for r in results]
    top = scores[0]
    second = scores[1] if len(scores) > 1 else 0.0
    articles = [r["article_title"] for r in results]
    in_domain = top >= calibration["in_domain_threshold"]
    if top >= calibration["strong_threshold"]:
        strength = "strong"
    elif in_domain:
        strength = "moderate"
    else:
        strength = "weak"
    return {
        "top_score": round(top, 4),
        "margin": round(top - second, 4),
        "distinct_articles_top5": len(set(articles[:5])),
        "strength": strength,
        "in_domain": in_domain,
        # Two different articles nearly tied at the top: possible disagreement.
        "conflict_risk": len(articles) > 1 and articles[0] != articles[1] and (top - second) < 0.02,
        "thresholds": {k: calibration[k] for k in ("in_domain_threshold", "strong_threshold")},
    }


class AdaptiveRouter:

    def __init__(self, retriever=None, strategy="adaptive", cost_preference="balanced",
                 vision_available=False, log_decisions=True):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if cost_preference not in ("economy", "balanced", "quality"):
            raise ValueError("cost_preference must be economy, balanced or quality")
        self.retriever = retriever
        self.strategy = strategy
        self.cost_preference = cost_preference
        self.vision_available = vision_available
        self.log_decisions = log_decisions
        version = getattr(retriever, "version", settings.RAG_VERSION)
        self.calibration = load_calibration(version)
        self.log_path = settings.LOG_DIR / "routing_decisions.jsonl"

    def probe(self, question, filters=None):
        if self.retriever is None or not question.strip():
            return []
        return self.retriever.retrieve(question, top_k=settings.TOP_K, filters=filters)

    def route(self, question, has_image=False, from_voice=False, answer_mode="5-mark",
              history=None, filters=None, probe_results=None):
        started = time.perf_counter()
        analysis = analyse_query(question, has_image, from_voice, answer_mode, history)

        needs_probe = self.strategy == "adaptive" and question.strip()
        if probe_results is None and needs_probe:
            probe_results = self.probe(question, filters)
        signal = retrieval_signal(probe_results, self.calibration) if probe_results is not None else None

        decide = getattr(self, f"_route_{self.strategy}")
        workflow, verify, reasons, fallback = decide(analysis, signal)

        if workflow in ("vision_llm", "hybrid_multimodal_rag") and not self.vision_available:
            fallback = (
                "No vision model is configured. The image could not be analysed; "
                + ("answering the text question from the knowledge base instead."
                   if question.strip() else "please describe the image in words.")
            )
            reasons.append("vision unavailable -> fallback to rag_llm on the text question")
            workflow = "rag_llm" if question.strip() else "direct_llm"

        # `verify_eligible`: the workflow has evidence to check against.
        # Adaptive runs verification only when a trigger fires; triggers that
        # depend on the generated answer (dates, figures) are added later by
        # the pipeline. always_verify verifies unconditionally.
        verify_eligible = verify
        triggers = []
        if verify:
            triggers = verification_triggers(analysis, signal, answer_mode=answer_mode)
            if self.strategy == "always_verify":
                triggers = triggers or ["always_verify baseline"]
            elif self.strategy == "adaptive":
                verify = bool(triggers)

        components = {
            "direct_llm": ["llm"],
            "rag_llm": ["embedder", "faiss", "llm"],
            "vision_llm": ["vision_model"],
            "hybrid_multimodal_rag": ["vision_model", "embedder", "faiss", "llm"],
        }[workflow] + (["verifier"] if verify else [])
        if from_voice:
            components = ["asr"] + components

        decision = RoutingDecision(
            workflow=workflow,
            strategy=self.strategy,
            verify=verify,
            verify_eligible=verify_eligible,
            reasons=reasons,
            components=components,
            verify_triggers=triggers,
            fallback=fallback,
            grounded=workflow in ("rag_llm", "hybrid_multimodal_rag"),
            retrieval_signal=signal,
            analysis=analysis.to_dict(),
            routing_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        if self.log_decisions:
            record = decision.to_dict()
            if not settings.LOG_QUERY_TEXT:
                record["analysis"] = {**record["analysis"], "question": None}
            append_jsonl(self.log_path, {
                "timestamp": datetime.now(timezone.utc).isoformat(), **record
            })
        return decision

    # ----------------------------------------------------------- strategies

    def _route_direct(self, analysis, signal):
        if analysis.modality.startswith("image"):
            return "vision_llm", False, ["baseline: direct (image -> vision)"], None
        return "direct_llm", False, ["baseline: always direct LLM"], None

    def _route_fixed_rag(self, analysis, signal):
        return "rag_llm", False, ["baseline: always retrieve"], None

    def _route_always_verify(self, analysis, signal):
        return "rag_llm", True, ["baseline: always retrieve and verify"], None

    def _route_modality(self, analysis, signal):
        if analysis.modality.startswith("image"):
            return "vision_llm", False, ["baseline: modality router (image -> vision)"], None
        return "rag_llm", False, ["baseline: modality router (text -> RAG)"], None

    def _route_adaptive(self, analysis, signal):
        reasons = list(analysis.reasons)
        in_domain = bool(signal and signal["in_domain"])
        if signal:
            reasons.append(
                f"retrieval probe top cosine {signal['top_score']} ({signal['strength']}; "
                f"in-domain threshold {signal['thresholds']['in_domain_threshold']})"
            )

        # 1. Image questions
        if analysis.modality.startswith("image"):
            # The answer is in the image itself ("this truth table", "the circuit
            # shown"): on the 11 Nov image benchmark the vision-only route was at
            # least as accurate on every such question and ~3.5x faster than
            # hybrid. Hybrid stays for explicit source requests.
            if analysis.image_dependent and not analysis.evidence_requested:
                reasons.append("question is about the image's own content -> vision LLM")
                return "vision_llm", False, reasons, None
            conceptual = analysis.intent in ("explanation", "definition", "comparison")
            if in_domain or (conceptual and analysis.word_count >= 6):
                reasons.append("image + domain knowledge needed -> hybrid multimodal RAG")
                return "hybrid_multimodal_rag", True, reasons, None
            reasons.append("answer is contained in the image -> vision LLM")
            return "vision_llm", False, reasons, None

        # 2. Small talk / study skills
        if analysis.general_chat:
            reasons.append("no academic evidence needed -> direct LLM")
            return "direct_llm", False, reasons, None

        # 3. Explicit request for sources
        if analysis.evidence_requested:
            reasons.append("student asked for source-backed content -> RAG")
            return "rag_llm", True, reasons, (
                None if in_domain else "The knowledge base may not cover this topic."
            )

        # 4. Out of domain for the knowledge base
        if signal is not None and not in_domain:
            reasons.append("knowledge base does not cover this topic -> direct LLM (ungrounded)")
            return "direct_llm", False, reasons, (
                "This topic is outside the current knowledge base, so the answer is "
                "not backed by retrieved sources."
            )

        # 5. Simple short answers
        simple_modes = {"economy": ("2-mark", "5-mark", "conversational"),
                        "balanced": ("2-mark",), "quality": ()}[self.cost_preference]
        if (analysis.intent == "definition" and analysis.complexity == 1
                and analysis.answer_mode in simple_modes and not analysis.ambiguous):
            reasons.append(f"simple {analysis.answer_mode} definition -> direct LLM (saves retrieval tokens)")
            return "direct_llm", False, reasons, None

        # 6. Default for in-domain academic questions
        reasons.append("in-domain academic question -> RAG")
        return "rag_llm", True, reasons, None
