"""
EduRoute AI orchestration (report §7): input processing -> query
understanding -> adaptive router -> selected workflow -> verification ->
formatted answer, with telemetry for evaluation (§22.2).

Typical use:
    pipeline = EduRoutePipeline()
    response = pipeline.ask("Explain backpropagation", answer_mode="5-mark")
    print(response.answer, response.sources, response.telemetry)
"""

import hashlib
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from config import settings
from src.llm.backends import ExtractiveBackend, LLMUnavailableError, build_backend
from src.llm.diagrams import (build_diagram_messages, extract_dot, heuristic_dot, is_connected,
                              wants_diagram)
from src.llm.prompts import (
    ANSWER_MODES, build_direct_messages, build_image_messages, build_rag_messages,
    build_revision_messages,
    strip_prompt_echo,
)
from src.agent import memory
from src.agent.agentic_rag import AgenticRetriever
from src.inference.answer_cache import SemanticAnswerCache
from src.rag.context import ContextBuilder
from src.rag.retrieval import EduRouteRetriever
from src.rag.session_document import refers_to_document, search_document

# MiniLM cosine above which an uploaded document's passage is relevant.
DOCUMENT_THRESHOLD = 0.35
from src.router.query_analysis import depends_on_image  # noqa: F401  (re-exported)
from src.router.router import AdaptiveRouter
from src.utils.io import append_jsonl
from src.verification.evidence_cache import EvidenceCache
from src.verification.arithmetic import check_answer as check_arithmetic
from src.verification.nli import build_nli_checker
from src.verification.verifier import AnswerVerifier, verification_triggers
from src.vision.analyzer import VisionAnalyzer
from src.vision.image_input import ImageValidationError, prepare_image
from src.vision.reasoning_tools import check_image_reading
from src.voice.speech import to_speakable

# A question that points at the image cannot be answered without it: answering
# it from the knowledge base alone produced a confident wrong answer in testing
# ("XOR gate" for an AND truth table). Detection lives in query_analysis.
IMAGE_NOT_READ = (
    "I could not analyse the image, and this question depends on what the image shows, "
    "so I have not guessed an answer. Please try again, or type out the relevant part "
    "of the image (for example the table values or the circuit components)."
)


INSUFFICIENT_EVIDENCE = re.compile(r"does not contain enough information", re.IGNORECASE)
CITATION = re.compile(r"\[(\d{1,2})\]")


@dataclass
class Telemetry:
    stages_ms: dict = field(default_factory=dict)
    total_ms: float = 0.0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    backends: list = field(default_factory=list)
    agent_trace: list = field(default_factory=list)     # agentic RAG steps (developer view)
    context: dict = field(default_factory=dict)         # context-engineering statistics
    memory: dict = field(default_factory=dict)          # conversation-memory statistics
    cache_hit: bool = False

    def add_generation(self, result):
        self.model_calls += 1 if result.is_model_output else 0
        self.input_tokens += result.input_tokens
        self.output_tokens += result.output_tokens
        tag = f"{result.backend}:{result.model}"
        if tag not in self.backends:
            self.backends.append(tag)


@dataclass
class EduRouteResponse:
    question: str
    answer: str
    answer_mode: str
    workflow: str
    strategy: str
    grounded: bool
    is_model_output: bool
    sources: list = field(default_factory=list)
    cited_sources: list = field(default_factory=list)
    invalid_citations: list = field(default_factory=list)
    visual_description: str | None = None
    diagram_dot: str | None = None      # Graphviz flowchart of the answer, when one was drawn
    arithmetic_issues: list = field(default_factory=list)   # calculations that do not work out
    verification: dict | None = None
    decision: dict | None = None
    notices: list = field(default_factory=list)
    spoken_text: str = ""
    telemetry: dict = field(default_factory=dict)
    error: str | None = None
    pending_check: bool = False          # a draft: response.complete() runs the checks

    def to_dict(self):
        return asdict(self)


def scoped_retrieve(retriever, question, filters, fallback_filters, threshold,
                    top_k=None, **retrieve_kwargs):
    """Retrieve within the selected scope, widening it if nothing there matches.

    Catalogue subjects overlap (the fractional distillation articles are filed
    under Mass Transfer and Petroleum Refining, not Petrochemical Technology). Without widening, a
    strict subject filter makes an in-KB question look out of domain, and it
    would be answered ungrounded. Scopes are tried in order; the first whose
    top score reaches the calibrated in-domain threshold wins, else the best.
    Returns (results, scope_used).
    """
    scopes = [filters] + list(fallback_filters or [])
    best, best_scope, best_top = [], filters, -1.0
    for scope in scopes:
        results = retriever.retrieve(question, top_k=top_k or settings.TOP_K, filters=scope,
                                     **retrieve_kwargs)
        top = (results[0]["similarity_score"] or 0.0) if results else 0.0
        if results and top >= threshold:
            return results, scope
        if top > best_top:
            best, best_scope, best_top = results, scope, top
    return best, best_scope


def _loggable(telemetry):
    """Telemetry for the log file. Agent queries and resolved follow-ups contain the
    student's words, so they are dropped unless question logging is enabled (§21)."""
    record = asdict(telemetry)
    if not settings.LOG_QUERY_TEXT:
        record["agent_trace"] = [{k: s.get(k) for k in ("action", "found", "top_score", "coverage")}
                                 for s in record.get("agent_trace", [])]
        record["memory"] = {k: v for k, v in record.get("memory", {}).items() if k != "resolved_query"}
    return record


class _Timer:
    def __init__(self, telemetry, stage):
        self.telemetry, self.stage = telemetry, stage

    def __enter__(self):
        self.started = time.perf_counter()

    def __exit__(self, *exc):
        elapsed = (time.perf_counter() - self.started) * 1000
        self.telemetry.stages_ms[self.stage] = round(
            self.telemetry.stages_ms.get(self.stage, 0) + elapsed, 1
        )


class EduRoutePipeline:

    def __init__(self, strategy="adaptive", cost_preference="balanced", llm_backend=None,
                 vision_backend=None, retriever=None, rag_version=None, retrieval_mode=None,
                 log=True, answer_cache=None):
        """`answer_cache`: a SemanticAnswerCache shared between pipelines/sessions;
        default: a private one when EDUROUTE_ANSWER_CACHE is on."""
        self.retriever = retriever or EduRouteRetriever(
            version=rag_version or settings.RAG_VERSION,
            mode=retrieval_mode or settings.RETRIEVAL_MODE,
        )
        self.llm = llm_backend if llm_backend is not None else build_backend(settings, "text")
        self.vision_backend = (
            vision_backend if vision_backend is not None else build_backend(settings, "vision")
        )
        self.extractive = ExtractiveBackend(sentence_scorer=self._score_sentences)
        self.verifier = AnswerVerifier(
            self.retriever.model,
            entailment_fn=build_nli_checker(settings.NLI_MODEL),
            evidence_cache=EvidenceCache.load_if_available(self.retriever.version),
        )
        self.router = AdaptiveRouter(
            retriever=self.retriever, strategy=strategy, cost_preference=cost_preference,
            vision_available=self.vision_backend is not None, log_decisions=log,
        )
        self.log = log
        self.telemetry_path = settings.LOG_DIR / "telemetry.jsonl"
        self.context_builder = ContextBuilder(self.retriever.model)
        self.answer_cache = answer_cache if answer_cache is not None else (
            SemanticAnswerCache(self.retriever.model) if settings.ANSWER_CACHE else None)

    def _agent(self, filters, fallback_filters):
        threshold = self.router.calibration["in_domain_threshold"]

        def search(query, k):
            return scoped_retrieve(self.retriever, query, filters, fallback_filters, threshold, top_k=k)[0]
        return AgenticRetriever(search, threshold, llm=self.llm)

    def _prepare_context(self, question, passages, answer_mode, telemetry):
        """Context engineering: de-duplicate, compress and budget the sources for the prompt."""
        if not settings.CONTEXT_ENGINEERING or not passages:
            return passages
        with _Timer(telemetry, "context"):
            selected, stats = self.context_builder.build(question, passages, answer_mode)
        telemetry.context = stats.to_dict()
        return selected or passages

    # ------------------------------------------------------------ helpers

    def _score_sentences(self, question, sentences):
        vectors = self.retriever.model.encode(
            [question] + sentences, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        return (vectors[1:] @ vectors[0]).tolist()

    def _generate(self, messages, telemetry, answer_mode, passages=None, question="",
                  notices=None, max_new_tokens=None, on_token=None):
        """Call the LLM; on failure fall back to extractive evidence (never silently)."""
        max_new = max_new_tokens or ANSWER_MODES[answer_mode]["max_new_tokens"]
        evidence = {"question": question, "passages": passages or []}
        try:
            result = self.llm.generate(messages, max_new_tokens=max_new,
                                       temperature=settings.TEMPERATURE, evidence=evidence,
                                       on_token=on_token)
            if result.extra.get("first_token_s") is not None:
                telemetry.stages_ms["first_token"] = round(result.extra["first_token_s"] * 1000, 1)
        except LLMUnavailableError as error:
            if notices is not None:
                notices.append(
                    f"Language model unavailable ({error}). Showing the most relevant "
                    "evidence excerpts instead of a generated answer."
                )
            result = self.extractive.generate(messages, max_new_tokens=max_new, evidence=evidence)
        telemetry.add_generation(result)
        return result

    def _retrieve(self, question, filters, telemetry, top_k=None):
        with _Timer(telemetry, "retrieval"):
            return self.retriever.retrieve(question, top_k=top_k or settings.TOP_K, filters=filters)

    def _scoped_retrieve(self, question, filters, fallback_filters):
        return scoped_retrieve(self.retriever, question, filters, fallback_filters,
                               self.router.calibration["in_domain_threshold"])

    @staticmethod
    def _default_fallbacks(filters):
        return [None] if filters else []

    # --------------------------------------------------------------- main

    def ask(self, question, image_bytes=None, answer_mode="5-mark", history=None,
            filters=None, subject_context=None, from_voice=False, fallback_filters=None,
            on_token=None, document=None, defer_checks=False):
        """`fallback_filters`: wider scopes tried in order when `filters` has no
        in-domain match (default: the whole index). Pass [] to keep the filter strict."""
        if answer_mode not in ANSWER_MODES:
            raise ValueError(f"answer_mode must be one of {list(ANSWER_MODES)}")
        question = (question or "").strip()
        started = time.perf_counter()
        telemetry = Telemetry()
        notices = []

        # 1. Input processing
        prepared = None
        if image_bytes:
            with _Timer(telemetry, "image_preprocessing"):
                try:
                    prepared = prepare_image(image_bytes)
                except ImageValidationError as error:
                    notices.append(f"Image rejected: {error}")
        if not question and prepared is None:
            raise ValueError("Please type a question or attach an image.")

        # 1b. Memory: a follow-up ("what are its advantages?") is searched as a
        # self-contained question; the prompt gets a budgeted memory, not raw turns.
        retrieval_query, follow_up_topic = memory.contextualize(question, history)
        prompt_history = history
        if settings.CONVERSATION_MEMORY and history:
            prompt_history = memory.render(history)
            telemetry.memory = {**memory.memory_stats(history, prompt_history),
                                "follow_up": follow_up_topic is not None}
        if follow_up_topic:
            telemetry.memory["resolved_query"] = retrieval_query

        # 1c. Answer cache: the same question, same scope and format, answered recently.
        cache_signature = cache_vector = None
        if (self.answer_cache is not None and question and prepared is None and document is None
                and follow_up_topic is None):
            cache_signature = SemanticAnswerCache.signature(answer_mode, filters, self.router.strategy)
            cache_vector = self.retriever.embed_query(question)[0]
            cached = self.answer_cache.get(question, cache_signature, cache_vector)
            if cached is not None:
                return self._from_cache(cached, telemetry, started, question)

        # 2-4. Query understanding + routing (the adaptive probe is reused below)
        with _Timer(telemetry, "routing"):
            probe = None
            if fallback_filters is None:
                fallback_filters = self._default_fallbacks(filters)
            adaptive = self.router.strategy == "adaptive"
            # Baselines never probe; retrieving baselines still get the same scope widening.
            if question and (adaptive or (filters and self.router.strategy != "direct")):
                probe, scope = self._scoped_retrieve(retrieval_query, filters, fallback_filters)
                if scope != filters:
                    narrow = "subject" if "subject_id" in filters else "branch"
                    wider = "the selected branch" if scope and "discipline" in scope else "the whole knowledge base"
                    notices.append(f"Nothing in the selected {narrow} matched closely, so sources "
                                   f"from {wider} were used.")
                    filters = scope
            decision = self.router.route(
                question, has_image=prepared is not None, from_voice=from_voice,
                answer_mode=answer_mode, history=history, filters=filters,
                probe_results=probe if adaptive else None, follow_up=follow_up_topic is not None,
            )
        passages, visual_description, result = [], None, None
        workflow = decision.workflow

        # An uploaded document is searched first; when it is relevant it grounds
        # the answer (with at most two strong knowledge-base passages beside it).
        if document is not None and question and workflow in ("direct_llm", "rag_llm") \
                and not (decision.analysis or {}).get("general_chat"):
            with _Timer(telemetry, "document_search"):
                doc_passages = search_document(document, question, self.retriever.model)
            top = (doc_passages[0].get("similarity_score") or 0.0) if doc_passages else 0.0
            if doc_passages and (refers_to_document(question) or top >= DOCUMENT_THRESHOLD):
                threshold = self.router.calibration["in_domain_threshold"]
                kb = [] if refers_to_document(question) else [
                    p for p in (probe or []) if (p.get("similarity_score") or 0.0) >= threshold][:2]
                probe = doc_passages + kb
                if workflow == "direct_llm":
                    decision.reasons.append("uploaded document covers the question -> RAG on the document")
                    decision.fallback = None          # the answer is grounded after all
                    decision.verify_eligible = True
                workflow = "rag_llm"
                notices.append(f"Answered from your document '{document.name}'"
                               + (" and the knowledge base." if kb else "."))
        if decision.fallback:
            notices.append(decision.fallback)
        if prepared is not None and self.vision_backend is None and depends_on_image(question):
            return self._finish(question, IMAGE_NOT_READ, answer_mode, decision, "none", telemetry,
                                notices, started, error="no vision model configured",
                                is_model_output=False)

        # 5-6. Execute the selected workflow
        if workflow in ("vision_llm", "hybrid_multimodal_rag"):
            analyzer = VisionAnalyzer(self.vision_backend)
            try:
                if workflow == "vision_llm" and settings.VISION_TWO_STEP and self.llm.is_available():
                    # Read, then reason: the vision model transcribes, the text model answers.
                    with _Timer(telemetry, "vision"):
                        description = analyzer.describe(prepared, question)
                    telemetry.add_generation(description)
                    visual_description = description.text + check_image_reading(description.text)
                    messages = build_image_messages(question, visual_description, answer_mode,
                                                    prompt_history, subject_context)
                    with _Timer(telemetry, "generation"):
                        result = self._generate(messages, telemetry, answer_mode, None,
                                                question, notices, on_token=on_token)
                elif workflow == "vision_llm":
                    with _Timer(telemetry, "vision"):
                        result = analyzer.answer(prepared, question, answer_mode, on_token=on_token)
                    telemetry.add_generation(result)
                else:
                    with _Timer(telemetry, "vision"):
                        description = analyzer.describe(prepared, question)
                    telemetry.add_generation(description)
                    visual_description = description.text + check_image_reading(description.text)
                    query = f"{question} {visual_description[:400]}".strip()
                    passages = self._retrieve(query, filters, telemetry)
                    passages = self._prepare_context(question or query, passages, answer_mode, telemetry)
                    messages = build_rag_messages(
                        question or "Explain this image.", passages, answer_mode, prompt_history,
                        subject_context, visual_description=visual_description,
                        stable_prefix=settings.CONTEXT_ENGINEERING,
                    )
                    with _Timer(telemetry, "generation"):
                        result = self._generate(messages, telemetry, answer_mode, passages,
                                                question, notices, on_token=on_token)
            except LLMUnavailableError as error:
                notices.append(f"Vision model unavailable: {error}")
                if not question or depends_on_image(question):
                    message = (IMAGE_NOT_READ if question else
                               "The image could not be analysed and no text question was given.")
                    return self._finish(question, message, answer_mode, decision, "none",
                                        telemetry, notices, started, error=str(error),
                                        is_model_output=False)
                notices.append("Answering the text question from the knowledge base, without the image.")
                workflow = "rag_llm"

        agent = None
        if workflow == "rag_llm":
            document_grounded = document is not None and probe and probe[0].get("page") is not None
            if settings.AGENTIC_RAG and question and not document_grounded:
                # Agentic RAG: plan sub-queries, judge coverage, refine (bounded).
                agent = self._agent(filters, fallback_filters)
                subject = (subject_context or {}).get("Subject") or (subject_context or {}).get("Branch")
                with _Timer(telemetry, "retrieval"):
                    run = agent.run(retrieval_query, probe=probe, subject=subject)
                passages = run.passages
                telemetry.agent_trace = run.trace_dicts()
            else:
                passages = probe if probe else self._retrieve(retrieval_query, filters, telemetry)
            if not passages:
                notices.append("No passages matched the selected filters; answering without sources.")
                workflow = "direct_llm"
            else:
                passages = self._prepare_context(question, passages, answer_mode, telemetry)
                messages = build_rag_messages(question, passages, answer_mode, prompt_history,
                                              subject_context, stable_prefix=settings.CONTEXT_ENGINEERING)
                with _Timer(telemetry, "generation"):
                    result = self._generate(messages, telemetry, answer_mode, passages,
                                            question, notices, on_token=on_token)

        if workflow == "direct_llm":
            messages = build_direct_messages(question, answer_mode, prompt_history, subject_context)
            fallback_passages = None
            signal = decision.retrieval_signal or {}
            if question and not self.llm.is_available() and signal.get("in_domain", True):
                # No model to answer directly: show evidence if the KB covers the topic.
                fallback_passages = probe or self._retrieve(question, filters, telemetry)
            with _Timer(telemetry, "generation"):
                result = self._generate(messages, telemetry, answer_mode, fallback_passages,
                                        question, notices, on_token=on_token)
            if not result.is_model_output and fallback_passages:
                passages = fallback_passages

        answer = strip_prompt_echo(result.text)
        draft_notices = list(notices)
        telemetry.stages_ms["draft_ready"] = round((time.perf_counter() - started) * 1000, 1)

        def complete(answer=answer, passages=passages):
            """Steps 7-8 (verification, arithmetic, diagram), then log and cache.
            Run at once, or later by the caller when `defer_checks` is set."""

            # 7. Selective verification (+ at most one revision call)
            # Figures the student gave, or that the vision model read off the image,
            # need no source (e.g. "32x32 image", "20 ms" on a graph axis).
            known_text = f"{question} {visual_description or ''}"
            verification = None
            if passages and decision.verify_eligible:
                triggers = list(decision.verify_triggers)
                if self.router.strategy == "adaptive" and not decision.verify:
                    triggers = verification_triggers(None, None, answer=answer)
                if triggers or self.router.strategy == "always_verify":
                    with _Timer(telemetry, "verification"):
                        report = self.verifier.verify(answer, passages, triggers, question=known_text)
                    if self.verifier.needs_revision(report) and result.is_model_output:
                        to_fix = self.verifier.claims_to_fix(report)
                        before = report.unsupported_claim_rate
                        if agent is not None:
                            # Agentic correction: fetch evidence for exactly the unsupported
                            # claims, so the revision can correct them, not only drop them.
                            with _Timer(telemetry, "retrieval"):
                                extra, steps = agent.correct(to_fix, passages)
                            telemetry.agent_trace += [s.to_dict() for s in steps]
                            if extra:
                                passages = passages + extra      # appended: earlier [n] keep their numbers
                        messages = build_revision_messages(question, answer, passages, to_fix, answer_mode)
                        with _Timer(telemetry, "revision"):
                            revised = self._generate(messages, telemetry, answer_mode, passages,
                                                     question, notices)
                        if revised.is_model_output and revised.text:
                            answer = strip_prompt_echo(revised.text)
                            with _Timer(telemetry, "verification"):
                                report = self.verifier.verify(answer, passages, triggers, question=known_text)
                            report.revised = True
                            report.note = f"Unsupported claim rate before revision: {before}"
                    unchecked = sorted({n for c in report.claims for n in c.missing_numbers},
                                       key=lambda n: float(n) if n.replace(".", "", 1).isdigit() else 0)
                    if unchecked:
                        notices.append("These figures could not be found in the sources, so check them "
                                       "before relying on them: " + ", ".join(unchecked))
                    contradicted = [c.text for c in report.claims if c.nli == "contradiction"]
                    if contradicted:
                        notices.append("The sources contradict this part of the answer, so treat it "
                                       "as likely wrong: " + " | ".join(t[:160] for t in contradicted))
                    verification = report.to_dict()

            if not result.is_model_output:
                notices.append("Extractive mode: these are source excerpts, not a generated explanation.")
            if passages and INSUFFICIENT_EVIDENCE.search(answer):
                notices.append("The knowledge base did not contain enough information for a full answer.")

            # 7b. Arithmetic: recompute what the answer computes (every workflow; no model needed).
            # One revision call if a calculation is wrong and none has been made yet.
            with _Timer(telemetry, "arithmetic"):
                arithmetic = check_arithmetic(answer)
            already_revised = bool(verification and verification.get("revised"))
            if arithmetic and result.is_model_output and not already_revised:
                fixes = [f"This calculation is wrong: {i.describe()}. Recompute it step by step."
                         for i in arithmetic[:3]]
                messages = build_revision_messages(question, answer, passages, fixes, answer_mode)
                with _Timer(telemetry, "revision"):
                    revised = self._generate(messages, telemetry, answer_mode, passages, question, notices)
                if revised.is_model_output and revised.text:
                    answer = strip_prompt_echo(revised.text)
                    arithmetic = check_arithmetic(answer)
                    if verification is not None:
                        verification["arithmetic_revised"] = True
            if arithmetic:
                notices.append("A calculation in this answer does not work out, so check it: "
                               + "; ".join(i.describe() for i in arithmetic[:3]))

            # 8. Workflow diagram of the final answer, when asked for or for 10-mark processes
            diagram = None
            intent = (decision.analysis or {}).get("intent")
            if question and answer and wants_diagram(question, answer_mode, intent):
                with _Timer(telemetry, "diagram"):
                    if result.is_model_output:
                        try:
                            drawn = self.llm.generate(build_diagram_messages(question, answer),
                                                      max_new_tokens=400, temperature=0.1)
                            telemetry.add_generation(drawn)
                            diagram = extract_dot(drawn.text)
                        except LLMUnavailableError:
                            diagram = None
                    # A disconnected model diagram (separate fragments) is replaced by the
                    # step-by-step flowchart of the answer, when the answer has steps.
                    if diagram and not is_connected(diagram):
                        diagram = heuristic_dot(answer, title=question) or diagram
                    diagram = diagram or heuristic_dot(answer, title=question)

            response = self._finish(question, answer, answer_mode, decision, workflow, telemetry,
                                    notices, started, passages=passages,
                                    visual_description=visual_description,
                                    verification=verification, is_model_output=result.is_model_output)
            response.diagram_dot = diagram
            response.arithmetic_issues = [i.describe() for i in arithmetic]
            if cache_signature is not None and SemanticAnswerCache.cacheable(response):
                self.answer_cache.put(question, cache_signature, response.to_dict(), cache_vector)
            return response

        if not defer_checks:
            return complete()
        # Answer first, check afterwards: the student reads the draft while the
        # checks (and any revision) run; nothing is logged or cached until then.
        draft = self._finish(question, answer, answer_mode, decision, workflow, telemetry,
                             draft_notices, started, passages=passages,
                             visual_description=visual_description,
                             is_model_output=result.is_model_output, log=False)
        draft.pending_check = True
        draft.complete = complete
        return draft

    def _from_cache(self, cached, telemetry, started, question):
        """A stored answer to the same question: no model call, no retrieval."""
        import copy
        response = EduRouteResponse(**copy.deepcopy(cached))
        response.question = question          # the student's own wording (the saved copy has none)
        telemetry.cache_hit = True
        telemetry.total_ms = round((time.perf_counter() - started) * 1000, 1)
        telemetry.stages_ms["cache"] = telemetry.total_ms
        response.telemetry = asdict(telemetry)
        response.notices = ["Answered instantly: this question was answered recently, "
                            "so this is the stored answer."]
        if self.log:
            append_jsonl(self.telemetry_path, {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "question_sha256": hashlib.sha256(response.question.encode("utf-8")).hexdigest()[:16],
                "question": response.question if settings.LOG_QUERY_TEXT else None,
                "workflow": response.workflow, "strategy": response.strategy,
                "answer_mode": response.answer_mode, "verified": response.verification is not None,
                "revised": bool(response.verification and response.verification.get("revised")),
                "unsupported_claim_rate": response.verification and response.verification.get(
                    "unsupported_claim_rate"),
                "sources": len(response.sources), "cited": len(response.cited_sources),
                **_loggable(telemetry),
            })
        return response

    def _finish(self, question, answer, answer_mode, decision, workflow, telemetry, notices,
                started, passages=None, visual_description=None, verification=None,
                is_model_output=True, error=None, log=True):
        passages = passages or []
        cited = sorted({int(n) for n in CITATION.findall(answer)})
        valid = [n for n in cited if 1 <= n <= len(passages)]
        invalid = [n for n in cited if n not in valid]
        sources = [
            {
                "number": i,
                "article_title": p["article_title"],
                "section": p.get("section"),
                "source_url": p.get("source_url"),
                "chunk_id": p.get("chunk_id"),
                "similarity_score": p.get("similarity_score"),
                "revision_id": p.get("revision_id"),
                "text": p["text"],
                "cited": i in valid,
            }
            for i, p in enumerate(passages, start=1)
        ]
        telemetry.total_ms = round((time.perf_counter() - started) * 1000, 1)

        response = EduRouteResponse(
            question=question, answer=answer, answer_mode=answer_mode, workflow=workflow,
            strategy=decision.strategy, grounded=bool(passages) and workflow != "direct_llm",
            is_model_output=is_model_output, sources=sources, cited_sources=valid,
            invalid_citations=invalid, visual_description=visual_description,
            verification=verification, decision=decision.to_dict(), notices=notices,
            spoken_text=to_speakable(answer), telemetry=asdict(telemetry), error=error,
        )
        if self.log and log:
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                # Store a hash, not the question, unless explicitly enabled (§21).
                "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest()[:16],
                "question": question if settings.LOG_QUERY_TEXT else None,
                "workflow": workflow,
                "strategy": decision.strategy,
                "answer_mode": answer_mode,
                "verified": verification is not None,
                "revised": bool(verification and verification.get("revised")),
                "unsupported_claim_rate": verification and verification.get("unsupported_claim_rate"),
                "sources": len(sources),
                "cited": len(valid),
                **_loggable(telemetry),
            }
            append_jsonl(self.telemetry_path, record)
        return response
