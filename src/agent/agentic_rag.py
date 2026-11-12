"""
Agentic RAG: retrieval that plans, checks its own evidence and corrects itself.

Single-shot RAG embeds the whole question once. That fails in predictable ways:
- a comparison ("difference between the Rankine and Carnot cycles") lands
  between the two topics and retrieves passages about neither in depth;
- a multi-part question retrieves for its dominant part only;
- a question phrased unlike the sources misses them, and nothing notices;
- when verification finds unsupported claims, the revision is given the same
  passages that failed to support them, so it can only delete, not correct.

The agent runs a short, bounded loop over one tool (search) and one critic
(evidence coverage), keeping a trace of every step for the developer view:

  PLAN      split comparisons and multi-part questions into sub-queries
  RETRIEVE  search each sub-query (scope widening as usual) and merge
  CRITIQUE  coverage = share of the question's key terms present in the evidence
  REFINE    if coverage is low: search again for the missing terms (with the
            subject as context); optionally ask the LLM for a better query
  CORRECT   after verification: search for evidence on each unsupported claim
            and hand it to the one permitted revision

Decisions are rules and measurements, not extra model calls, except the
optional LLM rewrite: on a CPU every model call costs tens of seconds.
"""

import math
import re
from dataclasses import dataclass, field

from config import settings
from src.utils.text import content_words

COMPARISON = [
    re.compile(r"(?:difference|differences|differentiate|distinguish|compare|comparison|contrast)s?\s+"
               r"(?:between\s+|of\s+)?(?P<a>.+?)\s+(?:and|vs\.?|versus|with|from)\s+(?P<b>.+?)\s*[?.]*$",
               re.IGNORECASE),
    re.compile(r"^\s*(?P<a>[^?]+?)\s+(?:vs\.?|versus)\s+(?P<b>[^?]+?)\s*[?.]*$", re.IGNORECASE),
]
PARTS = re.compile(r"\?\s+(?=\w)|;\s*|\b(?:and also|also explain|as well as|along with)\b", re.IGNORECASE)
LEADING = re.compile(r"^(?:the|a|an|between)\s+", re.IGNORECASE)


@dataclass
class AgentStep:
    action: str
    query: str
    found: int = 0
    top_score: float = 0.0
    coverage: float | None = None
    note: str = ""

    def to_dict(self):
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class AgentResult:
    passages: list
    trace: list = field(default_factory=list)
    coverage: float = 0.0
    sub_queries: list = field(default_factory=list)

    def trace_dicts(self):
        return [s.to_dict() for s in self.trace]


def _key(p):
    return p.get("chunk_id") or hash(p["text"][:200])


def merge(groups, limit):
    """Interleave ranked lists (one per sub-query), dropping duplicates, so every
    part of the question keeps its best passages."""
    seen, merged = set(), []
    for rank in range(max((len(g) for g in groups), default=0)):
        for group in groups:
            if rank < len(group) and _key(group[rank]) not in seen:
                seen.add(_key(group[rank]))
                merged.append(group[rank])
    return merged[:limit]


def coverage(question, passages):
    """Share of the question's key terms that appear somewhere in the evidence."""
    terms = set(content_words(question))
    if not terms:
        return 1.0
    evidence = set()
    for p in passages:
        evidence.update(content_words(p.get("text", "")))
        evidence.update(content_words(p.get("article_title", "")))
    return len(terms & evidence) / len(terms)


def plan(question):
    """Sub-queries for a comparison or a multi-part question; else the question itself."""
    for pattern in COMPARISON:
        m = pattern.search(question)
        if m:
            a, b = (LEADING.sub("", m.group(x)).strip(" ,") for x in ("a", "b"))
            if content_words(a) and content_words(b):
                return [a, b]
    parts = [p.strip(" ,?") for p in PARTS.split(question) if p and content_words(p)]
    return parts if len(parts) > 1 else [question]


class AgenticRetriever:

    def __init__(self, search, threshold, llm=None, max_steps=None, min_coverage=None):
        """`search(query, k)` -> ranked passages (the pipeline's scoped retrieval)."""
        self.search = search
        self.threshold = threshold
        self.llm = llm
        self.max_steps = max_steps or settings.AGENTIC_MAX_STEPS
        self.min_coverage = min_coverage or settings.AGENTIC_MIN_COVERAGE

    def _step(self, trace, action, query, results, question, evidence, note=""):
        top = (results[0].get("similarity_score") or 0.0) if results else 0.0
        trace.append(AgentStep(action, query, len(results), top, coverage(question, evidence), note))

    def run(self, question, top_k=None, probe=None, subject=None):
        top_k = top_k or settings.TOP_K
        trace = []
        sub_queries = plan(question)
        trace.append(AgentStep("plan", question, note=f"{len(sub_queries)} sub-quer"
                               f"{'y' if len(sub_queries) == 1 else 'ies'}: " + " | ".join(sub_queries)))

        # RETRIEVE: one search per sub-query (the router's probe is the whole-question search)
        if len(sub_queries) == 1:
            passages = list(probe) if probe else self.search(question, top_k)
            self._step(trace, "retrieve", question, passages, question, passages,
                       "reused the routing probe" if probe else "")
        else:
            per = max(2, math.ceil(top_k / len(sub_queries)))
            groups = []
            for sub in sub_queries:
                found = self.search(sub, per + 1)
                groups.append(found)
                self._step(trace, "retrieve", sub, found, question, found)
            if probe:
                groups.append(list(probe))
            passages = merge(groups, top_k + 1)

        # CRITIQUE -> REFINE, bounded
        score = coverage(question, passages)
        steps = 1
        while score < self.min_coverage and steps < self.max_steps:
            steps += 1
            evidence_words = {w for p in passages for w in content_words(p.get("text", ""))}
            missing = [w for w in dict.fromkeys(content_words(question)) if w not in evidence_words]
            rewrite = None
            if steps == self.max_steps and settings.AGENTIC_LLM_REWRITE and self.llm is not None:
                rewrite = self._llm_rewrite(question)
            query = rewrite or " ".join(missing + ([subject] if subject else [])) or question
            found = [p for p in self.search(query, top_k)
                     if (p.get("similarity_score") or 0.0) >= self.threshold * 0.9]
            before = len(passages)
            passages = merge([passages, found], top_k + 2)
            new_score = coverage(question, passages)
            self._step(trace, "refine", query, found, question, passages,
                       f"{'LLM rewrite; ' if rewrite else ''}missing terms: {', '.join(missing) or '-'}; "
                       f"+{len(passages) - before} passages")
            if new_score <= score:          # nothing gained: stop rather than loop
                break
            score = new_score
        trace.append(AgentStep("critique", question, len(passages), coverage=score,
                               note="enough evidence" if score >= self.min_coverage
                               else "coverage still low: the answer will say what is missing"))
        return AgentResult(passages, trace, score, sub_queries)

    def _llm_rewrite(self, question):
        try:
            result = self.llm.generate([{"role": "user", "content":
                "Write one search query of at most 10 words that would find a textbook passage "
                f"answering this question. Reply with the query only.\nQuestion: {question}"}],
                max_new_tokens=24, temperature=0.0)
            return result.text.strip().strip('"').splitlines()[0][:120] if result.text else None
        except Exception:
            return None

    def correct(self, claims, existing, per_claim=2, limit=3):
        """CORRECT: new evidence for unsupported claims (not already in `existing`)."""
        have = {_key(p) for p in existing}
        new, trace = [], []
        for claim in claims:
            found = [p for p in self.search(claim, per_claim)
                     if (p.get("similarity_score") or 0.0) >= self.threshold and _key(p) not in have]
            for p in found:
                if len(new) < limit:
                    have.add(_key(p))
                    new.append(p)
            trace.append(AgentStep("correct", claim[:120], len(found),
                                   (found[0].get("similarity_score") or 0.0) if found else 0.0,
                                   note=f"{len(found)} new passage(s) for an unsupported claim"))
        return new, trace
