"""
Claim extraction and evidence mapping (report §14.2 steps 1-4).

Each answer sentence that makes a statement becomes a claim. A claim is
compared with every evidence window (1-2 consecutive sentences of the
retrieved passages) using:
  * semantic similarity  - cosine of MiniLM embeddings
  * lexical coverage     - share of the claim's content words found in the window
  * number consistency   - every number in the claim must appear in an evidence
                           window that is relevant to the claim
  * NLI (optional)       - an entailment model judges the top windows
  * citation check       - a cited source [n] must itself support the claim

These are heuristic support signals, not entailment. Thresholds below were set
by inspection and should be calibrated against human judgements (the
evaluation harness exports claims for labelling). An NLI model can be plugged
in via `entailment_fn` for stronger checking.
"""

import re
from dataclasses import dataclass, field, asdict

import numpy as np

SUPPORTED_THRESHOLD = 0.58
PARTIAL_THRESHOLD = 0.45

_CITATION = re.compile(r"\[(\d{1,2})\]")
# A figure: 42, 3.14, 1,500 (thousands groups only). "(1,1)" is two numbers,
# not 11; dotted strings such as IP addresses (192.168.1.1) are not figures.
_NUMBER = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w]|\.\d)")
_DIMENSION = re.compile(r"(\d)\s*[x×]\s*(?=\d)")   # "32x32", "5 × 5" -> "32 x 32"
_WORD = re.compile(r"[a-z][a-z0-9\-]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")

_STOP = set("""
the and for with that this these those from into onto over under about which while where when
what whom whose their there they them then than also such each other some most more many much
very can could would should will shall may might must been being have has had does did done
are was were is be as at by on in of to or an a it its not but if so we you your our use used
using called known one two three first second example examples e.g i.e etc
""".split())

NON_CLAIM_PATTERNS = re.compile(
    r"^(in (summary|conclusion|short)|to summari[sz]e|overall|note:|for example:?$)|"
    r"knowledge base does not contain|i (am not|'m not) sure|let me|here is|here's",
    re.IGNORECASE,
)


# Not checkable statements: formula lines, continuations of a previous
# sentence ("where y is ..."), maths-heavy steps and echoed prompt rules. In
# the 10 Nov Colab run these made up 11 of 49 NLI "contradictions".
_FRAGMENT_START = re.compile(r"^\s*(\\\[|\$\$|\\\(|where\b|[a-z])")
_MATH_TOKEN = re.compile(r"\\(frac|left|right|times|approx|lambda|cdot|sum|int)\b|[=^_]")
# A calculation step ("H = ... = 0.940"): arithmetic is not checked against
# sources (a documented limitation), and its derived figures are never in them.
_EQUATION = re.compile(r"[\d)\]]\s*=|=\s*[-(\[]?\s*\d")
_RULE_ECHO = re.compile(r"^\s*(do not|never invent|example rule|important:|answer format:)", re.IGNORECASE)


def is_fragment(text):
    return bool(_FRAGMENT_START.match(text) or _RULE_ECHO.match(text)
                or len(_MATH_TOKEN.findall(text)) >= 3 or _EQUATION.search(text))


def split_sentences(text):
    sentences = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^(#{1,6}\s+|[-*•]\s+|\d+[.)]\s+)", "", line)
        line = line.replace("**", "").replace("__", "")
        for part in _SENTENCE_END.split(line):
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


def strip_citations(text):
    return re.sub(r"\s*\[\d{1,2}\]", "", text).strip()


def content_words(text):
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def numbers_in(text):
    text = _DIMENSION.sub(r"\1 x ", text or "")
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


@dataclass
class Claim:
    text: str
    cited_sources: list
    label: str = "unchecked"           # supported | partial | unsupported
    support_score: float = 0.0
    best_source: int | None = None     # 1-based source number
    best_evidence: str = ""
    semantic_similarity: float = 0.0
    lexical_coverage: float = 0.0
    numbers_ok: bool = True
    missing_numbers: list = field(default_factory=list)
    nli: str | None = None   # entailment | neutral | contradiction | possible_contradiction | None
    citation_ok: bool | None = None    # None when nothing was cited
    issues: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def extract_claims(answer):
    claims = []
    for sentence in split_sentences(answer):
        plain = strip_citations(sentence)
        if len(plain.split()) < 5 or plain.endswith("?") or plain.endswith(":"):
            continue
        if NON_CLAIM_PATTERNS.search(plain) or is_fragment(plain):
            continue
        cited = sorted({int(n) for n in _CITATION.findall(sentence)})
        claims.append(Claim(text=plain, cited_sources=cited))
    return claims


def evidence_windows(passages):
    """Return [(source_number, window_text)] with 1- and 2-sentence windows."""
    windows = []
    for number, passage in enumerate(passages, start=1):
        sentences = split_sentences(passage["text"])
        for i, sentence in enumerate(sentences):
            windows.append((number, sentence))
            if i + 1 < len(sentences):
                windows.append((number, sentence + " " + sentences[i + 1]))
    return windows


class EvidenceMapper:

    MEMO_SIZE = 2000  # passages kept in memory (~10 windows x 384 floats each)

    def __init__(self, embedding_model, entailment_fn=None, evidence_cache=None):
        self.model = embedding_model
        self.entailment_fn = entailment_fn
        self.evidence_cache = evidence_cache
        self._memo = {}  # chunk_id -> window vectors (insertion-ordered for eviction)

    def _passage_vectors(self, passage):
        """Window vectors for one passage: precomputed cache, memo, or embed now."""
        chunk_id = passage.get("chunk_id")
        expected = len(evidence_windows([passage]))
        if chunk_id and self.evidence_cache is not None:
            vectors = self.evidence_cache.vectors_for(chunk_id)
            if vectors is not None and len(vectors) == expected:
                return vectors
        if chunk_id and chunk_id in self._memo:
            return self._memo[chunk_id]
        vectors = self._embed([w for _, w in evidence_windows([passage])]) if expected else \
            np.zeros((0, 384), dtype=np.float32)
        if chunk_id:
            if len(self._memo) >= self.MEMO_SIZE:
                self._memo.pop(next(iter(self._memo)))
            self._memo[chunk_id] = vectors
        return vectors

    def _window_vectors(self, passages, windows):
        """Per-passage reuse so a partly cached set of passages is still cheap."""
        vectors = np.vstack([self._passage_vectors(p) for p in passages])
        if len(vectors) != len(windows):  # defensive: fall back to a fresh encode
            return self._embed([w for _, w in windows])
        return vectors

    def _embed(self, texts):
        return np.asarray(self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True,
            batch_size=64, show_progress_bar=False,
        ), dtype=np.float32)

    def map_claims(self, claims, passages, known_numbers=None):
        """`known_numbers`: figures the student gave in the question (e.g. "9
        positive and 5 negative examples"); they need no source."""
        known_numbers = set(known_numbers or ())
        if not claims:
            return claims
        windows = evidence_windows(passages)
        if not windows:
            for claim in claims:
                claim.label = "unsupported"
                claim.issues.append("no evidence available")
            return claims

        claim_vectors = self._embed([c.text for c in claims])
        window_vectors = self._window_vectors(passages, windows)
        similarity = claim_vectors @ window_vectors.T

        window_words = [content_words(w) for _, w in windows]
        window_numbers = [numbers_in(w) for _, w in windows]
        source_count = len(passages)
        nli_jobs = []

        for i, claim in enumerate(claims):
            words = content_words(claim.text)
            coverage = np.array([
                len(words & ww) / len(words) if words else 0.0 for ww in window_words
            ])
            combined = 0.65 * similarity[i] + 0.35 * coverage
            best = int(np.argmax(combined))

            claim.support_score = round(float(combined[best]), 3)
            claim.semantic_similarity = round(float(similarity[i][best]), 3)
            claim.lexical_coverage = round(float(coverage[best]), 3)
            claim.best_source = windows[best][0]
            claim.best_evidence = windows[best][1]

            # A figure must come from a passage about the same thing: "1,500 RPM"
            # in a sentence about a different motor does not support it.
            relevant = [j for j in range(len(windows)) if combined[j] >= PARTIAL_THRESHOLD] or [best]
            relevant_numbers = set().union(*(window_numbers[j] for j in relevant))
            missing_numbers = numbers_in(claim.text) - relevant_numbers - known_numbers
            claim.numbers_ok = not missing_numbers
            claim.missing_numbers = sorted(missing_numbers)
            if missing_numbers:
                claim.issues.append(f"numbers not in evidence: {sorted(missing_numbers)}")

            if claim.cited_sources:
                invalid = [n for n in claim.cited_sources if n < 1 or n > source_count]
                if invalid:
                    claim.issues.append(f"cites non-existent source(s) {invalid}")
                valid = [n for n in claim.cited_sources if 1 <= n <= source_count]
                cited_best = max(
                    (combined[j] for j, (src, _) in enumerate(windows) if src in valid),
                    default=0.0,
                )
                claim.citation_ok = bool(valid) and cited_best >= PARTIAL_THRESHOLD
                if valid and not claim.citation_ok:
                    claim.issues.append("cited source does not support the claim")

            if self.entailment_fn is not None:
                top = [int(j) for j in np.argsort(combined)[::-1][:3]]
                nli_jobs.append((claim, top))

        if nli_jobs:
            self._apply_nli(nli_jobs, windows)

        for claim in claims:
            score = claim.support_score
            # A figure that is in no relevant source is unsupported however
            # similar the wording is (Colab run, 8 Nov 2026: "partial" hid a
            # wrong worked example from the unsupported claim rate). So is a
            # claim the NLI model finds contradicted by a source.
            if not claim.numbers_ok or claim.nli == "contradiction":
                claim.label = "unsupported"
            elif claim.nli == "entailment" or score >= SUPPORTED_THRESHOLD:
                claim.label = "supported"
            elif score >= PARTIAL_THRESHOLD:
                claim.label = "partial"
            else:
                claim.label = "unsupported"
        return claims

    def _apply_nli(self, jobs, windows):
        """Entailed by any of the claim's top windows -> entailment. Contradiction
        needs the best-matching window, or 2 of the top 3, to contradict: a lower
        ranked window is often about a different case (on the induction-motor
        test, "no torque when the rotor is stopped" in a single-phase passage
        "contradicted" a correct sentence about torque). Otherwise neutral.

        A contradiction only counts for claims that state a figure. On the
        10 Nov Colab run, nli-deberta-v3-xsmall flagged 35 in-domain claims of
        which about 3 were wrong; restricted to claims with figures (and
        probability >= 0.95, set in NLIChecker) it kept the wrong-figure case
        with 1 false alarm. Other contradictions are recorded as
        "possible_contradiction": shown to developers, not counted."""
        pairs = [(windows[j][1], claim.text) for claim, top in jobs for j in top]
        judge = getattr(self.entailment_fn, "judge", None)
        if judge is not None:
            verdicts = judge(pairs)
        else:  # plain callable; bool results are accepted for backward compatibility
            verdicts = []
            for premise, hypothesis in pairs:
                v = self.entailment_fn(premise, hypothesis)
                verdicts.append(v if isinstance(v, str) else ("entailment" if v else "contradiction"))
        position = 0
        for claim, top in jobs:
            mine = verdicts[position:position + len(top)]
            position += len(top)
            if "entailment" in mine:
                claim.nli = "entailment"
            elif mine[:1] == ["contradiction"] or mine.count("contradiction") >= 2:
                source = windows[top[mine.index("contradiction")]][0]
                if numbers_in(claim.text):
                    claim.nli = "contradiction"
                    claim.issues.append(f"contradicted by source [{source}]")
                else:
                    claim.nli = "possible_contradiction"
                    claim.issues.append(f"possible contradiction with source [{source}] (not counted)")
            else:
                claim.nli = "neutral"
