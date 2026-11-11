"""
Query understanding (report §7 layer 3, §8.1 routing signals).

Rule-based and transparent on purpose: every signal can be printed and
justified in the evaluation, and the rules are cheap enough to run on every
query. A learned classifier can later replace `classify_intent` using the
logged decisions as training data.
"""

import re
from dataclasses import dataclass, field, asdict

INTENT_PATTERNS = [
    ("diagram", r"\b(this|the|attached|uploaded|given) (image|diagram|figure|circuit|graph|plot|screenshot|picture|drawing|chart)\b|\bin the (image|diagram|figure|picture)\b"),
    ("numerical", r"\b(calculate|compute|find the value|solve|evaluate|how many|how much|derive|numerical)\b|\d+\s*[\+\-\*/^=]\s*\d+"),
    ("comparison", r"\b(compare|comparison|difference(s)? between|differentiate|distinguish|versus|vs\.?|contrast|advantages and disadvantages|pros and cons)\b"),
    ("procedure", r"\b(steps|step-by-step|procedure|algorithm for|how to|how do (i|you|we)|process of|implement|write (a|the) (program|code))\b"),
    ("code", r"\b(code|program|python|java|c\+\+|function|error|exception|traceback|syntax)\b"),
    ("definition", r"^\s*(what is|what are|define|definition of|meaning of|what do you mean by|what does .{1,40} mean)\b"),
    ("explanation", r"\b(explain|describe|why|how does|how do|discuss|elaborate|illustrate|working of|role of|importance of)\b"),
]

# Explicit requests for sources / course material -> evidence needed.
EVIDENCE_REQUEST = re.compile(
    r"\b(according to|source|sources|cite|citation|reference|references|syllabus|"
    r"notes|textbook|course material|from the (knowledge base|kb)|evidence|"
    r"who (invented|proposed|introduced)|when was|in which year|history of)\b",
    re.IGNORECASE,
)

# Small talk / study-skills / meta questions that need no academic evidence.
GENERAL_CHAT = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|good (morning|evening)|who are you|"
    r"what can you do|help)\b|\b(study tips|how should i study|exam preparation tips|"
    r"motivat\w*|time management)\b",
    re.IGNORECASE,
)

VAGUE_REFERENCES = re.compile(r"^\s*(it|this|that|they|those|these)\b|\b(the above|same thing|that one)\b", re.IGNORECASE)

SUBQUESTION_SPLIT = re.compile(r"\?|;|\b(and also|also explain|as well as|along with)\b", re.IGNORECASE)

# A question that points at the image ("this truth table", "the circuit
# shown", "solve the problem in the image") is answered from the image itself.
IMAGE_REFERENCE = re.compile(
    r"\b(this|these|the|above|following|given|attached|uploaded|my)\s+"
    r"(image|picture|photo|screenshot|figure|diagram|circuit|graph|plot|chart|table|truth table|"
    r"flow ?chart|code|snippet|equation|problem|drawing|sketch|waveform|signal|schematic)s?\b"
    r"|\b(shown|pictured|in the (image|picture|photo|figure))\b",
    re.IGNORECASE,
)


def depends_on_image(question):
    return bool(question and IMAGE_REFERENCE.search(question))


@dataclass
class QueryAnalysis:
    question: str
    modality: str                 # text | image | voice | image+voice
    intent: str
    complexity: int               # 1 (simple) .. 4 (multi-part, long)
    evidence_requested: bool
    general_chat: bool
    ambiguous: bool
    answer_mode: str
    sub_questions: int
    word_count: int
    image_dependent: bool = False    # the answer is in the attached image itself
    reasons: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def classify_intent(question, has_image):
    lowered = question.lower()
    for intent, pattern in INTENT_PATTERNS:
        if re.search(pattern, lowered):
            if intent == "diagram" and not has_image:
                continue
            return intent
    return "diagram" if has_image else "explanation"


def analyse_query(question, has_image=False, from_voice=False, answer_mode="5-mark",
                  history=None):
    question = (question or "").strip()
    words = question.split()
    reasons = []

    if has_image and from_voice:
        modality = "image+voice"
    elif has_image:
        modality = "image"
    elif from_voice:
        modality = "voice"
    else:
        modality = "text"

    intent = classify_intent(question, has_image)
    parts = [p for p in SUBQUESTION_SPLIT.split(question) if p and len(p.split()) >= 3]
    sub_questions = max(1, len(parts))

    complexity = 1
    if len(words) > 18:
        complexity += 1
    if sub_questions > 1:
        complexity += 1
    if intent in ("comparison", "numerical", "procedure") or answer_mode == "10-mark":
        complexity += 1
    complexity = min(complexity, 4)

    evidence_requested = bool(EVIDENCE_REQUEST.search(question))
    general_chat = bool(GENERAL_CHAT.search(question)) and len(words) <= 15

    # Short questions that name a topic ("Explain backpropagation") are not
    # ambiguous; only references to something the system cannot see are.
    ambiguous = False
    if VAGUE_REFERENCES.search(question) and not history and not has_image:
        ambiguous = True
        reasons.append("refers to something not in the conversation")

    reasons.append(f"intent={intent}")
    if evidence_requested:
        reasons.append("explicit request for sources/course material")
    if general_chat:
        reasons.append("general/small-talk question")

    return QueryAnalysis(
        question=question,
        modality=modality,
        intent=intent,
        complexity=complexity,
        evidence_requested=evidence_requested,
        general_chat=general_chat,
        ambiguous=ambiguous,
        answer_mode=answer_mode,
        sub_questions=sub_questions,
        word_count=len(words),
        image_dependent=has_image and depends_on_image(question),
        reasons=reasons,
    )
