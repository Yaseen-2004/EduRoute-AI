"""
Practice quizzes: multiple-choice questions written from the knowledge base,
each one checked against the sources before a student sees it.

1. Retrieve passages for the topic (the same scoped retrieval as answers) and
   fit them to a token budget (context engineering).
2. Ask the model for N questions as JSON: four options, the index of the
   correct one, a one-sentence explanation and the source number it rests on.
3. Validate every question: well-formed, four distinct options, a valid answer
   index and source number. A malformed reply loses only its bad questions.
4. Check grounding: the correct option's key words must appear in the cited
   source (or, failing that, in any source). A question whose answer is not in
   the sources is dropped, so a small model cannot quiz students on something
   it made up.

Quizzes are saved per topic (outputs/cache/quizzes.jsonl), so the next student
gets one at once; "new questions" makes a fresh set.
"""

import json
import random
import re
import time
from dataclasses import asdict, dataclass, field

from config import settings
from src.utils.text import content_words

QUIZ_PROMPT = (
    "Write {n} multiple-choice questions for an engineering student revising '{topic}'.\n"
    "Use ONLY facts stated in the SOURCES below. Each question tests one fact or idea; "
    "avoid trick questions and 'all of the above'.\n"
    "Reply with a JSON array only, no other text, in exactly this form:\n"
    '[{{"question": "...", "options": ["...", "...", "...", "..."], "answer": 0, '
    '"explanation": "one sentence saying why", "source": 1}}]\n'
    '"answer" is the index (0-3) of the correct option; "source" is the number of the '
    "source that states it. Wrong options must be plausible but clearly wrong according to "
    "the sources.\n\nSOURCES:\n{sources}"
)
MIN_SUPPORT = 0.5


@dataclass
class QuizQuestion:
    question: str
    options: list
    answer: int
    explanation: str
    source: int
    support: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class Quiz:
    topic: str
    subject_id: str | None
    questions: list = field(default_factory=list)
    sources: list = field(default_factory=list)       # [{number, article_title, source_url, text}]
    dropped: int = 0                                   # generated but rejected (malformed/unsupported)
    seconds: float = 0.0
    created: float = 0.0

    def to_dict(self):
        return {**asdict(self), "questions": [q if isinstance(q, dict) else q.to_dict() for q in self.questions]}


def _json_array(text):
    """The first JSON array in a model reply (tolerates prose or code fences around it)."""
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    chunk = text[start:end + 1]
    try:
        data = json.loads(chunk)
    except json.JSONDecodeError:
        # Common small-model slips: trailing commas, single quotes.
        cleaned = re.sub(r",\s*([\]}])", r"\1", chunk)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return []
    return data if isinstance(data, list) else []


def support(option, passage_text):
    """Share of the option's key words found in a passage."""
    words = set(content_words(option, min_len=2))
    if not words:
        return 0.0
    return len(words & set(content_words(passage_text, min_len=2))) / len(words)


def validate(raw, passages, reasons=None):
    """Well-formed, grounded questions; returns (kept, dropped_count).
    `reasons`: a list that receives why each dropped question was dropped."""
    kept, dropped, seen = [], 0, set()
    reasons = reasons if reasons is not None else []
    for item in raw:
        try:
            question = str(item["question"]).strip()
            options = [str(o).strip() for o in item["options"]]
            answer = int(item["answer"])
            source = int(item.get("source", 1))
            explanation = str(item.get("explanation", "")).strip()
        except (KeyError, TypeError, ValueError) as error:
            dropped += 1
            reasons.append(f"malformed: {error!r}")
            continue
        well_formed = (len(question.split()) >= 4 and len(options) == 4 and all(options)
                       and len({o.lower() for o in options}) == 4 and 0 <= answer < 4
                       and question.lower() not in seen)
        if not well_formed:
            dropped += 1
            reasons.append(f"not well formed: {question[:60]!r} options={len(options)} answer={answer}")
            continue
        if not 1 <= source <= len(passages):
            source = 1
        correct = options[answer]
        score = support(correct, passages[source - 1]["text"])
        if score < MIN_SUPPORT:                          # not in the cited source: try them all
            best = max(range(len(passages)), key=lambda i: support(correct, passages[i]["text"]))
            score, source = support(correct, passages[best]["text"]), best + 1
        if score < MIN_SUPPORT:
            dropped += 1                                 # the sources do not back this answer
            reasons.append(f"unsupported ({score:.2f}): answer {correct!r} to {question[:60]!r}")
            continue
        seen.add(question.lower())
        kept.append(QuizQuestion(question, options, answer, explanation, source, round(score, 2)))
    return kept, dropped


class QuizMaker:

    def __init__(self, pipeline, path=None):
        self.pipeline = pipeline
        self.path = path or settings.BASE_DIR / "outputs" / "cache" / "quizzes.jsonl"

    def _saved(self, subject_id, topic, n):
        if not self.path.exists():
            return []
        found = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                q = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (q.get("subject_id") == subject_id and q.get("topic", "").lower() == topic.lower()
                    and len(q.get("questions", [])) >= min(n, 3) and q.get("model") == settings.LLM_SERVED_MODEL):
                found.append(q)
        return found

    def make(self, topic, subject_id=None, filters=None, fallback_filters=None, n=5, fresh=False):
        """A quiz on `topic`: a saved one unless `fresh`, else newly written and checked."""
        if not fresh:
            saved = self._saved(subject_id, topic, n)
            if saved:
                pick = random.choice(saved)
                pick.pop("model", None)
                return Quiz(**pick)
        started = time.perf_counter()
        from src.pipeline import scoped_retrieve
        threshold = self.pipeline.router.calibration["in_domain_threshold"]
        passages, _scope = scoped_retrieve(self.pipeline.retriever, topic, filters, fallback_filters or [None],
                                           threshold, top_k=5)
        if settings.CONTEXT_ENGINEERING and passages:
            passages, _stats = self.pipeline.context_builder.build(topic, passages, "10-mark")
        if not passages:
            return Quiz(topic, subject_id)
        from src.llm.prompts import format_sources
        prompt = QUIZ_PROMPT.format(n=n, topic=topic, sources=format_sources(passages))
        # One retry when a reply yields too few usable questions: on the CPU, one run kept
        # 0/5 (a malformed reply) and the next, same topic, kept 5/5 (14 Nov).
        questions, dropped = [], 0
        for _attempt in range(2):
            result = self.pipeline.llm.generate([{"role": "user", "content": prompt}],
                                                max_new_tokens=180 * n, temperature=0.4)
            got, lost = validate(_json_array(result.text or ""), passages)
            known = {q.question.lower() for q in questions}
            questions += [q for q in got if q.question.lower() not in known]
            dropped += lost
            if len(questions) >= min(n, 3):
                break
        quiz = Quiz(topic, subject_id, questions[:n], dropped=dropped,
                    sources=[{"number": i, "article_title": p.get("article_title"), "source_url": p.get("source_url"),
                              "text": p["text"]} for i, p in enumerate(passages, start=1)],
                    seconds=round(time.perf_counter() - started, 1), created=time.time())
        if quiz.questions:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**quiz.to_dict(), "model": settings.LLM_SERVED_MODEL}, ensure_ascii=False) + "\n")
        return quiz
