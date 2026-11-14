"""
Previous-year question papers: split a paper into its questions, read the
marks, and match each question to a catalogue subject and topic.

Papers are laid out in a few common ways, which the splitter handles:

    Q1. (a) Explain the Rankine cycle.            (5 marks)
    2. Derive the EMF equation of a transformer. [10]
    b) What is a deadlock?                        (2M)
    Q.No 3  Compare TCP and UDP.                  10 Marks

Exam instructions ("Answer any FIVE questions", "Time: 3 hours", "Max. Marks:
70") and part headings ("PART - A") are dropped. Marks choose the answer
format: up to 3 marks -> 2-mark, up to 7 -> 5-mark, more -> 10-mark; none
stated -> 5-mark. Matching uses the catalogue's "subject: topic" strings and
the same MiniLM model as retrieval.
"""

import re
from dataclasses import asdict, dataclass

import numpy as np

QUESTION_START = re.compile(
    r"^\s*(?:Q\.?\s*(?:No\.?\s*)?)?(?P<num>\d{1,2})\s*[.)]?\s*(?:\(?(?P<sub1>[a-hivx]{1,3})[.)]\s*)?(?=\S)"
    r"|^\s*\(?(?P<sub2>[a-h])[.)]\s+(?=\S)", re.IGNORECASE)
MARKS = re.compile(r"[(\[]\s*(\d{1,2})\s*(?:marks?|m)?\s*[)\]]\s*$|\b(\d{1,2})\s*marks?\s*$|\(\s*(\d{1,2})\s*M\s*\)",
                   re.IGNORECASE)
INSTRUCTION = re.compile(
    r"^\s*(answer\s+(any|all)|time\s*[:\-]|max(imum)?\.?\s*marks|note\s*[:\-]|instructions?|"
    r"part\s*[-–:]?\s*[a-c]\b|section\s*[-–:]?\s*[a-c]\b|unit\s*[-–:]?\s*[ivx\d]+\s*$|reg\.?\s*no|"
    r"(b\.?\s*tech|b\.?\s*e\.?)\b.*examination|question paper code|or\s*$)", re.IGNORECASE)


def marks_to_mode(marks):
    if marks is None:
        return "5-mark"
    return "2-mark" if marks <= 3 else "5-mark" if marks <= 7 else "10-mark"


@dataclass
class PaperQuestion:
    number: str
    text: str
    marks: int | None
    answer_mode: str
    subject_id: str | None = None
    subject_name: str | None = None
    topic: str | None = None
    match_score: float | None = None

    def to_dict(self):
        return asdict(self)


def _finish(number, lines):
    text = " ".join(l.strip() for l in lines if l.strip())
    marks = None
    m = MARKS.search(text)
    if m:
        marks = int(next(g for g in m.groups() if g))
        text = text[: m.start()].strip()
    text = re.sub(r"\s+", " ", text).strip(" .;:-?")
    if len(text.split()) < 3 or not re.search(r"[A-Za-z]{3}", text):
        return None
    return PaperQuestion(number, text + ("?" if text.lower().startswith(("what", "why", "how", "which",
                                                                          "when", "where", "who")) else "."),
                         marks, marks_to_mode(marks))


def split_questions(pages):
    """PaperQuestion objects from the text of a paper (a list of page strings)."""
    questions, current_num, current_lines = [], None, []
    last_main = None
    for line in "\n".join(pages).splitlines():
        if not line.strip() or INSTRUCTION.match(line):
            continue
        m = QUESTION_START.match(line)
        if m:
            if current_lines:
                q = _finish(current_num, current_lines)
                if q:
                    questions.append(q)
            main = m.group("num") or last_main
            last_main = main
            sub = m.group("sub1") or m.group("sub2")
            current_num = f"{main}{f'({sub.lower()})' if sub else ''}" if main else (sub or "?")
            current_lines = [line[m.end():]]
        elif current_lines:
            current_lines.append(line)            # a question that wraps onto the next line
    if current_lines:
        q = _finish(current_num, current_lines)
        if q:
            questions.append(q)
    return questions


class SubjectMatcher:
    """Matches questions to the catalogue's subjects and topics by meaning."""

    def __init__(self, catalogue, embedder):
        self.embedder = embedder
        self.entries = [(s["id"], s["name"], t) for s in catalogue["subjects"] for t in s["topics"]]
        texts = [f"{name}: {topic}" for _, name, topic in self.entries]
        self.vectors = np.asarray(embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False))

    def match(self, questions, subject_hint=None):
        """Fill subject and topic on each question. `subject_hint`: a subject id the
        whole paper belongs to (from the upload form), which restricts the match."""
        if not questions:
            return questions
        allowed = [i for i, e in enumerate(self.entries) if not subject_hint or e[0] == subject_hint]
        vectors = self.vectors[allowed]
        q = np.asarray(self.embedder.encode([x.text for x in questions], normalize_embeddings=True,
                                            show_progress_bar=False))
        scores = q @ vectors.T
        for question, row in zip(questions, scores):
            best = int(row.argmax())
            sid, name, topic = self.entries[allowed[best]]
            question.subject_id, question.subject_name, question.topic = sid, name, topic
            question.match_score = round(float(row[best]), 3)
        return questions
