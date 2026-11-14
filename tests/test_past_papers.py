"""Splitting previous-year papers into questions and matching them to subjects."""

from src.data_collection.past_papers import SubjectMatcher, marks_to_mode, split_questions

PAPER = """B.Tech Degree Examination, May 2025
Thermal Engineering
Time: 3 hours                         Max. Marks: 70
Answer any FIVE questions. All questions carry equal marks.
PART - A
Q1. (a) Explain the Rankine cycle with a neat sketch.        (5 marks)
    (b) What is the function of a condenser in a steam
        power plant?                                         (2M)
2. Derive the expression for the efficiency of an Otto cycle.  [10]
Q.No 3  Compare two-stroke and four-stroke engines.          10 Marks
OR
4. Define dryness fraction.
"""


def test_paper_is_split_into_questions_with_marks():
    qs = split_questions([PAPER])
    got = [(q.number, q.marks, q.answer_mode) for q in qs]
    assert got == [("1(a)", 5, "5-mark"), ("1(b)", 2, "2-mark"), ("2", 10, "10-mark"),
                   ("3", 10, "10-mark"), ("4", None, "5-mark")]
    assert qs[1].text == "What is the function of a condenser in a steam power plant?"     # wrapped line joined
    assert not any("Answer any" in q.text or "Max" in q.text for q in qs)                   # instructions dropped


def test_marks_to_mode():
    assert [marks_to_mode(m) for m in (None, 2, 3, 5, 7, 8, 16)] == \
        ["5-mark", "2-mark", "2-mark", "5-mark", "5-mark", "10-mark", "10-mark"]


class TopicEmbedder:
    VOCAB = ["rankine", "steam", "otto", "engine", "deadlock", "process", "transformer", "emf", "cycle"]

    def encode(self, texts, **kw):
        import numpy as np
        out = [np.array([t.lower().count(w) for w in self.VOCAB], dtype=float) + 0.01 for t in texts]
        return np.vstack([v / np.linalg.norm(v) for v in out])


CATALOGUE = {"subjects": [
    {"id": "ME1", "name": "Thermal Engineering", "topics": ["Rankine cycle and steam plants", "Otto cycle engines"]},
    {"id": "CS1", "name": "Operating Systems", "topics": ["Deadlock and process scheduling"]},
    {"id": "EE1", "name": "Electrical Machines", "topics": ["Transformer EMF equation"]},
]}


def test_questions_are_matched_to_subjects():
    qs = SubjectMatcher(CATALOGUE, TopicEmbedder()).match(split_questions([PAPER]))
    assert qs[0].subject_id == "ME1" and "Rankine" in qs[0].topic
    assert qs[2].topic == "Otto cycle engines"
    hinted = SubjectMatcher(CATALOGUE, TopicEmbedder()).match(split_questions(["1. What is a deadlock in an OS?"]),
                                                             subject_hint="CS1")
    assert hinted[0].subject_id == "CS1"
