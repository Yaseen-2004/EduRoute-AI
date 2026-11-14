"""The Past papers page lists a parsed paper and offers an answer for each question."""

from streamlit.testing.v1 import AppTest

from config import settings

QUESTIONS = [
    {"number": "1(a)", "text": "Explain the Rankine cycle with a neat sketch.", "marks": 5, "answer_mode": "5-mark",
     "subject_id": "ME003", "subject_name": "Thermal Engineering", "topic": "Rankine cycle", "match_score": 0.71},
    {"number": "2", "text": "Define dryness fraction.", "marks": 2, "answer_mode": "2-mark",
     "subject_id": "ME003", "subject_name": "Thermal Engineering", "topic": "Steam", "match_score": 0.55},
]


def test_page_lists_questions(monkeypatch):
    monkeypatch.setattr(settings, "ACCESS_CODE", "")
    at = AppTest.from_file("../pages/4_Past_Papers.py", default_timeout=60)
    at.session_state["paper_questions"] = QUESTIONS
    at.run()
    assert not at.exception
    assert at.subheader[0].value == "2 questions found"
    assert len(at.dataframe[0].value) == 2
    assert any(b.label == "Answer this question" for b in at.button)
    assert at.segmented_control[0].value == "5-mark"           # format follows the paper's marks


def test_page_needs_the_access_code(monkeypatch):
    monkeypatch.setattr(settings, "ACCESS_CODE", "student-code")
    at = AppTest.from_file("../pages/4_Past_Papers.py", default_timeout=60).run()
    assert not at.exception and not at.dataframe
