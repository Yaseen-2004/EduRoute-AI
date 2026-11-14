"""The Practice page shows a quiz, marks it and explains each answer."""

from streamlit.testing.v1 import AppTest

from config import settings

QUIZ = {"topic": "t", "subject_id": None, "dropped": 1, "seconds": 1.0, "created": 0.0,
        "sources": [{"number": 1, "article_title": "Rankine cycle", "source_url": "https://en.wikipedia.org/wiki/Rankine_cycle",
                     "text": "..."}],
        "questions": [
            {"question": "Where does steam expand to produce work?", "options": ["Turbine", "Pump", "Condenser", "Tank"],
             "answer": 0, "explanation": "Steam expands in the turbine.", "source": 1, "support": 1.0},
            {"question": "Which component raises the water pressure?", "options": ["Turbine", "Pump", "Condenser", "Tank"],
             "answer": 1, "explanation": "The pump raises the pressure.", "source": 1, "support": 1.0}]}


def test_quiz_is_marked(monkeypatch):
    monkeypatch.setattr(settings, "ACCESS_CODE", "")
    at = AppTest.from_file("../pages/5_Practice.py", default_timeout=60).run()
    assert not at.exception
    key = (at.selectbox[1].value, at.selectbox[2].value)              # the subject and topic on screen
    at.session_state["quiz"], at.session_state["quiz_key"] = {**QUIZ, "subject_id": key[0]}, key
    at.run()
    assert len(at.radio) == 2
    at.radio[0].set_value(0)                                            # right
    at.radio[1].set_value(0)                                            # wrong
    at.button(key="FormSubmitter:quiz_form-Check my answers").click().run()
    assert not at.exception
    assert at.subheader[0].value == "Score: 1 / 2"
    shown = " ".join(m.value for m in at.markdown)
    assert "Correct answer: **Pump**" in shown and "Your answer: Turbine" in shown
