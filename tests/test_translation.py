"""Answer translation keeps the Markdown structure, citations, code and formulas."""

import pytest

from src.language import translate


@pytest.fixture
def upper(monkeypatch):
    """A stand-in translator: upper-cases text, so we can see exactly what was translated."""
    seen = []

    def fake(sentences, src="en", tgt="hi", **kw):
        seen.extend(sentences)
        return [s.upper() for s in sentences]
    monkeypatch.setattr(translate, "translate_sentences", fake)
    return seen


ANSWER = """## Rankine cycle
The cycle has four parts [1]. Steam drives the turbine [2][3].

- The pump raises pressure.
- Use `pip install x` and the formula $\\eta = W/Q$ here.

| Part | Role |
|---|---|
| Boiler | Heats water |

```python
print("keep me")
```"""


def test_structure_citations_code_and_formulas_survive(upper):
    out = translate.translate_answer(ANSWER, "hi", keep_terms=False)
    lines = out.split("\n")
    assert lines[0] == "## RANKINE CYCLE"
    assert lines[1] == "THE CYCLE HAS FOUR PARTS. [1] STEAM DRIVES THE TURBINE. [2][3]"
    assert lines[3].startswith("- THE PUMP")
    assert "`pip install x`" in lines[4] and "$\\eta = W/Q$" in lines[4]     # protected spans restored
    assert lines[6] == "| PART | ROLE |" and lines[7] == "|---|---|"
    assert 'print("keep me")' in out and "PRINT" not in out                  # code block untouched
    assert not any("[1]" in s or "`" in s for s in upper)                    # never sent to the model


def test_lost_placeholder_is_appended_not_dropped(monkeypatch):
    monkeypatch.setattr(translate, "translate_sentences", lambda s, *a, **k: ["text without markers" for _ in s])
    out = translate.translate_answer("Run `make all` now.", "hi")
    assert "`make all`" in out                                               # a formula/code span is never lost


def test_transliterated_placeholder_is_restored(monkeypatch):
    # NLLB wrote "X0X" as "एक्स0एक्स" in Hindi (17 Nov); the term must still go back in place.
    import re
    monkeypatch.setattr(translate, "glossary", lambda: re.compile(r"\b(orthographic projection)\b", re.I))
    monkeypatch.setattr(translate, "translate_sentences",
                        lambda s, *a, **k: ["एक्स0एक्स एक विधि है।" for _ in s])
    out = translate.translate_answer("Orthographic projection is a method.", "hi", keep_terms=True)
    assert out == "Orthographic projection एक विधि है।"


def test_english_is_returned_unchanged(upper):
    assert translate.translate_answer(ANSWER, "en") == ANSWER and upper == []


def test_questions_are_translated_to_english(upper):
    assert translate.to_english("रैंकिन चक्र क्या है?", "hi") == "रैंकिन चक्र क्या है?".upper()
    assert translate.to_english("Explain it", "en") == "Explain it"


def test_technical_terms_stay_in_english(upper, monkeypatch):
    import re
    monkeypatch.setattr(translate, "glossary", lambda: re.compile(r"\b(deadlock|steam|rankine cycle)\b", re.I))
    out = translate.translate_answer("Steam drives the Rankine cycle. A deadlock stops processes.", "ta",
                                     keep_terms=True)
    assert "Steam" in out and "Rankine cycle" in out and "deadlock" in out       # kept as written
    assert "DRIVES THE" in out and "STOPS PROCESSES" in out                       # the rest translated
    assert not any("deadlock" in s.lower() or "steam" in s.lower() for s in upper)
