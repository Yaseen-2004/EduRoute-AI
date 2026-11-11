"""Headless Streamlit smoke test (loads the real V4 index; ~30-60 s on CPU)."""

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.slow
def test_app_answers_a_question_end_to_end():
    at = AppTest.from_file("../app.py", default_timeout=180).run()
    assert not at.exception
    assert at.title[0].value == "EduRoute AI"

    at.chat_input[0].set_value("Explain backpropagation in neural networks.").run()
    assert not at.exception

    rendered = " ".join(m.value for m in at.markdown)
    assert "backpropagation" in rendered.lower()
    # The answer must be labelled with its workflow and show sources.
    assert any("knowledge base" in c.value.lower() for c in at.caption)
    assert any("Sources" in e.label for e in at.expander)
