"""The access gate blocks the app and the studio until the right code is entered."""

import pytest
from streamlit.testing.v1 import AppTest

from config import settings


@pytest.fixture
def codes(monkeypatch):
    monkeypatch.setattr(settings, "ACCESS_CODE", "student-code")
    monkeypatch.setattr(settings, "STUDIO_CODE", "team-code")


def test_studio_needs_the_team_code(codes):
    at = AppTest.from_file("../pages/2_Evaluation_Studio.py", default_timeout=60).run()
    assert not at.exception
    assert not at.tabs                                   # nothing of the studio is shown
    at.text_input[0].set_value("student-code")           # the students' code is not enough
    at.button[0].click().run()
    assert any("not right" in e.value for e in at.error)
    assert not at.tabs
    at.text_input[0].set_value("team-code")
    at.button[0].click().run()
    assert not at.exception
    assert at.tabs                                       # the studio opens


@pytest.mark.slow
def test_app_needs_the_access_code(codes):
    at = AppTest.from_file("../app.py", default_timeout=180).run()
    assert not at.exception
    assert not at.chat_input                             # no way to ask before the code
    at.text_input[0].set_value("student-code")
    at.button[0].click().run()
    assert not at.exception
    assert at.chat_input
