"""The operations dashboard renders from logs and needs the team code."""

import json
from datetime import datetime, timedelta, timezone

import pytest
from streamlit.testing.v1 import AppTest

from config import settings


@pytest.fixture
def logs(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    rows = [{"timestamp": (now - timedelta(hours=h)).isoformat(), "workflow": w, "total_ms": ms,
             "verified": v, "revised": False, "unsupported_claim_rate": 0.1 if v else None,
             "cited": c, "output_tokens": 300}
            for h, w, ms, v, c in [(1, "rag_llm", 4000, True, 2), (2, "direct_llm", 1500, False, 0),
                                   (30, "rag_llm", 6000, False, 1), (50, "vision_llm", 9000, False, 0)]]
    (tmp_path / "telemetry.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (tmp_path / "feedback.jsonl").write_text(json.dumps(
        {"timestamp": now.isoformat(), "rating": "helpful"}), encoding="utf-8")
    (tmp_path / "errors.jsonl").write_text(json.dumps(
        {"timestamp": now.isoformat(), "reference": "ab12cd34", "error": "TimeoutError"}), encoding="utf-8")
    monkeypatch.setattr(settings, "LOG_DIR", tmp_path)
    monkeypatch.setattr(settings, "STUDIO_CODE", "")


def test_dashboard_shows_usage(logs):
    at = AppTest.from_file("../pages/3_Operations_Dashboard.py", default_timeout=60).run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Questions"] == "4"
    assert metrics["Rated helpful"] == "100%"
    assert metrics["Failed answers"] == "1"
    assert metrics["Answered from sources"] == "50%"


def test_dashboard_needs_team_code(logs, monkeypatch):
    monkeypatch.setattr(settings, "STUDIO_CODE", "team-code")
    at = AppTest.from_file("../pages/3_Operations_Dashboard.py", default_timeout=60).run()
    assert not at.exception
    assert not at.metric
