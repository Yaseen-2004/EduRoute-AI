"""Tests for the evaluation tools and log retention (no models, no network)."""

import csv
import json
from datetime import datetime, timedelta, timezone

import pytest

from evaluation.analyse_grading import weighted_kappa
from evaluation.evaluate_asr import word_errors
from evaluation.run_image_benchmark import facts_found
from src.utils.retention import purge_logs


def test_retention_removes_only_old_records(tmp_path):
    now = datetime(2026, 11, 9, tzinfo=timezone.utc)
    log = tmp_path / "telemetry.jsonl"
    log.write_text("".join(json.dumps({"timestamp": (now - timedelta(days=d)).isoformat(), "d": d}) + "\n"
                           for d in (1, 29, 31, 90)) + '{"no_timestamp": true}\n', encoding="utf-8")
    removed = purge_logs(days=30, log_dir=tmp_path, now=now)
    kept = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert removed == {"telemetry.jsonl": 2}
    assert [k.get("d") for k in kept] == [1, 29, None]   # undated records are never deleted
    assert purge_logs(days=0, log_dir=tmp_path, now=now) == {}


def test_word_error_rate_counts_edits_after_normalising():
    assert word_errors("What is Ohm's law?", "what is ohm's law") == (0, 4)
    assert word_errors("What is a Hohmann transfer orbit?", "What is a Homan transfer orbit") == (1, 6)
    assert word_errors("Explain SLAM", "Explain the SLAM method") == (2, 2)


def test_image_answer_fact_matching():
    assert facts_found("The total resistance is 30 Ω and the current is 0.4 A.",
                       "30 ω|30 ohm|30;0.4 a|0.4") == (2, 2)
    assert facts_found("This is an OR gate.", "and gate|and") == (0, 1)


def test_gate_names_are_not_confused_with_english_words():
    # Regression: "XOR gate ... inputs and outputs" used to score as the AND gate.
    expected = "AND gate|AND|and gate|and-gate"
    assert facts_found("It is an XOR gate: inputs and outputs differ.", expected) == (0, 1)
    assert facts_found("It is a NAND gate.", expected) == (0, 1)
    assert facts_found("This is an AND gate.", expected) == (1, 1)
    assert facts_found("R = 300 Ω and I = 0.45 A", "30 Ω|30 ohm;0.4 A") == (0, 2)


def test_questions_that_depend_on_the_image_are_detected():
    from src.pipeline import depends_on_image
    for q in ("Which logic gate does this truth table represent?", "Solve the problem shown in the image.",
              "Find the total resistance in this circuit.", "What does the graph above show?"):
        assert depends_on_image(q), q
    for q in ("Explain how a transformer works.", "What is Ohm's law?", ""):
        assert not depends_on_image(q), q


def test_weighted_kappa():
    assert weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0
    assert weighted_kappa([0, 0, 3, 3], [3, 3, 0, 0]) < 0
    assert weighted_kappa([2], [2]) is None


def test_claim_label_export_and_score(tmp_path, monkeypatch):
    from evaluation import claim_labels
    monkeypatch.setattr(claim_labels, "RESULTS", tmp_path)
    partial = tmp_path / "partial.jsonl"
    rows = []
    for i in range(30):
        unsupported = i % 3 == 0
        rows.append({"id": f"Q{i}", "strategy": "adaptive", "question": f"q{i}", "claims": [{
            "text": f"Claim number {i} states a fact about the topic.", "label": "unsupported" if unsupported else "supported",
            "support_score": 0.3 if unsupported else 0.7, "nli": None, "missing_numbers": [],
            "best_evidence": "evidence"}]})
    partial.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    claim_labels.export([partial], n=30, seed=1)
    sheet = next(tmp_path.glob("claim_labels_*.csv"))

    labelled = list(csv.DictReader(open(sheet, encoding="utf-8")))
    for r in labelled:  # a human who agrees with the verifier
        r["human_label"] = "unsupported" if r["system_label"] == "unsupported" else "supported"
    with open(sheet, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=claim_labels.FIELDS)
        writer.writeheader()
        writer.writerows(labelled)
    claim_labels.score(sheet)
    report = json.loads(sheet.with_suffix(".score.json").read_text(encoding="utf-8"))
    assert report["labelled_claims"] == 30
    assert report["verifier_as_deployed"]["accuracy"] == 1.0


def test_claim_label_export_refuses_old_runs(tmp_path, monkeypatch):
    from evaluation import claim_labels
    monkeypatch.setattr(claim_labels, "RESULTS", tmp_path)
    old = tmp_path / "old.jsonl"
    old.write_text(json.dumps({"id": "R1", "strategy": "direct"}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        claim_labels.export([old], n=10, seed=1)


def test_aicte_title_cleaning():
    from src.data_collection.compare_aicte import BRANCH_NAME_ONLY, NOT_A_COURSE, clean, normalise_title
    assert clean("Complier Design (New)") == "Compiler Design"          # AICTE's typo
    assert clean("Analog Circuit Laboratory[0L:0T:2P 1 credit]") == "Analog Circuit Laboratory"
    assert clean("IT Workshop (Sci") == "IT Workshop"                     # wrapped fragment
    assert normalise_title("Measurements & Metrology") == normalise_title("Measurements and Metrology")
    assert NOT_A_COURSE.search("Electronic Devices Lab") and NOT_A_COURSE.search("Introduction")
    assert not NOT_A_COURSE.search("Digital System Design")
    assert BRANCH_NAME_ONLY.match("Mechanical Engineering") and not BRANCH_NAME_ONLY.match("Automobile Engineering")


def test_streaming_reads_server_sent_events():
    from src.llm.backends import OpenAICompatibleBackend

    class FakeResponse:
        status_code = 200
        lines = ['data: {"model":"m","choices":[{"delta":{"content":"Ohm"}}]}', "",
                 'data: {"choices":[{"delta":{"content":"s law"}}]}',
                 'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3}}',
                 "data: [DONE]"]

        def iter_lines(self, decode_unicode=True):
            return iter(self.lines)

    class FakeSession:
        def post(self, *args, **kwargs):
            assert kwargs["stream"] and kwargs["json"]["stream"]
            return FakeResponse()

    backend = OpenAICompatibleBackend("http://x/v1", "m")
    backend._session = FakeSession()
    seen = []
    result = backend.generate([{"role": "user", "content": "q"}], on_token=seen.append)
    assert seen == ["Ohm", "Ohms law"]
    assert result.text == "Ohms law" and result.output_tokens == 3 and result.input_tokens == 12
    assert result.extra["first_token_s"] is not None


def test_speech_term_correction_is_conservative():
    from src.voice.term_correction import correct_terms
    vocab = {"hohmann transfer orbit": "Hohmann transfer orbit", "backpropagation": "Backpropagation",
             "applications of artificial intelligence": "Applications of artificial intelligence",
             "neural network": "Neural network", "rankine cycle": "Rankine cycle"}
    fixed, changes = correct_terms("What is a Homan transfer orbit?", vocab)
    assert fixed == "What is a Hohmann transfer orbit?" and changes == [("Homan transfer orbit", "Hohmann transfer orbit")]
    assert correct_terms("Explain back propagation in neural networks.", vocab)[0] == \
        "Explain backpropagation in neural networks."          # joined; plural and case kept
    # Small words differ ("in" vs "of"): the student's words are not changed.
    assert correct_terms("its application in artificial intelligence", vocab)[1] == []
    assert correct_terms("What is the base of a transistor?", vocab)[1] == []
    assert correct_terms("Explain the Rankine cycle.", vocab)[1] == []
