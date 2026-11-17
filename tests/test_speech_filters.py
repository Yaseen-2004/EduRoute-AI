"""Speech-to-text clean-up: Whisper's silence inventions and stuck loops are dropped,
real speech is kept."""

from types import SimpleNamespace

import pytest

from src.voice.speech import STUCK, is_hallucination


def seg(text, logprob=-0.3, no_speech=0.05):
    return SimpleNamespace(text=text, avg_logprob=logprob, no_speech_prob=no_speech)


@pytest.mark.parametrize("text", ["Thank you for watching!", " Thank you.", "Please subscribe", "...", "you",
                                  "Subtitles by the Amara.org community"])
def test_stock_silence_phrases_are_dropped(text):
    assert is_hallucination(seg(text))


def test_unsure_non_speech_is_dropped_but_real_speech_kept():
    assert is_hallucination(seg("the the the", logprob=-1.4, no_speech=0.9))
    assert not is_hallucination(seg("Explain the Rankine cycle", logprob=-0.2, no_speech=0.01))
    assert not is_hallucination(seg("Thank you, now explain the Rankine cycle."))   # a real question


@pytest.mark.parametrize("text,stuck", [
    ("nananananana", True), ("lalalala", True),
    ("Explain the Mississippi delta", False), ("10000 rpm", False), ("Wait....", False),
    ("What is a 3-phase induction motor?", False),
])
def test_stuck_decoder_loops_are_recognised(text, stuck):
    assert bool(STUCK.search(text)) is stuck


CATALOGUE = {"subjects": [{"id": "ME1", "name": "Thermodynamics", "topics": ["Rankine cycle", "Entropy"]},
                          {"id": "CS1", "name": "Operating Systems", "topics": ["Deadlock"]}]}


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def retrieve(self, text, top_k=5):
        return self.hits


def test_subject_finder_votes_and_ignores_weak_matches():
    from src.voice.subject_hint import make_subject_finder
    strong = [{"subject_id": "CS1", "similarity_score": 0.62}, {"subject_id": "ME1", "similarity_score": 0.6},
              {"subject_id": "ME1", "similarity_score": 0.58}, {"subject_id": "ME1", "similarity_score": 0.5}]
    assert make_subject_finder(FakeRetriever(strong), CATALOGUE)("x") == ("Thermodynamics", ["Rankine cycle", "Entropy"])
    weak = [{"subject_id": "ME1", "similarity_score": 0.3}]
    assert make_subject_finder(FakeRetriever(weak), CATALOGUE)("x") is None


def test_second_pass_uses_the_found_subject_only_when_none_was_chosen(monkeypatch):
    from src.voice import speech
    calls = []

    def fake_decode(self, audio, subject_name=None, topics=None):
        calls.append(subject_name)
        return [SimpleNamespace(text=f"heard with {subject_name}", avg_logprob=-0.2)], \
            SimpleNamespace(language="en", duration=2.0)
    monkeypatch.setattr(speech.SpeechRecognizer, "_decode", fake_decode)
    monkeypatch.setattr(speech.SpeechRecognizer, "_load", lambda self: None)
    monkeypatch.setattr(speech, "decode_audio", lambda path: [0.0])
    monkeypatch.setattr(speech.settings, "WHISPER_TERM_CORRECTION", False)
    rec = speech.SpeechRecognizer()
    finder = lambda text: ("Thermodynamics", ["Rankine cycle"])      # noqa: E731

    t = rec.transcribe(b"RIFF", ".wav", subject_name="Mechanical Engineering", find_subject=finder)
    assert calls == ["Mechanical Engineering", "Thermodynamics"] and t.subject_guess == "Thermodynamics"
    assert t.text == "heard with Thermodynamics"

    calls.clear()
    t = rec.transcribe(b"RIFF", ".wav", subject_name="Operating Systems", topics=["Deadlock"], find_subject=finder)
    assert calls == ["Operating Systems"] and t.subject_guess is None              # subject chosen: one pass
