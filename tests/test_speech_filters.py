"""Whisper's silence inventions are dropped; real speech is kept."""

from types import SimpleNamespace

import pytest

from src.language.languages import detect_language
from src.voice.speech import is_hallucination


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
    assert not is_hallucination(seg("रैंकिन चक्र समझाइए", logprob=-0.5, no_speech=0.1))


@pytest.mark.parametrize("text,expected", [
    ("Explain the Rankine cycle", "en"), ("रैंकिन चक्र क्या है?", "hi"), ("ರ್ಯಾಂಕಿನ್ ಚಕ್ರ ಎಂದರೇನು?", "kn"),
    ("రాంకిన్ చక్రం అంటే ఏమిటి?", "te"), ("ரேங்கின் சுழற்சி என்றால் என்ன?", "ta"),
    ("Explain Rankine cycle in हिंदी please", "en"),            # mostly English: stays English
])
def test_language_detection_by_script(text, expected):
    assert detect_language(text) == expected


def test_marathi_and_hindi_share_a_script():
    assert detect_language("रँकिन चक्र म्हणजे काय?", preferred="mr") == "mr"
    assert detect_language("रँकिन चक्र म्हणजे काय?") == "hi"


def test_script_share_flags_wrong_script_transcripts():
    from src.voice.speech import SCRIPT_SHARE_MIN, script_share
    assert script_share("ಭಾಗಶಃ ಡಿಸ್ಟಿಲೇಶನ್ ಎಂದರೇನು?", "kn") == 1.0
    assert script_share("वागश्यर दिस्टिलेशन इंदरेनू", "kn") == 0.0          # Kannada heard as Devanagari
    assert script_share("circumstant qualis involved two", "kn") < SCRIPT_SHARE_MIN
    assert script_share("RSA ಅಲ್ಗಾರಿದಮ್ ವಿವರಿಸಿ", "kn") >= SCRIPT_SHARE_MIN   # English terms mixed in
    assert script_share("anything", "en") == 1.0
