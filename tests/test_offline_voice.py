"""Read-aloud works without internet: the offline system voice is the last fallback."""

import io
import wave

import pytest

import src.voice.speech as speech

needs_voice = pytest.mark.skipif(not speech.offline_voice_available(), reason="no offline voice on this OS")


@needs_voice
def test_offline_voice_produces_speech():
    audio = speech.synthesize_offline("The Rankine cycle has four main components.")
    with wave.open(io.BytesIO(audio)) as w:
        assert w.getnframes() / w.getframerate() > 1.0          # at least a second of speech
    assert speech.audio_mime(audio) == "audio/wav"


@needs_voice
def test_falls_back_when_online_voices_fail(monkeypatch):
    """12-14 Nov: with the Wi-Fi down, edge-tts and gTTS both failed and read-aloud broke."""
    real = speech.importlib.util.find_spec
    monkeypatch.setattr(speech.importlib.util, "find_spec",
                        lambda name, *a: None if name in ("edge_tts", "gtts") else real(name, *a))
    audio, engine = speech.synthesize("Ohm's law: V = I R [1].")
    assert engine == "offline system voice" and audio[:4] == b"RIFF"


def test_audio_type_detection():
    assert speech.audio_mime(b"RIFF....WAVEfmt ") == "audio/wav"
    assert speech.audio_mime(b"\xff\xfb\x90d") == "audio/mp3"
