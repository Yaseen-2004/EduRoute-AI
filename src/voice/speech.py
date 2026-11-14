"""
Voice input and output (report §13).

ASR  faster-whisper (CPU int8). Returns the transcript plus per-segment
     confidence so the UI can show it for correction before routing.
TTS  Edge-TTS (neural voices, needs internet) with gTTS as fallback.
     Answers are converted to speakable text first: markdown, citation
     markers and URLs removed; common abbreviations expanded.

Both libraries are optional and imported lazily; availability() reports what
is installed so the UI never offers a feature that cannot run.
"""

import asyncio
import importlib.util
import io
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from config import settings

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac"}

# Spoken forms for abbreviations TTS engines often mispronounce.
PRONUNCIATIONS = {
    r"\bCNNs?\b": "C N N",
    r"\bRNNs?\b": "R N N",
    r"\bLSTMs?\b": "L S T M",
    r"\bGANs?\b": "GAN",
    r"\bLLMs?\b": "L L M",
    r"\bNLP\b": "N L P",
    r"\bMLOps\b": "M L Ops",
    r"\bSHAP\b": "shap",
    r"\bYOLO\b": "yolo",
    r"\bRAG\b": "rag",
    r"\bReLU\b": "rel-you",
    r"\bSQL\b": "sequel",
    r"\be\.g\.": "for example",
    r"\bi\.e\.": "that is",
    r"\betc\.": "et cetera",
    r"\bvs\.?": "versus",
}


class VoiceUnavailableError(RuntimeError):
    pass


def availability():
    return {
        "asr": importlib.util.find_spec("faster_whisper") is not None,
        "tts_edge": importlib.util.find_spec("edge_tts") is not None,
        "tts_gtts": importlib.util.find_spec("gtts") is not None,
        "tts_offline": offline_voice_available(),
    }


@dataclass
class Transcript:
    text: str
    language: str
    duration_s: float
    avg_logprob: float | None
    low_confidence_segments: list = field(default_factory=list)
    latency_s: float = 0.0
    corrections: list = field(default_factory=list)   # (heard, corrected) technical terms


def decode_audio(path, sampling_rate=16000):
    """
    Decode any container PyAV supports to mono float32 at 16 kHz.

    faster-whisper 1.2.1 passes an argument that PyAV 19 removed, so we
    decode ourselves and hand Whisper a NumPy array instead.
    """
    import av
    import numpy as np

    resampler = av.AudioResampler(format="s16", layout="mono", rate=sampling_rate)
    pieces = []
    with av.open(str(path), mode="r") as container:
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                pieces.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):  # flush
            pieces.append(out.to_ndarray().reshape(-1))
    if not pieces:
        raise ValueError("No audio samples could be decoded.")
    return np.concatenate(pieces).astype(np.float32) / 32768.0


class SpeechRecognizer:

    def __init__(self, model_size=None):
        self.model_size = model_size or settings.WHISPER_MODEL_SIZE
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as error:
                raise VoiceUnavailableError(
                    "Speech recognition needs faster-whisper: pip install faster-whisper"
                ) from error
            self._model = WhisperModel(self.model_size, device=settings.WHISPER_DEVICE,
                                       compute_type=settings.WHISPER_COMPUTE_TYPE)
        return self._model

    @staticmethod
    def vocabulary_prompt(subject_name=None, topics=None):
        """
        Whisper initial prompt that biases recognition towards the student's
        course vocabulary (technical terms are a known ASR weakness, §13.3).
        """
        if subject_name:
            terms = ", ".join((topics or [])[:8])
            return f"B.Tech engineering question about {subject_name}" + (f": {terms}." if terms else ".")
        return "B.Tech engineering question about a technical topic."

    def transcribe(self, audio_bytes, suffix=".wav", subject_name=None, topics=None):
        if not audio_bytes:
            raise ValueError("Empty audio.")
        if len(audio_bytes) > settings.MAX_AUDIO_BYTES:
            raise ValueError("Audio file is too large.")
        if suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported audio type {suffix}.")

        model = self._load()
        started = time.perf_counter()
        # Audio is written to a temporary file only for decoding and deleted
        # immediately (report §21: no retention of voice recordings).
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(audio_bytes)
            path = Path(handle.name)
        try:
            try:
                audio = decode_audio(path)
            except Exception as error:
                raise ValueError(f"Could not decode the audio file: {error}") from error
            segments, info = model.transcribe(
                audio, beam_size=settings.WHISPER_BEAM_SIZE, vad_filter=True,
                initial_prompt=self.vocabulary_prompt(subject_name, topics),
            )
            segments = list(segments)
        finally:
            path.unlink(missing_ok=True)

        text = " ".join(s.text.strip() for s in segments).strip()
        # Fix mis-heard technical terms against the knowledge base's vocabulary
        # (synthetic test set: word error rate 5.7% -> 4.3%).
        corrections = []
        if settings.WHISPER_TERM_CORRECTION and text:
            from src.voice.term_correction import correct_terms
            text, corrections = correct_terms(text)
        logprobs = [s.avg_logprob for s in segments]
        return Transcript(
            text=text,
            corrections=corrections,
            language=info.language,
            duration_s=round(info.duration, 2),
            avg_logprob=round(sum(logprobs) / len(logprobs), 3) if logprobs else None,
            low_confidence_segments=[s.text.strip() for s in segments if s.avg_logprob < -1.0],
            latency_s=round(time.perf_counter() - started, 2),
        )


def to_speakable(text, max_words=180):
    text = re.sub(r"\[\d{1,2}\]", "", text)                     # citations
    text = re.sub(r"https?://\S+", "", text)                    # URLs
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)               # code
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)  # headings
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_>|#]", "", text)
    for pattern, spoken in PRONUNCIATIONS.items():
        text = re.sub(pattern, spoken, text)
    text = re.sub(r"\s*\n\s*", ". ", text)
    text = re.sub(r"\.(\s*\.)+", ".", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\b(See|see|refer to)\s*\.?$", "", text).strip()
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rsplit(".", 1)[0] + "."
    return text


def synthesize(text, voice=None):
    """Return (audio_bytes, engine_name): MP3 from the online voices, WAV from the offline one. Raises VoiceUnavailableError."""
    speakable = to_speakable(text)
    if not speakable:
        raise ValueError("Nothing to speak.")
    errors = []

    try:  # edge-tts and gTTS use HTTPS; verify against the OS trust store
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass

    if importlib.util.find_spec("edge_tts"):
        try:
            import edge_tts

            async def _run():
                communicate = edge_tts.Communicate(speakable, voice or settings.TTS_VOICE)
                audio = bytearray()
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio.extend(chunk["data"])
                return bytes(audio)

            audio = asyncio.run(_run())
            if audio:
                return audio, "edge-tts"
        except Exception as error:  # network or service failure -> try gTTS
            errors.append(f"edge-tts: {error}")

    if importlib.util.find_spec("gtts"):
        try:
            from gtts import gTTS
            buffer = io.BytesIO()
            gTTS(speakable, lang="en", tld="co.in").write_to_fp(buffer)
            return buffer.getvalue(), "gTTS"
        except Exception as error:
            errors.append(f"gTTS: {error}")

    # Both engines above are online services; without internet (12-14 Nov: the
    # Wi-Fi dropped) read-aloud failed. The operating system's own voice works offline.
    try:
        audio = synthesize_offline(speakable)
        if audio:
            return audio, "offline system voice"
    except Exception as error:
        errors.append(f"offline voice: {error}")

    raise VoiceUnavailableError(
        "Text-to-speech failed (no internet for the online voices, and no offline voice "
        "available). " + "; ".join(errors)
    )


OFFLINE_TIMEOUT_S = 120

_WINDOWS_TTS = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = $s.GetInstalledVoices() | Where-Object { $_.Enabled } | ForEach-Object { $_.VoiceInfo }
$pick = $voices | Where-Object { $_.Culture.Name -eq 'en-IN' } | Select-Object -First 1
if (-not $pick) { $pick = $voices | Where-Object { $_.Culture.Name -like 'en-*' -and $_.Gender -eq 'Female' } | Select-Object -First 1 }
if (-not $pick) { $pick = $voices | Where-Object { $_.Culture.Name -like 'en-*' } | Select-Object -First 1 }
if ($pick) { $s.SelectVoice($pick.Name) }
$s.Rate = 0
$s.SetOutputToWaveFile($args[1])
$s.Speak([IO.File]::ReadAllText($args[0], [Text.Encoding]::UTF8))
$s.Dispose()
"""


def offline_voice_available():
    import shutil
    import sys
    return sys.platform == "win32" or bool(shutil.which("espeak-ng") or shutil.which("espeak"))


def synthesize_offline(text):
    """WAV bytes from the operating system's own voice (no internet):
    Windows SAPI (System.Speech) or espeak-ng/espeak on Linux."""
    import shutil
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        source, wav = Path(tmp) / "speech.txt", Path(tmp) / "speech.wav"
        source.write_text(text, encoding="utf-8")       # a file, not a command line: no length limit or quoting
        if sys.platform == "win32":
            script = Path(tmp) / "speak.ps1"
            script.write_text(_WINDOWS_TTS, encoding="utf-8")
            subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
                            str(source), str(wav)], check=True, capture_output=True, timeout=OFFLINE_TIMEOUT_S)
        else:
            engine = shutil.which("espeak-ng") or shutil.which("espeak")
            if not engine:
                raise VoiceUnavailableError("no offline voice (install espeak-ng)")
            subprocess.run([engine, "-v", "en", "-f", str(source), "-w", str(wav)], check=True,
                           capture_output=True, timeout=OFFLINE_TIMEOUT_S)
        return wav.read_bytes() if wav.exists() else b""


def audio_mime(audio):
    """'audio/wav' for the offline voice, 'audio/mp3' for the online ones."""
    return "audio/wav" if audio[:4] == b"RIFF" else "audio/mp3"
