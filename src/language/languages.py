"""Languages EduRoute can answer in, and how each one is handled.

Each entry: display name, native name, NLLB-200 code (translation), Whisper code
(speech recognition) and an Edge neural voice (read-aloud). A language is only
offered in the app when it is listed in ENABLED, i.e. when its translation passed
the quality check in evaluation/evaluate_translation.py.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    native: str
    nllb: str
    whisper: str
    voice: str
    script_range: tuple | None = None      # Unicode block of its script (detection)


LANGUAGES = {
    "en": Language("en", "English", "English", "eng_Latn", "en", "en-IN-NeerjaNeural"),
    "hi": Language("hi", "Hindi", "हिन्दी", "hin_Deva", "hi", "hi-IN-SwaraNeural", (0x0900, 0x097F)),
    "kn": Language("kn", "Kannada", "ಕನ್ನಡ", "kan_Knda", "kn", "kn-IN-SapnaNeural", (0x0C80, 0x0CFF)),
    "te": Language("te", "Telugu", "తెలుగు", "tel_Telu", "te", "te-IN-ShrutiNeural", (0x0C00, 0x0C7F)),
    "ta": Language("ta", "Tamil", "தமிழ்", "tam_Taml", "ta", "ta-IN-PallaviNeural", (0x0B80, 0x0BFF)),
    "ml": Language("ml", "Malayalam", "മലയാളം", "mal_Mlym", "ml", "ml-IN-SobhanaNeural", (0x0D00, 0x0D7F)),
    "mr": Language("mr", "Marathi", "मराठी", "mar_Deva", "mr", "mr-IN-AarohiNeural", (0x0900, 0x097F)),
    "bn": Language("bn", "Bengali", "বাংলা", "ben_Beng", "bn", "bn-IN-TanishaaNeural", (0x0980, 0x09FF)),
    "gu": Language("gu", "Gujarati", "ગુજરાતી", "guj_Gujr", "gu", "gu-IN-DhwaniNeural", (0x0A80, 0x0AFF)),
}

# Set from the measured translation quality (evaluation/evaluate_translation.py).
ENABLED = ["en", "hi", "kn", "te", "ta", "ml", "mr", "bn", "gu"]

# Languages offered for spoken questions (Whisper `small` on the CPU). Gate fixed before
# testing: on 10 synthetic questions, the transcripts the app accepts have CER <= 0.35 and
# at most 30% are refused (evaluate_asr.py, 17 Nov). Others are typed instead.
VOICE_ENABLED = ["en"]


def detect_language(text, preferred=None):
    """The language of a question from its script; `preferred` breaks ties between
    languages sharing a script (Hindi/Marathi). English if no Indian script is used."""
    counts = {}
    for ch in text or "":
        cp = ord(ch)
        for code, lang in LANGUAGES.items():
            if lang.script_range and lang.script_range[0] <= cp <= lang.script_range[1]:
                counts[code] = counts.get(code, 0) + 1
    letters = sum(ch.isalpha() for ch in text or "") or 1
    if not counts or max(counts.values()) / letters < 0.3:
        return "en"
    best = max(counts.values())
    candidates = [c for c, n in counts.items() if n == best]
    if preferred in candidates:
        return preferred
    return "hi" if "hi" in candidates else candidates[0]
