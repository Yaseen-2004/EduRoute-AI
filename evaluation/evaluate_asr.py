"""
Speech recognition accuracy (word error rate) for voice questions (report §13, §22.2).

Real recordings (the measurement that counts):
  1. Put recordings in a folder, e.g. evaluation/voice_eval/  (wav, mp3, m4a, webm, ogg)
  2. Add evaluation/voice_eval/references.csv with columns:
         file,reference,speaker,accent
     `reference` is exactly what was said. Speaker/accent are optional labels
     for the breakdown (e.g. S1, telugu-english). Get consent before recording.
  3. venv\\Scripts\\python -m evaluation.evaluate_asr evaluation\\voice_eval

Synthetic baseline (checks the pipeline end to end; NOT student voices):
     venv\\Scripts\\python -m evaluation.evaluate_asr --make-synthetic 20
  Reads benchmark questions aloud with two Indian-English neural voices
  (Edge-TTS, needs internet) into evaluation/voice_eval_synthetic/, then scores them.

WER = (substitutions + deletions + insertions) / reference words, after
lower-casing and removing punctuation. Reported overall (pooled), per speaker
and per accent, with latency and real-time factor.
"""

import argparse
import csv
import random
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

from config import settings
from src.utils.io import load_jsonl, write_json

RESULTS = settings.BASE_DIR / "evaluation" / "results"
SYNTHETIC_DIR = settings.BASE_DIR / "evaluation" / "voice_eval_synthetic"
SYNTHETIC_VOICES = ["en-IN-NeerjaNeural", "en-IN-PrakashNeural"]


def normalise(text):
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^\w\s']", " ", text)
    return text.split()


def word_errors(reference, hypothesis):
    """Levenshtein distance over words."""
    ref, hyp = normalise(reference), normalise(hypothesis)
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h))
        previous = current
    return previous[-1], len(ref)


def char_errors(reference, hypothesis):
    """Levenshtein distance over characters (spaces and punctuation removed): the usual
    measure for Indian scripts, where word boundaries and spellings vary."""
    # Keep letters, combining marks and digits. Python's \W also strips the vowel signs of
    # Indian scripts (category M), which would hide real errors ("है" vs "हैं").
    import unicodedata

    def keep(text):
        return "".join(c for c in text.lower() if unicodedata.category(c)[0] in "LMN")
    ref, hyp = keep(reference), keep(hypothesis)
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h))
        previous = current
    return previous[-1], len(ref)


def make_synthetic_indic(n, seed, language):
    """Benchmark questions translated by NLLB and read by that language's Indian neural voice
    (Edge-TTS, needs internet), into evaluation/voice_eval_synthetic_<lang>/."""
    import asyncio

    import edge_tts
    try:                                   # use the Windows certificate store (HTTPS inspection on some networks)
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass

    from src.language.languages import LANGUAGES
    from src.language.translate import translate_sentences
    lang = LANGUAGES[language]
    questions = load_jsonl(settings.BASE_DIR / "evaluation" / "benchmarks" / "retrieval_engineering_v1.jsonl")
    sample = random.Random(seed).sample(questions, min(n, len(questions)))
    texts = translate_sentences([q["question"] for q in sample], "en", language)
    folder = settings.BASE_DIR / "evaluation" / f"voice_eval_synthetic_{language}"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for q, text in zip(sample, texts):
        name = f"{q['id']}_{language}.mp3"

        async def speak(t=text, path=folder / name):
            await edge_tts.Communicate(t, lang.voice).save(str(path))
        asyncio.run(speak())
        rows.append({"file": name, "reference": text, "speaker": lang.voice, "accent": f"synthetic-{language}",
                     "english": q["question"]})
        print(f"  {name}: {text}", flush=True)
    with (folder / "references.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "reference", "speaker", "accent", "english"])
        writer.writeheader()
        writer.writerows(rows)
    return folder


def make_synthetic(n, seed):
    from src.voice.speech import synthesize

    questions = load_jsonl(settings.BASE_DIR / "evaluation" / "benchmarks" / "retrieval_engineering_v1.jsonl") \
        + load_jsonl(settings.BASE_DIR / "evaluation" / "benchmarks" / "retrieval_aiml_v1.jsonl")
    sample = random.Random(seed).sample(questions, min(n, len(questions)))
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, q in enumerate(sample):
        voice = SYNTHETIC_VOICES[i % len(SYNTHETIC_VOICES)]
        audio, engine = synthesize(q["question"], voice=voice)
        label = voice.split("-")[2] if engine == "edge-tts" else engine  # fallback engine named honestly
        name = f"{q['id']}_{label}.mp3"
        (SYNTHETIC_DIR / name).write_bytes(audio)
        rows.append({"file": name, "reference": q["question"], "speaker": voice if engine == "edge-tts" else engine,
                     "accent": "synthetic-en-IN"})
        print(f"  {name} ({engine})", flush=True)
    with (SYNTHETIC_DIR / "references.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "reference", "speaker", "accent"])
        writer.writeheader()
        writer.writerows(rows)
    return SYNTHETIC_DIR


def evaluate(folder, use_subject_prompt, model=None, language=None):
    from src.voice.speech import SpeechRecognizer

    folder = Path(folder)
    refs = list(csv.DictReader(open(folder / "references.csv", encoding="utf-8")))
    english = language in (None, "en")
    recognizer = SpeechRecognizer(model_size=model if english else None,
                                  multilingual_size=None if english else model)
    score = word_errors if english else char_errors
    rows = []
    for ref in refs:
        path = folder / ref["file"]
        if not path.exists():
            print(f"  missing: {path.name}")
            continue
        started = time.perf_counter()
        t = recognizer.transcribe(path.read_bytes(), suffix=path.suffix,
                                  subject_name=ref.get("subject") if use_subject_prompt else None,
                                  language=language)
        errors, words = score(ref["reference"], t.text)
        rows.append({"file": ref["file"], "speaker": ref.get("speaker") or "-", "accent": ref.get("accent") or "-",
                     "reference": ref["reference"], "hypothesis": t.text, "errors": errors, "words": words,
                     "wer": round(errors / words, 3) if words else None,
                     "duration_s": t.duration_s, "latency_s": round(time.perf_counter() - started, 2),
                     "low_confidence": bool(t.low_confidence_segments), "reliable": t.reliable})
        print(f"  {ref['file']:<28} WER {rows[-1]['wer']}  | {t.text}", flush=True)
    if not rows:
        raise SystemExit("No audio files were found.")

    def pooled(subset):
        words = sum(r["words"] for r in subset)
        return round(sum(r["errors"] for r in subset) / words, 3) if words else None

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "folder": str(folder), "language": language or "en",
        "whisper_model": (model or settings.WHISPER_MODEL_SIZE) if english else (model or settings.WHISPER_MULTILINGUAL_MODEL),
        "metric": "WER" if english else "CER (characters)",
        "files": len(rows), "wer": pooled(rows),
        "exact_transcripts": round(sum(r["errors"] == 0 for r in rows) / len(rows), 3),
        # The app asks the student to repeat or type when a transcript is in the wrong script.
        "refused_share": round(sum(not r["reliable"] for r in rows) / len(rows), 3),
        "error_rate_of_accepted": pooled([r for r in rows if r["reliable"]]),
        "by_speaker": {s: pooled([r for r in rows if r["speaker"] == s]) for s in sorted({r["speaker"] for r in rows})},
        "by_accent": {a: pooled([r for r in rows if r["accent"] == a]) for a in sorted({r["accent"] for r in rows})},
        "median_latency_s": round(statistics.median(r["latency_s"] for r in rows), 2),
        "real_time_factor": round(sum(r["latency_s"] for r in rows) / max(1e-6, sum(r["duration_s"] or 0 for r in rows)), 2),
        "rows": rows,
    }
    out = RESULTS / f"asr_eval_{folder.name}_{report['whisper_model']}_{datetime.now():%Y%m%d_%H%M%S}.json"
    write_json(out, report)
    print(f"\nFiles {report['files']}  {report['metric']} {report['wer']}  exact {report['exact_transcripts']}  "
          f"median latency {report['median_latency_s']} s  RTF {report['real_time_factor']}  "
          f"refused {report['refused_share']}  error rate of accepted {report['error_rate_of_accepted']}")
    print("By speaker:", report["by_speaker"])
    print("By accent:", report["by_accent"])
    print("Saved:", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", help="folder with audio files and references.csv")
    parser.add_argument("--make-synthetic", type=int, metavar="N",
                        help="create N synthetic Indian-English recordings first, then score them")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--subject-prompt", action="store_true",
                        help="pass the optional `subject` column to Whisper as a vocabulary hint")
    parser.add_argument("--model", help="Whisper model to test, e.g. base.en, small.en, small")
    parser.add_argument("--language", help="Whisper language code for non-English audio, e.g. hi, kn, te, ta")
    args = parser.parse_args()
    if args.make_synthetic and args.language and args.language != "en":
        folder = make_synthetic_indic(args.make_synthetic, args.seed, args.language)
    elif args.make_synthetic:
        folder = make_synthetic(args.make_synthetic, args.seed)
    else:
        folder = args.folder
    if not folder:
        parser.error("give a folder or --make-synthetic N")
    evaluate(folder, args.subject_prompt, model=args.model, language=args.language)


if __name__ == "__main__":
    main()
