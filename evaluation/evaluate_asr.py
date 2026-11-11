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


def evaluate(folder, use_subject_prompt):
    from src.voice.speech import SpeechRecognizer

    folder = Path(folder)
    refs = list(csv.DictReader(open(folder / "references.csv", encoding="utf-8")))
    recognizer = SpeechRecognizer()
    rows = []
    for ref in refs:
        path = folder / ref["file"]
        if not path.exists():
            print(f"  missing: {path.name}")
            continue
        started = time.perf_counter()
        t = recognizer.transcribe(path.read_bytes(), suffix=path.suffix,
                                  subject_name=ref.get("subject") if use_subject_prompt else None)
        errors, words = word_errors(ref["reference"], t.text)
        rows.append({"file": ref["file"], "speaker": ref.get("speaker") or "-", "accent": ref.get("accent") or "-",
                     "reference": ref["reference"], "hypothesis": t.text, "errors": errors, "words": words,
                     "wer": round(errors / words, 3) if words else None,
                     "duration_s": t.duration_s, "latency_s": round(time.perf_counter() - started, 2),
                     "low_confidence": bool(t.low_confidence_segments)})
        print(f"  {ref['file']:<28} WER {rows[-1]['wer']}  | {t.text}", flush=True)
    if not rows:
        raise SystemExit("No audio files were found.")

    def pooled(subset):
        words = sum(r["words"] for r in subset)
        return round(sum(r["errors"] for r in subset) / words, 3) if words else None

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "folder": str(folder), "whisper_model": settings.WHISPER_MODEL_SIZE,
        "files": len(rows), "wer": pooled(rows),
        "exact_transcripts": round(sum(r["errors"] == 0 for r in rows) / len(rows), 3),
        "by_speaker": {s: pooled([r for r in rows if r["speaker"] == s]) for s in sorted({r["speaker"] for r in rows})},
        "by_accent": {a: pooled([r for r in rows if r["accent"] == a]) for a in sorted({r["accent"] for r in rows})},
        "median_latency_s": round(statistics.median(r["latency_s"] for r in rows), 2),
        "real_time_factor": round(sum(r["latency_s"] for r in rows) / max(1e-6, sum(r["duration_s"] or 0 for r in rows)), 2),
        "rows": rows,
    }
    out = RESULTS / f"asr_eval_{folder.name}_{datetime.now():%Y%m%d_%H%M%S}.json"
    write_json(out, report)
    print(f"\nFiles {report['files']}  WER {report['wer']}  exact {report['exact_transcripts']}  "
          f"median latency {report['median_latency_s']} s  RTF {report['real_time_factor']}")
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
    args = parser.parse_args()
    folder = make_synthetic(args.make_synthetic, args.seed) if args.make_synthetic else args.folder
    if not folder:
        parser.error("give a folder or --make-synthetic N")
    evaluate(folder, args.subject_prompt)


if __name__ == "__main__":
    main()
