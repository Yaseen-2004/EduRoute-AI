"""
Translation quality gate: which languages are good enough to offer?

No human references exist for these answers, so quality is measured by
round-trip translation (English -> language -> English) on real EduRoute text:

  chrF        character n-gram F-score (Popović, 2015) between the original and the
              round-trip English; 100 = identical. Measures meaning kept overall.
  terms       share of technical terms (content words of 6+ letters) of the original
              that come back in the round trip: does "turbine" survive?
  citations   answers whose [n] citations all survive the translation
  seconds     time to translate one answer into the language (CPU)

A language is recommended when its median chrF >= 50 and term survival >= 0.6
(both set before looking at the results). Round trips understate quality a little
(errors on the way back count too), so the gate errs on the side of caution.

Run: venv\\Scripts\\python -m evaluation.evaluate_translation [--answers 12] [--langs hi kn te ta]
"""

import argparse
import json
import re
import statistics
import time
from collections import Counter
from datetime import datetime

from config import settings
from src.language import translate
from src.language.languages import LANGUAGES
from src.utils.io import load_jsonl

RESULTS = settings.BASE_DIR / "evaluation" / "results"
CHRF_MIN, TERMS_MIN = 50, 0.6


def chrf(hyp, ref, n=6, beta=2):
    """chrF (character n-grams 1..n, recall weighted by beta), whitespace removed."""
    hyp, ref = re.sub(r"\s+", "", hyp.lower()), re.sub(r"\s+", "", ref.lower())
    precisions, recalls = [], []
    for k in range(1, n + 1):
        h = Counter(hyp[i:i + k] for i in range(len(hyp) - k + 1))
        r = Counter(ref[i:i + k] for i in range(len(ref) - k + 1))
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precisions.append(overlap / sum(h.values())); recalls.append(overlap / sum(r.values()))
    if not precisions:
        return 0.0
    p, r = statistics.mean(precisions), statistics.mean(recalls)
    return 0.0 if p + r == 0 else round(100 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r), 1)


def terms(text):
    return {w.lower() for w in re.findall(r"[A-Za-z]{6,}", text)}


def strip_markdown(text):
    text = re.sub(r"^\s*(#{1,6}|[-*•]|\d+[.)])\s*", "", text, flags=re.M)
    return re.sub(r"\[\d{1,2}\]|\*\*|__", "", text)


def sample_texts(n_answers):
    """Real answers from the saved question bank, plus benchmark questions."""
    answers = []
    if settings.ANSWER_CACHE_PATH.exists():
        for line in settings.ANSWER_CACHE_PATH.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            a = (r.get("response") or {}).get("answer", "")
            if r.get("pinned") and a and a not in answers:
                answers.append(a)
    questions = [q["question"] for q in load_jsonl(settings.BASE_DIR / "evaluation" / "benchmarks"
                                                  / "retrieval_engineering_v1.jsonl")][:10]
    return answers[:n_answers], questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=int, default=12)
    parser.add_argument("--langs", nargs="+", default=[c for c in LANGUAGES if c != "en"])
    parser.add_argument("--keep-terms", choices=["on", "off"], default="on",
                        help="keep technical terms in English inside the translation")
    args = parser.parse_args()
    answers, questions = sample_texts(args.answers)
    print(f"{len(answers)} answers, {len(questions)} questions", flush=True)
    report = {"generated_at": datetime.now().isoformat(timespec="seconds"), "model": translate.MODEL_DIR.name,
              "thresholds": {"chrf": CHRF_MIN, "terms": TERMS_MIN}, "keep_terms": args.keep_terms, "languages": {}}
    for code in args.langs:
        rows = []
        for text in answers + questions:
            t0 = time.perf_counter()
            forward = translate.translate_answer(text, code, keep_terms=args.keep_terms == "on")
            seconds = time.perf_counter() - t0
            back = translate.translate_answer(forward, "en", src=code)
            src_plain, back_plain = strip_markdown(text), strip_markdown(back)
            want = terms(src_plain)
            rows.append({"chrf": chrf(back_plain, src_plain),
                         "terms": len(want & terms(back_plain)) / len(want) if want else 1.0,
                         "citations_kept": sorted(re.findall(r"\[\d{1,2}\]", text)) ==
                                           sorted(re.findall(r"\[\d{1,2}\]", forward)),
                         "seconds": round(seconds, 2), "is_answer": text in answers,
                         "sample": {"en": text[:160], "xx": forward[:160], "back": back[:160]}})
        summary = {"median_chrf": statistics.median(r["chrf"] for r in rows),
                   "mean_terms": round(statistics.mean(r["terms"] for r in rows), 2),
                   "citations_kept": f"{sum(r['citations_kept'] for r in rows)}/{len(rows)}",
                   "median_seconds_per_answer": statistics.median(r["seconds"] for r in rows if r["is_answer"])
                   if answers else None}
        summary["recommended"] = summary["median_chrf"] >= CHRF_MIN and summary["mean_terms"] >= TERMS_MIN
        report["languages"][code] = {"summary": summary, "rows": rows}
        print(f"{code}: {summary}", flush=True)
    out = RESULTS / f"translation_eval_terms-{args.keep_terms}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Saved:", out)


if __name__ == "__main__":
    main()
