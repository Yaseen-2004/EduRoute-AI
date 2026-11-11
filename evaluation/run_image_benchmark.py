"""
Image-question benchmark (report §12, §22.2).

Runs each image question in evaluation/image_eval/questions.csv through the
pipeline with a vision model, for the adaptive router and the modality
baseline (image -> vision model only), and scores each answer against the
`expected` facts (all ';'-separated facts must appear; '|' separates accepted
spellings). Keyword scoring is a coarse automatic check; the answers are also
written to a grading CSV for human review.

Requires a vision model, e.g. on Colab:
    ollama pull qwen2.5vl:3b
    set EDUROUTE_VISION_BASE_URL=http://localhost:11434/v1
    set EDUROUTE_VISION_MODEL=qwen2.5vl:3b
    python -m evaluation.run_image_benchmark
"""

import argparse
import csv
import json
import re
import statistics
import traceback
from datetime import datetime
from pathlib import Path

from config import settings
from src.pipeline import EduRoutePipeline
from src.rag.retrieval import EduRouteRetriever
from src.utils.io import write_json

FOLDER = settings.BASE_DIR / "evaluation" / "image_eval"
RESULTS = settings.BASE_DIR / "evaluation" / "results"


def _matches(alternative, answer):
    """Whole-word match; an all-capitals alternative (e.g. AND) is case-sensitive,
    so the logic gate is not confused with the English word "and"."""
    alt = alternative.strip()
    if not alt:
        return False
    flags = 0 if alt.isupper() else re.IGNORECASE
    return re.search(r"(?<![\w.])" + re.escape(alt) + r"(?![\w])", answer or "", flags) is not None


_LATEX = [(r"\\text\{([^}]*)\}", r"\1"), (r"\\mathrm\{([^}]*)\}", r"\1"), (r"\\Omega", "Ω"),
          (r"\\(,|;|:|!| |quad)", " "), (r"\\[()\[\]]", " "), (r"\$", " "), (r"\s+", " ")]


def normalise_answer(answer):
    """Models often write maths in LaTeX ("30 \\, \\Omega"); score the plain form ("30 Ω")."""
    text = answer or ""
    for pattern, replacement in _LATEX:
        text = re.sub(pattern, replacement, text)
    return text


def facts_found(answer, expected):
    answer = normalise_answer(answer)
    groups = [g for g in expected.split(";") if g.strip()]
    hits = [any(_matches(alt, answer) for alt in g.split("|")) for g in groups]
    return sum(hits), len(groups)


def summarise(rows, strategies):
    summary = {}
    for s in strategies:
        mine = [r for r in rows if r["strategy"] == s and not r["error"]]
        if mine:
            summary[s] = {
                "answers": len(mine), "failures": sum(r["strategy"] == s and bool(r["error"]) for r in rows),
                "all_facts_correct": round(sum(r["all_facts"] for r in mine) / len(mine), 3),
                "fact_recall": round(sum(r["facts_found"] for r in mine) / max(1, sum(r["facts_total"] for r in mine)), 3),
                "median_latency_ms": round(statistics.median(r["total_ms"] for r in mine), 1),
                "workflows": {w: sum(r["workflow"] == w for r in mine) for w in {r["workflow"] for r in mine}},
            }
    return summary


def rescore(path, folder):
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = {i["file"]: i["expected"] for i in csv.DictReader(open(folder / "questions.csv", encoding="utf-8"))}
    for r in report["rows"]:
        if not r["error"]:
            r["facts_found"], r["facts_total"] = facts_found(r["answer"], expected[r["file"]])
            r["all_facts"] = r["facts_found"] == r["facts_total"]
    report["summary"] = summarise(report["rows"], sorted({r["strategy"] for r in report["rows"]}))
    report["rescored_at"] = datetime.now().isoformat(timespec="seconds")
    out = path.with_name(path.stem + "_rescored.json")
    write_json(out, report)
    for s, v in report["summary"].items():
        print(f"  {s:<10} all facts {v['all_facts_correct']}  fact recall {v['fact_recall']}")
    for r in report["rows"]:
        print(f"    {r['file']:<20} {r['strategy']:<10} {r.get('facts_found')}/{r.get('facts_total')}")
    print("Saved:", out)


def fail(reason):
    """Record why the run could not start, so a missing result is never silent."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"image_benchmark_FAILED_{datetime.now():%Y%m%d_%H%M%S}.json"
    write_json(out, {"generated_at": datetime.now().isoformat(timespec="seconds"), "status": "failed",
                     "reason": reason, "vision_base_url": settings.VISION_BASE_URL,
                     "vision_model": settings.VISION_MODEL})
    print(f"IMAGE BENCHMARK NOT RUN: {reason}\nRecorded in {out}", flush=True)
    raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=str(FOLDER))
    parser.add_argument("--strategies", nargs="+", default=["adaptive", "modality"])
    parser.add_argument("--rescore", metavar="RESULT_JSON",
                        help="re-score saved answers with the current questions.csv (no models needed)")
    args = parser.parse_args()
    if args.rescore:
        rescore(Path(args.rescore), Path(args.folder))
        return

    if not settings.VISION_BASE_URL or not settings.VISION_MODEL:
        fail("No vision model configured: set EDUROUTE_VISION_BASE_URL and EDUROUTE_VISION_MODEL.")
    folder = Path(args.folder)
    items = list(csv.DictReader(open(folder / "questions.csv", encoding="utf-8")))

    retriever = EduRouteRetriever(version=settings.RAG_VERSION, mode=settings.RETRIEVAL_MODE)
    pipelines = {s: EduRoutePipeline(strategy=s, retriever=retriever, log=False) for s in args.strategies}
    ok, detail = next(iter(pipelines.values())).vision_backend.health_check()
    print(f"Vision model: {'available' if ok else 'UNAVAILABLE'} - {detail}")
    if not ok:
        fail(f"Vision model is not reachable: {detail}")

    rows = []
    for item in items:
        image = (folder / item["file"]).read_bytes()
        for strategy, pipeline in pipelines.items():
            row = {"file": item["file"], "kind": item.get("kind", ""), "strategy": strategy,
                   "question": item["question"], "error": None}
            try:
                r = pipeline.ask(item["question"], image_bytes=image, answer_mode=item.get("answer_mode") or "5-mark")
                found, total = facts_found(r.answer, item["expected"])
                row.update(workflow=r.workflow, grounded=r.grounded, is_model_output=r.is_model_output,
                           facts_found=found, facts_total=total, all_facts=found == total,
                           total_ms=r.telemetry["total_ms"], model_calls=r.telemetry["model_calls"],
                           answer=r.answer, visual_description=r.visual_description,
                           sources="; ".join(s["article_title"] for s in r.sources), notices=r.notices)
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
                traceback.print_exc()
            rows.append(row)
            status = row["error"] or f"{row['workflow']:<22} facts {row['facts_found']}/{row['facts_total']}  {row['total_ms'] / 1000:.1f}s"
            print(f"  {item['file']:<20} {strategy:<10} {status}", flush=True)

    summary = summarise(rows, list(pipelines))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"image_benchmark_{stamp}.json"
    write_json(out, {"generated_at": datetime.now().isoformat(timespec="seconds"),
                     "vision_model": settings.VISION_MODEL, "llm_model": settings.LLM_SERVED_MODEL,
                     "folder": str(folder), "summary": summary, "rows": rows})
    with (RESULTS / f"image_grading_{stamp}.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "strategy", "question", "answer", "expected", "correct_0_3", "notes"])
        expected = {i["file"]: i["expected"] for i in items}
        for r in rows:
            writer.writerow([r["file"], r["strategy"], r["question"], r.get("answer") or r["error"],
                             expected[r["file"]], "", ""])
    print("\nSummary:")
    for s, v in summary.items():
        print(f"  {s:<10} all facts {v['all_facts_correct']}  fact recall {v['fact_recall']}  "
              f"median {v['median_latency_ms'] / 1000:.1f}s  {v['workflows']}")
    print("Saved:", out)


if __name__ == "__main__":
    main()
