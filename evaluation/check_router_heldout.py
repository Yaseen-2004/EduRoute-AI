"""
Held-out check of the router's in-domain threshold (report §8.2).

The threshold in config/router_calibration_<index>.json was fitted on the
benchmark questions, so its 98% / 0% figures are optimistic. This script
applies it, unchanged, to router_heldout_v1.jsonl: 30 new in-domain questions
(one or two per branch), 10 easy and 16 hard out-of-domain questions
(history, law, economics, medicine, arts), and 4 borderline ones reported
separately.

    venv\\Scripts\\python -m evaluation.check_router_heldout [--version engineering_v1]

No language model is needed; it runs on a laptop CPU in a few minutes.
"""

import argparse
from datetime import datetime

from config import settings
from src.rag.retrieval import EduRouteRetriever
from src.router.router import load_calibration
from src.utils.io import load_jsonl, write_json

BENCH = settings.BASE_DIR / "evaluation" / "benchmarks" / "router_heldout_v1.jsonl"
RESULTS = settings.BASE_DIR / "evaluation" / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=settings.RAG_VERSION)
    args = parser.parse_args()

    calibration = load_calibration(args.version)
    threshold = calibration["in_domain_threshold"]
    retriever = EduRouteRetriever(version=args.version, verbose=False)
    items = load_jsonl(BENCH)

    rows = []
    for item in items:
        top = retriever.retrieve(item["question"], top_k=1)
        score = top[0]["similarity_score"] if top else 0.0
        rows.append({**item, "top_score": round(score, 4), "accepted": score >= threshold,
                     "top_article": top[0]["article_title"] if top else None})

    def share(subset, accepted):
        return round(sum(r["accepted"] == accepted for r in subset) / len(subset), 3) if subset else None

    inside = [r for r in rows if r["expected"] == "in"]
    easy = [r for r in rows if r.get("difficulty") == "easy"]
    hard = [r for r in rows if r.get("difficulty") == "hard"]
    border = [r for r in rows if r["expected"] == "either"]
    outside = easy + hard
    correct = sum(r["accepted"] for r in inside) + sum(not r["accepted"] for r in outside)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "index": args.version,
        "threshold": threshold,
        "calibration_source": calibration.get("source"),
        "in_domain_kept": share(inside, True),
        "out_of_domain_rejected_easy": share(easy, False),
        "out_of_domain_rejected_hard": share(hard, False),
        "accuracy_excluding_borderline": round(correct / (len(inside) + len(outside)), 3),
        "missed_in_domain": [(r["id"], r["question"], r["top_score"], r["top_article"])
                             for r in inside if not r["accepted"]],
        "false_accepts": [(r["id"], r["question"], r["top_score"], r["top_article"])
                          for r in outside if r["accepted"]],
        "borderline": [(r["id"], r["question"], r["top_score"], r["accepted"], r["top_article"])
                       for r in border],
        "rows": rows,
    }
    out = RESULTS / f"router_heldout_{args.version}_{datetime.now():%Y%m%d_%H%M%S}.json"
    write_json(out, report)

    print(f"Index {args.version}, threshold {threshold} (not refitted)")
    print(f"In-domain kept:           {report['in_domain_kept']}  ({len(inside)} questions)")
    print(f"Easy out-of-domain reject: {report['out_of_domain_rejected_easy']}  ({len(easy)})")
    print(f"Hard out-of-domain reject: {report['out_of_domain_rejected_hard']}  ({len(hard)})")
    print(f"Accuracy (excl. borderline): {report['accuracy_excluding_borderline']}")
    for label, key in (("Missed in-domain", "missed_in_domain"), ("False accepts", "false_accepts"),
                       ("Borderline (reported, not scored)", "borderline")):
        print(f"\n{label}:")
        for r in report[key]:
            print("  ", r)
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
