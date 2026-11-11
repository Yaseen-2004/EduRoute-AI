"""
Calibrate the router's retrieval-probe thresholds (report §8.2).

The report warns that a similarity threshold must not be used as a universal
confidence score without calibration on a validation set. This script
measures the top-1 cosine of the retrieval probe for in-domain questions
(the AIML retrieval benchmark) and out-of-domain questions (other B.Tech
branches + general questions not covered by the current knowledge base),
then picks:

  in_domain_threshold  maximises Youden's J (TPR - FPR) between the groups.
  strong_threshold     25th percentile of in-domain questions whose top-1
                       chunk was actually relevant (a "typical good match").

It writes config/router_calibration.json with the distributions and AUC so
the choice is auditable. Re-run whenever the index or embedding model changes.
Caveat: the thresholds are fitted on these 90 questions; confirm on a
held-out set before reporting router accuracy.

Usage: venv\\Scripts\\python -m evaluation.calibrate_router
"""

from datetime import datetime, timezone

import numpy as np

from config import settings
from evaluation.evaluate_retrieval import is_relevant
from src.rag.retrieval import EduRouteRetriever
from src.utils.io import load_jsonl, write_json

BENCHMARKS = settings.BASE_DIR / "evaluation" / "benchmarks"
# Which questions count as in/out of domain depends on what the index covers.
CALIBRATION_SETS = {
    "v4": (["retrieval_aiml_v1.jsonl"], "router_out_of_domain_v1.jsonl"),
    "engineering_v1": (["retrieval_aiml_v1.jsonl", "retrieval_engineering_v1.jsonl"],
                       "router_out_of_domain_v2.jsonl"),
}


def auc(positive, negative):
    pos, neg = np.asarray(positive), np.asarray(negative)
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=settings.RAG_VERSION)
    args = parser.parse_args()

    in_files, out_file = CALIBRATION_SETS[args.version]
    retriever = EduRouteRetriever(version=args.version, mode=settings.RETRIEVAL_MODE)
    in_domain = [q for f in in_files for q in load_jsonl(BENCHMARKS / f)]
    out_domain = load_jsonl(BENCHMARKS / out_file)
    print(f"Index {args.version}: {len(in_domain)} in-domain vs {len(out_domain)} out-of-domain questions")

    in_scores, good_match_scores, out_scores, out_rows = [], [], [], []
    for item in in_domain:
        top = retriever.retrieve(item["question"], top_k=1)[0]
        in_scores.append(top["similarity_score"])
        if is_relevant(top, item):
            good_match_scores.append(top["similarity_score"])
    for item in out_domain:
        top = retriever.retrieve(item["question"], top_k=1)[0]
        out_scores.append(top["similarity_score"])
        out_rows.append({"id": item["id"], "group": item["group"], "question": item["question"],
                         "top_score": top["similarity_score"], "top_article": top["article_title"]})

    candidates = np.unique(np.round(np.concatenate([in_scores, out_scores]), 4))
    best_threshold, best_j = None, -1.0
    for t in candidates:
        tpr = np.mean(np.asarray(in_scores) >= t)
        fpr = np.mean(np.asarray(out_scores) >= t)
        if tpr - fpr > best_j:
            best_j, best_threshold = tpr - fpr, float(t)

    strong = float(np.percentile(good_match_scores, 25)) if good_match_scores else best_threshold
    in_arr, out_arr = np.asarray(in_scores), np.asarray(out_scores)

    calibration = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "index_version": args.version,
        "calibration_sets": {"in_domain": in_files, "out_of_domain": out_file},
        "retrieval_mode": settings.RETRIEVAL_MODE,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "in_domain_threshold": round(best_threshold, 4),
        "strong_threshold": round(max(strong, best_threshold), 4),
        "youden_j": round(best_j, 3),
        "auc": round(auc(in_scores, out_scores), 3),
        "in_domain_recall_at_threshold": round(float(np.mean(in_arr >= best_threshold)), 3),
        "out_of_domain_false_accept_rate": round(float(np.mean(out_arr >= best_threshold)), 3),
        "in_domain_scores": {"n": len(in_scores), "min": round(float(in_arr.min()), 4),
                             "median": round(float(np.median(in_arr)), 4), "max": round(float(in_arr.max()), 4)},
        "out_of_domain_scores": {"n": len(out_scores), "min": round(float(out_arr.min()), 4),
                                 "median": round(float(np.median(out_arr)), 4), "max": round(float(out_arr.max()), 4)},
        "out_of_domain_false_accepts": [r for r in out_rows if r["top_score"] >= best_threshold],
        "note": "Fitted on the calibration questions; validate on held-out questions before reporting.",
    }
    out_path = settings.router_calibration_path(args.version)
    write_json(out_path, calibration)

    print(f"AUC (in vs out of domain): {calibration['auc']}")
    print(f"in_domain_threshold: {calibration['in_domain_threshold']}  (Youden J {calibration['youden_j']})")
    print(f"strong_threshold:    {calibration['strong_threshold']}")
    print(f"In-domain kept: {calibration['in_domain_recall_at_threshold']:.0%}   "
          f"Out-of-domain falsely accepted: {calibration['out_of_domain_false_accept_rate']:.0%}")
    print("In-domain scores:", calibration["in_domain_scores"])
    print("Out-of-domain scores:", calibration["out_of_domain_scores"])
    for row in calibration["out_of_domain_false_accepts"]:
        print(f"  false accept {row['id']} {row['top_score']} {row['question']} -> {row['top_article']}")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
