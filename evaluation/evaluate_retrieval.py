"""
EduRoute AI - retrieval evaluation (report §19.1, §22.2).

Relevance protocol (fixed, version-independent):
  A retrieved chunk is RELEVANT to a question if
    (a) its article title is in the question's `relevant_articles`, AND
    (b) its text contains at least one of the question's `evidence_terms`
        (compared after removing case, spaces and punctuation, so V3's
        tokenizer-decoded "self - attention" matches "self-attention").
  Because labels are defined on articles + evidence terms rather than chunk
  IDs, V3 and V4 are scored with identical labels even though their chunk
  boundaries differ.

Metrics
  Success@K  fraction of questions with >=1 relevant chunk in the top K.
             (This is the report's "Recall@K" definition, made explicit.)
  Recall@K   mean fraction of all relevant chunks in the index that appear in
             the top K (strict; low when an article has many relevant chunks).
  MRR@10     mean reciprocal rank of the first relevant chunk (0 if none in 10).
  Article@1  top-1 chunk comes from a relevant article.

Questions whose relevant articles do not exist in an index are still scored
(a miss is a real coverage failure), but results are also reported on the
subset answerable by every compared index ("common" subset), so V3 vs V4
is not only a coverage comparison.

Usage (project root):
    venv\\Scripts\\python -m evaluation.evaluate_retrieval
    venv\\Scripts\\python -m evaluation.evaluate_retrieval --configs v3:dense v4:hybrid
"""

import argparse
import re
import statistics
import time
from datetime import datetime, timezone

from config.settings import BASE_DIR
from src.pipeline import scoped_retrieve
from src.rag.retrieval import EduRouteRetriever
from src.router.router import load_calibration
from src.utils.catalogue_loader import load_catalogue
from src.utils.io import load_jsonl, write_json

BENCHMARK = BASE_DIR / "evaluation" / "benchmarks" / "retrieval_aiml_v1.jsonl"
RESULTS_DIR = BASE_DIR / "evaluation" / "results"

DEFAULT_CONFIGS = ["v3:dense", "v3:hybrid", "v4:dense", "v4:hybrid"]
KS = (1, 3, 5, 10)


def squash(text):
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def is_relevant(chunk, item):
    # Titles compared without punctuation: "Nyquist-Shannon" == "Nyquist–Shannon".
    if squash(chunk["article_title"]) not in {squash(t) for t in item["relevant_articles"]}:
        return False
    text = squash(chunk["text"])
    return any(squash(term) in text for term in item["evidence_terms"])


def parse_config(spec):
    parts = spec.split(":")
    version, mode = parts[0], parts[1] if len(parts) > 1 else "dense"
    rerank = len(parts) > 2 and parts[2] == "rerank"
    return version, mode, rerank


def subject_branches():
    catalogue = load_catalogue()
    return {s["id"]: code for code, subjects in catalogue["groups"].items() for s in subjects}


def evaluate(retriever, benchmark, mode, rerank, subject_filter=None):
    """subject_filter: None, "strict" (subject only) or "widen" (subject ->
    branch -> everything below the in-domain threshold, as the app does)."""
    branch_of = subject_branches() if subject_filter == "widen" else {}
    threshold = load_calibration(retriever.version)["in_domain_threshold"]
    relevant_totals = {}
    for item in benchmark:
        relevant_totals[item["id"]] = sum(is_relevant(m, item) for m in retriever.metadata)

    per_question = []
    for item in benchmark:
        started = time.perf_counter()
        # subject_filter simulates a student who selected the subject in the app.
        filters = {"subject_id": item["subject_id"]} if subject_filter and item.get("subject_id") else None
        fallbacks = []
        if subject_filter == "widen" and filters:
            branch = branch_of.get(item["subject_id"])
            fallbacks = ([{"discipline": branch}] if branch else []) + [None]
        results, scope = scoped_retrieve(retriever, item["question"], filters, fallbacks, threshold,
                                         top_k=max(KS), mode=mode, rerank=rerank)
        latency = time.perf_counter() - started

        flags = [is_relevant(r, item) for r in results]
        first = next((i + 1 for i, f in enumerate(flags) if f), None)
        total = relevant_totals[item["id"]]
        per_question.append({
            "id": item["id"],
            "question": item["question"],
            "answerable": total > 0,
            "relevant_in_index": total,
            "first_relevant_rank": first,
            "top1_article": results[0]["article_title"] if results else None,
            "top1_article_relevant": bool(results) and squash(results[0]["article_title"]) in
                {squash(t) for t in item["relevant_articles"]},
            "flags": flags,
            "latency_ms": round(latency * 1000, 1),
            "scope": scope,
        })
    return per_question


def summarize(rows):
    if not rows:
        return {}
    n = len(rows)
    summary = {"questions": n}
    for k in KS:
        summary[f"success@{k}"] = round(sum(any(r["flags"][:k]) for r in rows) / n, 3)
    summary["recall@5"] = round(statistics.mean(
        (sum(r["flags"][:5]) / r["relevant_in_index"]) if r["relevant_in_index"] else 0.0
        for r in rows
    ), 3)
    summary["mrr@10"] = round(statistics.mean(
        1 / r["first_relevant_rank"] if r["first_relevant_rank"] else 0.0 for r in rows
    ), 3)
    summary["article@1"] = round(sum(r["top1_article_relevant"] for r in rows) / n, 3)
    summary["median_latency_ms"] = round(statistics.median(r["latency_ms"] for r in rows), 1)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                        help="version:mode[:rerank], e.g. v4:hybrid:rerank or engineering_v1:dense")
    parser.add_argument("--subject-filter", nargs="?", const="strict", choices=("strict", "widen"),
                        help="filter retrieval to each question's subject (as when a student selects "
                             "it); 'widen' also applies the app's subject -> branch -> all fallback")
    parser.add_argument("--benchmark", default=BENCHMARK.name,
                        help="file in evaluation/benchmarks/")
    args = parser.parse_args()

    benchmark_file = BENCHMARK.parent / args.benchmark
    benchmark = load_jsonl(benchmark_file)
    print(f"Benchmark: {benchmark_file.name} ({len(benchmark)} questions)")

    shared_model = None
    runs = {}
    for spec in args.configs:
        version, mode, rerank = parse_config(spec)
        retriever = EduRouteRetriever(version=version, mode=mode, rerank=rerank,
                                      embedding_model=shared_model)
        shared_model = retriever.model
        retriever.retrieve("warm up", top_k=1)  # exclude one-time BM25/model warm-up from latency
        print(f"Running {spec} ...")
        label = spec + {None: "", "strict": ":subject", "widen": ":subject+widen"}[args.subject_filter]
        runs[label] = evaluate(retriever, benchmark, mode, rerank, args.subject_filter)

    common_ids = set.intersection(*(
        {r["id"] for r in rows if r["answerable"]} for rows in runs.values()
    ))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": str(benchmark_file.relative_to(BASE_DIR)),
        "relevance_protocol": "relevant article AND contains an evidence term",
        "label_status": "Draft labels; require review by the project team before final reporting.",
        "common_subset_size": len(common_ids),
        "configs": {},
    }

    header = f"{'config':<36}{'subset':<8}{'n':>4}{'S@1':>7}{'S@3':>7}{'S@5':>7}{'R@5':>7}{'MRR@10':>8}{'Art@1':>7}{'ms':>8}"
    print("\n" + header + "\n" + "-" * len(header))
    for spec, rows in runs.items():
        overall = summarize(rows)
        common = summarize([r for r in rows if r["id"] in common_ids])
        unanswerable = [r["id"] for r in rows if not r["answerable"]]
        report["configs"][spec] = {
            "overall": overall,
            "common_subset": common,
            "not_answerable_in_index": unanswerable,
            "misses_at_5": [r["id"] for r in rows if not any(r["flags"][:5])],
            "per_question": rows,
        }
        for label, s in (("all", overall), ("common", common)):
            print(f"{spec:<36}{label:<8}{s['questions']:>4}{s['success@1']:>7}{s['success@3']:>7}"
                  f"{s['success@5']:>7}{s['recall@5']:>7}{s['mrr@10']:>8}{s['article@1']:>7}"
                  f"{s['median_latency_ms']:>8}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"retrieval_eval_{benchmark_file.stem}_{stamp}.json"
    write_json(out, report)
    print("\nMisses@5 per config:")
    for spec, data in report["configs"].items():
        print(f"  {spec}: {data['misses_at_5']}  (not in index: {data['not_answerable_in_index']})")
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
