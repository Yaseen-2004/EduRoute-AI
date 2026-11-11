"""
EduRoute AI - system benchmark across routing strategies (report §22).

Runs every strategy (direct, fixed_rag, always_verify, modality, adaptive) on
the same questions with the same model/prompt/temperature/retrieval settings
and records, per question:
    workflow, grounded, latency, model calls, input/output tokens,
    verification performed, unsupported claim rate, citations, failures.

Answer quality cannot be scored automatically without bias, so the script
also writes a grading CSV (one row per question x strategy, strategy names
hidden behind random codes) for human raters using the rubric in
evaluation/ANSWER_RUBRIC.md.

The question set is assembled from:
    retrieval_aiml_v1.jsonl        50 in-domain AIML questions
    router_out_of_domain_v1.jsonl  40 questions outside the current KB
    system_extra_v1.jsonl          10 comparison/numerical/chat/evidence items
Image and voice subsets are not included yet: they need a labelled image set
and recorded audio with reference transcripts (WER), see docs/PROGRESS.md.

Usage:
    venv\\Scripts\\python -m evaluation.run_system_benchmark --limit 10
    venv\\Scripts\\python -m evaluation.run_system_benchmark            # all 100
Requires a configured LLM for meaningful token/quality numbers; without one
the run is labelled "extractive" in the output and only routing/latency apply.
"""

import argparse
import csv
import random
import statistics
import traceback
from pathlib import Path
from datetime import datetime, timezone

from config import settings
from src.pipeline import EduRoutePipeline
from src.rag.retrieval import EduRouteRetriever
from src.router.router import STRATEGIES
from src.utils.io import append_jsonl, load_jsonl, write_json

BENCH = settings.BASE_DIR / "evaluation" / "benchmarks"
RESULTS = settings.BASE_DIR / "evaluation" / "results"
MODES = ("2-mark", "5-mark", "10-mark")


def keep_awake():
    """Ask Windows not to sleep while this process runs (no permanent setting change)."""
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        print("Keep-awake: on (Windows will not sleep while the benchmark runs)", flush=True)
    except Exception as error:  # not Windows
        print(f"Keep-awake unavailable: {error}", flush=True)


def load_questions():
    # In/out of domain depends on what the index covers.
    if settings.RAG_VERSION == "engineering_v1":
        in_files = ["retrieval_aiml_v1.jsonl", "retrieval_engineering_v1.jsonl"]
        out_file = "router_out_of_domain_v2.jsonl"
    else:
        in_files, out_file = ["retrieval_aiml_v1.jsonl"], "router_out_of_domain_v1.jsonl"
    questions = []
    in_items = [item for f in in_files for item in load_jsonl(BENCH / f)]
    for i, item in enumerate(in_items):
        questions.append({"id": item["id"], "question": item["question"], "set": "in_domain",
                          "answer_mode": MODES[i % 3]})
    for i, item in enumerate(load_jsonl(BENCH / out_file)):
        questions.append({"id": item["id"], "question": item["question"], "set": "out_of_domain",
                          "answer_mode": MODES[i % 3]})
    for item in load_jsonl(BENCH / "system_extra_v1.jsonl"):
        questions.append({**item, "set": item.get("set", "extra")})
    return questions


def summarise(rows):
    ok = [r for r in rows if not r["error"]]
    if not ok:
        return {"runs": len(rows), "failures": len(rows)}
    ucr = [r["unsupported_claim_rate"] for r in ok if r["unsupported_claim_rate"] is not None]
    workflows = {}
    for r in ok:
        workflows[r["workflow"]] = workflows.get(r["workflow"], 0) + 1
    return {
        "runs": len(rows),
        "failures": len(rows) - len(ok),
        "median_latency_ms": round(statistics.median(r["total_ms"] for r in ok), 1),
        "p90_latency_ms": round(sorted(r["total_ms"] for r in ok)[int(0.9 * (len(ok) - 1))], 1),
        "mean_model_calls": round(statistics.mean(r["model_calls"] for r in ok), 2),
        "mean_input_tokens": round(statistics.mean(r["input_tokens"] for r in ok), 1),
        "mean_output_tokens": round(statistics.mean(r["output_tokens"] for r in ok), 1),
        "grounded_rate": round(sum(r["grounded"] for r in ok) / len(ok), 3),
        "verified_rate": round(sum(r["verified"] for r in ok) / len(ok), 3),
        "revised_rate": round(sum(r["revised"] for r in ok) / len(ok), 3),
        "mean_unsupported_claim_rate": round(statistics.mean(ucr), 3) if ucr else None,
        "answers_with_citations": round(sum(r["cited"] > 0 for r in ok) / len(ok), 3),
        "answers_with_contradicted_claims": round(sum(r.get("contradicted_claims", 0) > 0 for r in ok) / len(ok), 3),
        "answers_with_unconfirmed_figures": round(sum(r.get("figure_warnings", 0) > 0 for r in ok) / len(ok), 3),
        "answers_with_arithmetic_issues": round(sum(r.get("arithmetic_issues", 0) > 0 for r in ok) / len(ok), 3),
        "workflow_distribution": workflows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    parser.add_argument("--limit", type=int, default=None, help="questions per set (for quick runs)")
    parser.add_argument("--sample", type=int, default=None,
                        help="random questions per set (seeded; for CPU-sized runs)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", default=None,
                        help="partial .jsonl from an interrupted run: finished answers are kept "
                             "and only the missing question x strategy pairs are run")
    parser.add_argument("--keep-awake", action="store_true",
                        help="stop Windows from sleeping while the run is going (long CPU runs)")
    args = parser.parse_args()
    if args.keep_awake:
        keep_awake()

    questions = load_questions()
    if args.limit:
        by_set = {}
        for q in questions:
            by_set.setdefault(q["set"], []).append(q)
        questions = [q for qs in by_set.values() for q in qs[:args.limit]]
    elif args.sample:
        sampler = random.Random(args.seed)
        by_set = {}
        for q in questions:
            by_set.setdefault(q["set"], []).append(q)
        questions = [q for qs in by_set.values()
                     for q in sampler.sample(qs, min(args.sample, len(qs)))]

    retriever = EduRouteRetriever(version=settings.RAG_VERSION, mode=settings.RETRIEVAL_MODE)
    pipelines = {s: EduRoutePipeline(strategy=s, retriever=retriever, log=False) for s in args.strategies}
    llm_ok, llm_detail = next(iter(pipelines.values())).llm.health_check()
    print(f"LLM: {'available' if llm_ok else 'UNAVAILABLE'} - {llm_detail}")
    print(f"{len(questions)} questions x {len(pipelines)} strategies")

    rows, answers = [], []
    RESULTS.mkdir(parents=True, exist_ok=True)
    # Written as it goes so an interrupted run (Colab disconnect, laptop restart) keeps its work.
    partial = RESULTS / f"system_benchmark_partial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    done = set()
    if args.resume:
        partial = Path(args.resume)
        wanted = {q["id"] for q in questions}
        for old in load_jsonl(partial):
            if old["id"] not in wanted or old["strategy"] not in pipelines or old.get("error"):
                continue  # failed answers are retried
            done.add((old["id"], old["strategy"]))
            rows.append({k: v for k, v in old.items() if k not in ("answer", "question", "sources", "claims")})
            if old.get("answer") is not None:
                answers.append({"id": old["id"], "question": old.get("question", ""),
                                "answer_mode": old["answer_mode"], "strategy": old["strategy"],
                                "answer": old["answer"], "sources": old.get("sources", "")})
        print(f"Resuming {partial.name}: {len(done)} answers already done", flush=True)
    for n, q in enumerate(questions, start=1):
        for strategy, pipeline in pipelines.items():
            if (q["id"], strategy) in done:
                continue
            record = {"id": q["id"], "set": q["set"], "strategy": strategy, "answer_mode": q["answer_mode"],
                      "error": None}
            claims = []
            try:
                r = pipeline.ask(q["question"], answer_mode=q["answer_mode"])
                v = r.verification or {}
                record.update(
                    workflow=r.workflow, grounded=r.grounded, is_model_output=r.is_model_output,
                    total_ms=r.telemetry["total_ms"], model_calls=r.telemetry["model_calls"],
                    input_tokens=r.telemetry["input_tokens"], output_tokens=r.telemetry["output_tokens"],
                    verified=bool(v), revised=bool(v.get("revised")),
                    unsupported_claim_rate=v.get("unsupported_claim_rate"),
                    contradicted_claims=sum(c.get("nli") == "contradiction" for c in v.get("claims", [])),
                    figure_warnings=sum(bool(c.get("missing_numbers")) for c in v.get("claims", [])),
                    arithmetic_issues=len(r.arithmetic_issues),
                    cited=len(r.cited_sources), invalid_citations=len(r.invalid_citations),
                    stages_ms=r.telemetry["stages_ms"],
                )
                # Per-claim verification results, for human claim labelling
                # (evaluation/claim_labels.py) and verifier calibration.
                claims = [{k: c.get(k) for k in ("text", "label", "support_score", "semantic_similarity",
                                                  "lexical_coverage", "nli", "missing_numbers", "best_source")}
                          | {"best_evidence": (c.get("best_evidence") or "")[:400]}
                          for c in v.get("claims", [])]
                answers.append({"id": q["id"], "question": q["question"], "answer_mode": q["answer_mode"],
                                "strategy": strategy, "answer": r.answer,
                                "sources": "; ".join(s["article_title"] for s in r.sources)})
            except Exception as error:  # report failures, never drop them (§22.3)
                record["error"] = f"{type(error).__name__}: {error}"
                traceback.print_exc()
            rows.append(record)
            mine = answers[-1] if answers and answers[-1]["id"] == q["id"] \
                and answers[-1]["strategy"] == strategy and not record["error"] else None
            append_jsonl(partial, {**record, "question": q["question"],
                                   "answer": mine["answer"] if mine else None,
                                   "sources": mine["sources"] if mine else "", "claims": claims})
            status = record["error"] or (
                f"{record['workflow']} {record['total_ms'] / 1000:.0f}s "
                f"{'model' if record['is_model_output'] else 'EXTRACTIVE FALLBACK'}")
            print(f"    {datetime.now():%H:%M} {q['id']:<5} {strategy:<14} {status}", flush=True)
        if n % 10 == 0 or args.sample:
            print(f"  {n}/{len(questions)} questions done", flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "llm_available": llm_ok,
        "llm": llm_detail,
        "config": {"rag_version": settings.RAG_VERSION, "retrieval_mode": settings.RETRIEVAL_MODE,
                   "temperature": settings.TEMPERATURE, "top_k": settings.TOP_K,
                   "llm_backend": settings.LLM_BACKEND, "llm_model": settings.LLM_SERVED_MODEL,
                   "nli_model": settings.NLI_MODEL or None},
        "questions": len(questions),
        "selection": ({"limit": args.limit} if args.limit else
                      {"sample_per_set": args.sample, "seed": args.seed} if args.sample else "all"),
        "question_ids": [q["id"] for q in questions],
        "summary": {
            s: {
                "all": summarise([r for r in rows if r["strategy"] == s]),
                **{qs: summarise([r for r in rows if r["strategy"] == s and r["set"] == qs])
                   for qs in sorted({q["set"] for q in questions})},
            }
            for s in pipelines
        },
        "rows": rows,
    }
    out = RESULTS / f"system_benchmark_{stamp}.json"
    write_json(out, report)

    # Blind grading sheet: strategies hidden behind shuffled codes.
    rng = random.Random(args.seed)
    codes = {s: f"S{i}" for i, s in enumerate(rng.sample(list(pipelines), len(pipelines)), start=1)}
    rng.shuffle(answers)
    grading = RESULTS / f"grading_sheet_{stamp}.csv"
    with grading.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "system_code", "answer_mode", "question", "answer", "sources",
                         "correctness_0_3", "completeness_0_3", "relevance_0_3", "clarity_0_3", "notes"])
        for a in answers:
            writer.writerow([a["id"], codes[a["strategy"]], a["answer_mode"], a["question"],
                             a["answer"], a["sources"], "", "", "", "", ""])
    write_json(RESULTS / f"grading_key_{stamp}.json", codes)

    print(f"\n{'strategy':<15}{'median ms':>10}{'calls':>7}{'in tok':>8}{'out tok':>8}"
          f"{'grounded':>9}{'verified':>9}{'UCR':>7}  workflows")
    for s, summ in report["summary"].items():
        a = summ["all"]
        if "median_latency_ms" not in a:
            print(f"{s:<15} all runs failed")
            continue
        ucr = "-" if a["mean_unsupported_claim_rate"] is None else a["mean_unsupported_claim_rate"]
        print(f"{s:<15}{a['median_latency_ms']:>10}{a['mean_model_calls']:>7}{a['mean_input_tokens']:>8}"
              f"{a['mean_output_tokens']:>8}{a['grounded_rate']:>9}{a['verified_rate']:>9}{ucr:>7}  "
              f"{a['workflow_distribution']}")
    print("\nSaved:", out)
    print("Grading sheet (blind):", grading)


if __name__ == "__main__":
    main()
