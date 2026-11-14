"""
Same questions, two text models: is a smaller model good enough for short answers?

For a seeded sample of in-domain benchmark questions it asks each question
through the full pipeline with each model and records time, output tokens,
whether the answer contains the benchmark's expected key terms (a lexical
correctness proxy), citations and verification results.

Run: venv\\Scripts\\python -m evaluation.compare_models --models qwen2.5:3b qwen2.5:1.5b --mode 2-mark --n 12
"""

import argparse
import json
import random
import statistics
from datetime import datetime

from config import settings
from src.llm.backends import build_backend
from src.utils.io import load_jsonl

BENCH = settings.BASE_DIR / "evaluation" / "benchmarks"
RESULTS = settings.BASE_DIR / "evaluation" / "results"


def key_terms_present(answer, item):
    text = answer.lower()
    return any(term.lower() in text for term in item.get("evidence_terms", []))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen2.5:3b", "qwen2.5:1.5b"])
    parser.add_argument("--mode", default="2-mark")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    items = load_jsonl(BENCH / "retrieval_engineering_v1.jsonl") + load_jsonl(BENCH / "retrieval_aiml_v1.jsonl")
    sample = random.Random(args.seed).sample(items, args.n)
    from src.pipeline import EduRoutePipeline
    from src.rag.retrieval import EduRouteRetriever
    retriever = EduRouteRetriever(version=settings.RAG_VERSION, mode=settings.RETRIEVAL_MODE)

    rows = []
    for model in args.models:
        settings.LLM_SERVED_MODEL = model
        pipeline = EduRoutePipeline(strategy="adaptive", retriever=retriever, log=False,
                                    llm_backend=build_backend(settings, "text"))
        pipeline.answer_cache = None                     # every answer generated, never cached
        for item in sample:
            r = pipeline.ask(item["question"], answer_mode=args.mode)
            v = r.verification or {}
            row = {"model": model, "id": item["id"], "workflow": r.workflow, "error": r.error,
                   "seconds": round(r.telemetry["total_ms"] / 1000, 1),
                   "output_tokens": r.telemetry["output_tokens"], "words": len(r.answer.split()),
                   "key_terms": key_terms_present(r.answer, item), "cited": len(r.cited_sources),
                   "verified": r.verification is not None, "unsupported_rate": v.get("unsupported_claim_rate"),
                   "revised": bool(v.get("revised")), "answer": r.answer}
            rows.append(row)
            print(f"{model:<14} {item['id']:<4} {row['workflow']:<11} {row['seconds']:>6}s "
                  f"terms={row['key_terms']} cited={row['cited']} ucr={row['unsupported_rate']}", flush=True)

    summary = {}
    for model in args.models:
        g = [r for r in rows if r["model"] == model and not r["error"]]
        ucr = [r["unsupported_rate"] for r in g if r["unsupported_rate"] is not None]
        grounded = [r for r in g if r["workflow"] == "rag_llm"]
        summary[model] = {
            "answers": len(g), "median_seconds": statistics.median(r["seconds"] for r in g),
            "mean_seconds": round(statistics.mean(r["seconds"] for r in g), 1),
            "key_terms_rate": round(sum(r["key_terms"] for r in g) / len(g), 2),
            "grounded_citing_rate": round(sum(r["cited"] > 0 for r in grounded) / len(grounded), 2) if grounded else None,
            "mean_unsupported_rate": round(statistics.mean(ucr), 3) if ucr else None,
            "revised": sum(r["revised"] for r in g),
        }
    out = RESULTS / f"model_comparison_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"mode": args.mode, "summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print("Saved:", out)


if __name__ == "__main__":
    main()
