"""
With vs without the four new features, on the real model (CPU or GPU).

Same questions, same order, two configurations:
  baseline   agentic RAG, context engineering, memory and answer cache OFF
  enhanced   all four ON

Question kinds (from the benchmark files, fixed by seed):
  single      in-domain questions, 2- and 5-mark
  comparison  "Compare A and B" for same-branch benchmark topics
  follow_up   a question, then "What are its advantages?" (scored: does the
              follow-up's evidence come from the original topic?)
  repeat      one single question asked again (cache demonstration)

Per answer it records time, time to first word, prompt/output tokens,
citations, verification and, where it applies, whether the evidence covered
the topic. Resumable: re-run to continue. ~1.5 h on the laptop CPU.

Run: venv\\Scripts\\python -m evaluation.run_feature_ablation [--per-kind 3]
"""

import argparse
import ctypes
import json
import random
import statistics
from collections import defaultdict
from datetime import datetime

from config import settings
from evaluation.evaluate_retrieval import is_relevant
from src.utils.io import append_jsonl, load_jsonl

BENCH = settings.BASE_DIR / "evaluation" / "benchmarks"
RESULTS = settings.BASE_DIR / "evaluation" / "results"
PARTIAL = RESULTS / "feature_ablation_partial.jsonl"
FLAGS = ("AGENTIC_RAG", "CONTEXT_ENGINEERING", "CONVERSATION_MEMORY", "ANSWER_CACHE")


def keep_awake():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)
    except Exception:
        pass


def build_cases(per_kind, seed=7):
    rng = random.Random(seed)
    items = load_jsonl(BENCH / "retrieval_engineering_v1.jsonl") + load_jsonl(BENCH / "retrieval_aiml_v1.jsonl")
    picked = rng.sample(items, per_kind * 3)
    cases = []
    for i, item in enumerate(picked[:per_kind]):
        cases.append({"id": f"S{i}", "kind": "single", "turns": [item["question"]],
                      "mode": ("2-mark", "5-mark")[i % 2], "items": [item]})
    by_branch = defaultdict(list)
    for item in items:
        by_branch[item.get("subject_id", "")[:3]].append(item)
    branches = [b for b, group in sorted(by_branch.items()) if len(group) >= 2]
    for i, branch in enumerate(rng.sample(branches, per_kind)):
        a, b = rng.sample(by_branch[branch], 2)
        cases.append({"id": f"C{i}", "kind": "comparison", "mode": "5-mark", "items": [a, b],
                      "turns": [f"Compare {a['relevant_articles'][0]} and {b['relevant_articles'][0]}."]})
    for i, item in enumerate(picked[per_kind:per_kind * 2]):
        cases.append({"id": f"F{i}", "kind": "follow_up", "mode": "2-mark", "items": [item],
                      "turns": [item["question"], "What are its advantages?"]})
    first = cases[0]
    cases.append({"id": "R0", "kind": "repeat", "mode": first["mode"], "items": first["items"],
                  "turns": [first["turns"][0]]})
    return cases


def run_config(name, cases, done):
    for flag in FLAGS:
        setattr(settings, flag, name == "enhanced")
    from src.pipeline import EduRoutePipeline
    pipeline = EduRoutePipeline(strategy="adaptive", log=False)
    for case in cases:
        if (name, case["id"]) in done:
            continue
        history, response = [], None
        for turn in case["turns"]:
            response = pipeline.ask(turn, answer_mode=case["mode"], history=history or None)
            history.append({"question": turn, "answer": response.answer})
        r, t = response, response.telemetry
        sources = [{"article_title": s["article_title"], "text": s["text"]} for s in r.sources]
        covered = [any(is_relevant(s, item) for s in sources) for item in case["items"]]
        verification = r.verification or {}
        row = {
            "config": name, "id": case["id"], "kind": case["kind"], "mode": case["mode"],
            "workflow": r.workflow, "error": r.error, "total_s": round(t["total_ms"] / 1000, 1),
            "first_token_s": round(t["stages_ms"].get("first_token", 0) / 1000, 1) or None,
            "input_tokens": t["input_tokens"], "output_tokens": t["output_tokens"],
            "model_calls": t["model_calls"], "cited": len(r.cited_sources), "sources": len(r.sources),
            "verified": r.verification is not None, "revised": bool(verification.get("revised")),
            "unsupported_claim_rate": verification.get("unsupported_claim_rate"),
            "evidence_covers_topic": all(covered) if covered else None,
            "cache_hit": t.get("cache_hit", False),
            "context_saved_pct": (t.get("context") or {}).get("saved_pct"),
            "agent_steps": len(t.get("agent_trace") or []),
        }
        append_jsonl(PARTIAL, row)
        done.add((name, case["id"]))
        print(f"  {name:<9} {case['id']:<3} {case['kind']:<10} {row['total_s']:>6}s "
              f"first {row['first_token_s']}s  covers={row['evidence_covers_topic']}  cache={row['cache_hit']}",
              flush=True)


def summarise(rows):
    out = {}
    for config in ("baseline", "enhanced"):
        mine = [r for r in rows if r["config"] == config and not r["error"]]
        by_kind = defaultdict(list)
        for r in mine:
            by_kind[r["kind"]].append(r)
        out[config] = {}
        for kind, group in sorted(by_kind.items()):
            def med(key):
                values = [r[key] for r in group if r[key] is not None]
                return round(statistics.median(values), 1) if values else None
            out[config][kind] = {
                "answers": len(group), "median_total_s": med("total_s"), "median_first_token_s": med("first_token_s"),
                "median_input_tokens": med("input_tokens"), "cited_rate": round(sum(r["cited"] > 0 for r in group) / len(group), 2),
                "evidence_covers_topic": round(sum(bool(r["evidence_covers_topic"]) for r in group) / len(group), 2),
                "mean_unsupported_claim_rate": (round(statistics.mean(v), 3) if (v := [
                    r["unsupported_claim_rate"] for r in group if r["unsupported_claim_rate"] is not None]) else None),
                "cache_hits": sum(r["cache_hit"] for r in group),
            }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-kind", type=int, default=3)
    args = parser.parse_args()
    keep_awake()
    cases = build_cases(args.per_kind)
    RESULTS.mkdir(parents=True, exist_ok=True)
    PARTIAL.touch()
    rows = load_jsonl(PARTIAL)
    done = {(r["config"], r["id"]) for r in rows}
    print(f"{len(cases)} cases x 2 configurations; {len(done)} answers already done", flush=True)
    for config in ("baseline", "enhanced"):
        run_config(config, cases, done)
    rows = load_jsonl(PARTIAL)
    report = {"generated_at": datetime.now().isoformat(timespec="seconds"), "llm": settings.LLM_SERVED_MODEL,
              "cases": [{k: c[k] for k in ("id", "kind", "mode", "turns")} for c in cases],
              "summary": summarise(rows), "rows": rows}
    out = RESULTS / f"feature_ablation_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=1))
    print("Saved:", out)


if __name__ == "__main__":
    main()
