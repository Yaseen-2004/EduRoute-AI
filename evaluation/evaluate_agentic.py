"""
Measures the agentic RAG, memory, context-engineering and cache features against
the single-shot baseline, on the real engineering index. No language model is
needed, so it runs in minutes on a CPU.

1. follow-up    For each in-domain benchmark question, three follow-ups that
                only make sense after it ("What are its advantages?", ...).
                Hit@5 = a relevant passage of the ORIGINAL topic in the top 5.
                Baseline: the follow-up searched alone. Memory: contextualised.
2. comparison   "Compare A and B" for pairs of benchmark topics from the same
                branch. Both-hit@5 = relevant passages for A AND for B in the
                final evidence. Baseline: single-shot top 5. Agent: planned
                sub-queries + coverage check.
3. context      Prompt tokens of the sources for every in-domain question, per
                answer mode, before and after context engineering, and whether
                the top relevant passage survives in the prompt.
4. cache        Exact repeats and reworded repeats must hit; every pair of
                DIFFERENT benchmark questions must never hit (false hits).

Run: venv\\Scripts\\python -m evaluation.evaluate_agentic
"""

import itertools
import json
import random
import statistics
from datetime import datetime

from config import settings
from evaluation.evaluate_retrieval import is_relevant
from src.agent import memory
from src.agent.agentic_rag import AgenticRetriever
from src.inference.answer_cache import SemanticAnswerCache
from src.rag.context import ContextBuilder
from src.rag.retrieval import EduRouteRetriever
from src.utils.io import load_jsonl

BENCH = settings.BASE_DIR / "evaluation" / "benchmarks"
RESULTS = settings.BASE_DIR / "evaluation" / "results"
FOLLOW_UPS = ["What are its advantages?", "Give an example of it.", "What are its main limitations?"]


def hit(passages, item, k=5):
    return any(is_relevant(p, item) for p in passages[:k])


def main():
    settings.CONVERSATION_MEMORY = True
    random.seed(7)
    retriever = EduRouteRetriever(version=settings.RAG_VERSION, mode=settings.RETRIEVAL_MODE)
    calibration = json.loads(settings.router_calibration_path(settings.RAG_VERSION).read_text())
    threshold = calibration["in_domain_threshold"]
    items = load_jsonl(BENCH / "retrieval_engineering_v1.jsonl") + load_jsonl(BENCH / "retrieval_aiml_v1.jsonl")
    report = {"generated_at": datetime.now().isoformat(timespec="seconds"), "index": settings.RAG_VERSION,
              "questions": len(items)}

    # 1. follow-ups
    base_hits = mem_hits = total = 0
    for item in items:
        history = [{"question": item["question"], "answer": ""}]
        for follow in FOLLOW_UPS:
            total += 1
            base_hits += hit(retriever.retrieve(follow, top_k=5), item)
            query, _topic = memory.contextualize(follow, history)
            mem_hits += hit(retriever.retrieve(query, top_k=5), item)
    report["follow_up"] = {"cases": total, "baseline_hit@5": round(base_hits / total, 3),
                           "memory_hit@5": round(mem_hits / total, 3)}
    print("follow-up:", report["follow_up"], flush=True)

    # 2. comparisons
    by_branch = {}
    for item in items:
        by_branch.setdefault(item.get("subject_id", "")[:3], []).append(item)
    pairs = [p for group in by_branch.values() for p in itertools.combinations(group, 2)]
    pairs = random.sample(pairs, min(40, len(pairs)))
    base_both = agent_both = 0
    agent_steps = []
    for a, b in pairs:
        question = f"Compare {a['relevant_articles'][0]} and {b['relevant_articles'][0]}."
        single = retriever.retrieve(question, top_k=5)
        base_both += hit(single, a) and hit(single, b)
        agent = AgenticRetriever(lambda q, k: retriever.retrieve(q, top_k=k), threshold)
        run = agent.run(question, top_k=5, probe=single)
        agent_both += hit(run.passages, a, k=len(run.passages)) and hit(run.passages, b, k=len(run.passages))
        agent_steps.append(len(run.trace))
    report["comparison"] = {"pairs": len(pairs), "baseline_both_hit@5": round(base_both / len(pairs), 3),
                            "agent_both_hit": round(agent_both / len(pairs), 3),
                            "agent_mean_steps": round(statistics.mean(agent_steps), 2)}
    print("comparison:", report["comparison"], flush=True)

    # 3. context engineering
    builder = ContextBuilder(retriever.model)
    context = {}
    for mode in ("2-mark", "5-mark", "10-mark"):
        before, after, kept_relevant, had_relevant = [], [], 0, 0
        for item in items:
            passages = retriever.retrieve(item["question"], top_k=5)
            selected, stats = builder.build(item["question"], passages, mode)
            before.append(stats.tokens_before)
            after.append(stats.tokens_after)
            first = next((p for p in passages if is_relevant(p, item)), None)
            if first is not None:
                had_relevant += 1
                kept_relevant += any(p.get("chunk_id") == first.get("chunk_id") for p in selected)
        context[mode] = {"tokens_before_median": statistics.median(before),
                         "tokens_after_median": statistics.median(after),
                         "saved_pct": round(100 * (1 - sum(after) / sum(before)), 1),
                         "top_relevant_kept": round(kept_relevant / had_relevant, 3) if had_relevant else None,
                         "prompt_seconds_saved_at_20_tok_s": round(
                             (statistics.median(before) - statistics.median(after)) / 20, 1)}
    report["context"] = context
    print("context:", context, flush=True)

    # 4. cache
    cache = SemanticAnswerCache(retriever.model, max_entries=1000)
    sig = SemanticAnswerCache.signature("5-mark", None, "adaptive")
    for item in items:
        cache.put(item["question"], sig, {"id": item["id"]})
    exact = sum((cache.get(i["question"].upper(), sig) or {}).get("id") == i["id"] for i in items)
    reworded = 0
    for item in items:
        variant = item["question"].rstrip("?.") + ", please?"
        reworded += (cache.get(variant, sig) or {}).get("id") == item["id"]
    false_hits = 0
    cache2 = SemanticAnswerCache(retriever.model, max_entries=1000)
    for i, item in enumerate(items):
        found = cache2.get(item["question"], sig)
        false_hits += found is not None                   # any hit is wrong: each question is new
        cache2.put(item["question"], sig, {"id": item["id"]})
    report["cache"] = {"exact_repeat_hits": f"{exact}/{len(items)}",
                       "reworded_repeat_hits": f"{reworded}/{len(items)}",
                       "false_hits_between_different_questions": false_hits}
    print("cache:", report["cache"], flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"agentic_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved:", out)


if __name__ == "__main__":
    main()
