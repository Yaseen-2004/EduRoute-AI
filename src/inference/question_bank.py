"""
Question bank: answer the likely exam questions ahead of time, so students get
them instantly (on this CPU a live answer takes 1-3 minutes).

For every subject in the catalogue it asks "Explain <topic>." (5-mark by
default) through the normal pipeline, with the subject selected exactly as a
student would select it, and pins the answer in the saved answer cache under
the three ways a student can ask it: with that subject selected, with its
branch selected, and with nothing selected. The cache's semantic tier then
matches rewordings ("What is <topic>?", "Can you explain <topic>?").

Breadth first: the first topic of every subject, then the second, and so on,
so an interrupted run still covers every subject. Resumable: answers already
in the cache are skipped. Only clean answers are pinned (no warnings).

Run:  venv\\Scripts\\python -m src.inference.question_bank --limit 300
      (add --modes 2-mark 5-mark, --branches CSE ECE, --per-subject 2)
"""

import argparse
import ctypes
import time
from datetime import datetime

from config import settings
from src.inference.answer_cache import SemanticAnswerCache
from src.utils.catalogue_loader import load_catalogue

LOG = settings.BASE_DIR / "outputs" / "question_bank.log"


def keep_awake():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)
    except Exception:
        pass


def log(text):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {text}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def plan(per_subject, branches=None):
    """(question, subject, branch code, branch name) in breadth-first order."""
    catalogue = load_catalogue()
    names = {g["code"]: g["name"] for g in catalogue["metadata"]["groups"]}
    home = {}
    for code, subjects in catalogue["groups"].items():
        for s in subjects:
            # A subject's home branch: the one its id starts with (COM subjects: the first listing).
            if s["id"].startswith(code) or s["id"] not in home:
                home[s["id"]] = code
    subjects = [s for s in catalogue["subjects"] if not branches or home.get(s["id"]) in branches]
    jobs = []
    for rank in range(per_subject):
        for s in subjects:
            if rank < len(s["topics"]):
                code = home.get(s["id"])
                jobs.append((f"Explain {s['topics'][rank]}.", s, code, names.get(code, code)))
    return jobs


def signatures(mode, subject, branch_code, strategy="adaptive"):
    sig = SemanticAnswerCache.signature
    out = [sig(mode, {"subject_id": subject["id"]}, strategy), sig(mode, None, strategy)]
    if branch_code and subject["id"].startswith(branch_code):
        out.append(sig(mode, {"discipline": branch_code}, strategy))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-subject", type=int, default=2, help="topics per subject (breadth first)")
    parser.add_argument("--modes", nargs="+", default=["5-mark"])
    parser.add_argument("--branches", nargs="+", help="only these branch codes, e.g. CSE ECE")
    parser.add_argument("--limit", type=int, default=None, help="stop after this many new answers")
    args = parser.parse_args()
    keep_awake()

    from src.pipeline import EduRoutePipeline
    from src.rag.retrieval import EduRouteRetriever
    retriever = EduRouteRetriever(version=settings.RAG_VERSION, mode=settings.RETRIEVAL_MODE)
    cache = SemanticAnswerCache(retriever.model, path=settings.ANSWER_CACHE_PATH)
    pipeline = EduRoutePipeline(strategy="adaptive", retriever=retriever, log=False, answer_cache=cache)

    jobs = [(q, s, code, name, mode) for q, s, code, name in plan(args.per_subject, args.branches)
            for mode in args.modes]
    log(f"Question bank: {len(jobs)} questions planned ({len({j[1]['id'] for j in jobs})} subjects, "
        f"modes {' '.join(args.modes)}); cache has {cache.stats()['pinned']} pinned answers")
    made = skipped = rejected = 0
    started = time.monotonic()
    for question, subject, code, name, mode in jobs:
        sigs = signatures(mode, subject, code)
        if all(cache.contains(question, sg) for sg in sigs):
            skipped += 1
            continue
        t0 = time.monotonic()
        response = pipeline.ask(
            question, answer_mode=mode, filters={"subject_id": subject["id"]},
            fallback_filters=[{"discipline": code}, None] if code else [None],
            subject_context={"Branch": name, "Subject": f"{subject['name']} (Year {subject['year']})"},
        )
        if not response.telemetry.get("cache_hit") and not SemanticAnswerCache.cacheable(response):
            rejected += 1
            log(f"  not pinned ({'; '.join(response.notices)[:90] or response.workflow}): {question} [{subject['id']}]")
            continue
        record = response.to_dict()
        vector = retriever.embed_query(question)[0]
        for sg in sigs:
            cache.put(question, sg, record, vector=vector, pinned=True)
        made += 1
        log(f"  pinned {made}: {question} [{subject['id']} {mode}] in {time.monotonic() - t0:.0f}s")
        if args.limit and made >= args.limit:
            break
    hours = (time.monotonic() - started) / 3600
    log(f"Done: {made} new, {skipped} already in the bank, {rejected} not pinned, {hours:.1f} h; "
        f"cache now holds {cache.stats()['pinned']} pinned answers")


if __name__ == "__main__":
    main()
