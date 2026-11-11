"""
Fill knowledge-base gaps without rebuilding the index.

Inputs:
  data/metadata/aicte_gaps.csv   AICTE core courses no catalogue subject covers
  EXTRA_TOPICS below             gaps found in evaluation (held-out router check,
                                 speech term correction)

Each gap topic goes through the same audited path as the main corpus
(exact titles first, search only when needed, embedding-scored candidates,
stubs and disambiguation pages rejected). It is attached to the closest
existing subject in its branch, so subject filters find it. Articles already
in the corpus are skipped. Only the NEW passages are chunked, embedded and
appended to the FAISS index and its metadata (the full index took 3.7 h to
embed on a laptop CPU; an extension takes minutes).

Safety: the index and metadata are backed up first (*.bak), and the extended
index is reloaded and checked before the run reports success.

    venv\\Scripts\\python -m src.data_collection.extend_corpus [--dry-run]
"""

import argparse
import csv
import json
import shutil
import time
from datetime import datetime, timezone

import faiss
import numpy as np

from config import settings
from src.data_collection.build_engineering_corpus import (
    EXACT_ACCEPT_SCORE, MAX_ARTICLE_CHARS, MIN_ARTICLE_CHARS, OUTPUT_FILE, JsonlCache, Scorer,
    WikipediaClient, fetch_articles, fetch_intros, norm, search_with_intros, truncate_sections)
from src.data_collection.wikitext import tidy_formula_wrappers
from src.rag.build_index_v4 import (EMBEDDING_DIMENSION, chunk_section, context_prefix, dataset_paths,
                                    merge_short_sections)
from src.rag.text_cleaning import clean_article
from src.utils.catalogue_loader import load_catalogue
from src.utils.io import load_jsonl, write_json, write_jsonl

GAPS_FILE = settings.METADATA_DIR / "aicte_gaps.csv"
AUDIT_FILE = settings.METADATA_DIR / "engineering_supplement_v1.csv"
REPORT_FILE = settings.METADATA_DIR / "engineering_supplement_report_v1.json"
EXTRA_TOPICS = [
    ("ECE", "Phase-locked loop", "router held-out check: no article (H09)"),
    ("AIML", "Bayes' theorem", "speech term correction: term missing from the vocabulary"),
]


# Authoritative page hints for AICTE course titles that are compounds or
# ambiguous. The dry run without them matched journals ("Computer Networks
# (journal)"), a company ("Analog Devices") and unrelated pages.
GAP_HINTS = {
    "Computer Networks": "Computer network",
    "Computer Network": "Computer network",
    "Design and Analysis of Algorithms": "Analysis of algorithms",
    "Database Management Systems": "Database",
    "Formal Language & Automata Theory": "Automata theory",
    "Compiler Design": "Compiler",
    "Material and Energy Balance Computations": "Material balance",
    "Analog & Digital Electronics": "Electronic circuit",
    "Analog Electronic Circuits": "Analogue electronics",
    "Signals and Systems": "Signal processing",
    "Electromagnetic Fields": "Electromagnetic field",
    "Industrial Electronics and Power": "Power electronics",
    "Data Modeling and Visualization": "Data visualization",
    "Fundamentals of Materials Science & Smart Materials": "Smart material",
    "Materials, Testing & Evaluation": "Nondestructive testing",
    "Disaster Preparedness & Planning Management": "Emergency management",
    "Knowledge Engineering and Expert System": "Expert system",
    "Instrumentation & Sensor Technologies for Civil Engineering Applications": "Structural health monitoring",
    "Hydraulic & Pneumatic Drives for Robots": "Fluid power",
    "Data Structures, Files and Algorithms": "Data structure",
    "Data Structure and Algorithms": "Data structure",
    "Embedded Systems Design": "Embedded system",
    "Theory of Machines & Machine Design": "Mechanism (engineering)",
}


def gap_topics():
    rows = list(csv.DictReader(open(GAPS_FILE, encoding="utf-8"))) if GAPS_FILE.exists() else []
    topics = [(r["branch"], r["aicte_course"], f"AICTE core course gap ({r['document']} p.{r['page']})")
              for r in rows]
    return topics + EXTRA_TOPICS


def attach_subjects(raw_topics, model):
    """Each gap topic -> the closest existing subject in its branch."""
    catalogue = load_catalogue()
    names = {g["code"]: g["name"] for g in catalogue["metadata"]["groups"]}
    topics = []
    for branch, topic, reason in raw_topics:
        subjects = catalogue["groups"].get(branch, [])
        if not subjects:
            continue
        sub_vecs = model.encode([f"{s['name']}: {', '.join(s['topics'])}" for s in subjects],
                                normalize_embeddings=True, show_progress_bar=False)
        t_vec = model.encode([topic], normalize_embeddings=True, show_progress_bar=False)[0]
        best = subjects[int(np.argmax(sub_vecs @ t_vec))]
        topics.append({"discipline": branch, "branch_name": names.get(branch, branch),
                       "subject_id": best["id"], "subject_name": best["name"], "year": best["year"],
                       "topic": topic, "hint": GAP_HINTS.get(topic),
                       "candidates": list(dict.fromkeys(c for c in (GAP_HINTS.get(topic), topic) if c)),
                       "reason": reason})
    return topics


def resolve(topics, client, caches, scorer):
    search_cache, intro_cache = caches
    fetch_intros(client, intro_cache, [c for t in topics for c in t["candidates"]])
    scorer.score_topics(topics, intro_cache)
    weak = [t for t in topics if t["best_score"] < EXACT_ACCEPT_SCORE]
    for t in weak:
        found = search_with_intros(client, search_cache, intro_cache, f"{t['topic']} {t['subject_name']}")
        found += search_with_intros(client, search_cache, intro_cache, t["topic"], limit=3)
        t["candidates"] = list(dict.fromkeys(t["candidates"] + found))
    fetch_intros(client, intro_cache, [c for t in weak for c in t["candidates"]])
    scorer.score_topics(weak, intro_cache)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="resolve and report only; change nothing")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    started = time.time()
    model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device="cpu")
    client = WikipediaClient()
    search_cache, intro_cache, article_cache = JsonlCache("search"), JsonlCache("intros"), JsonlCache("articles")

    topics = attach_subjects(gap_topics(), model)
    print(f"Gap topics: {len(topics)}", flush=True)
    resolve(topics, client, (search_cache, intro_cache), Scorer(model))

    corpus = load_jsonl(OUTPUT_FILE)
    existing = {norm(a["title"]) for a in corpus}
    resolved = [t for t in topics if t.get("resolved_title")]
    new_titles = sorted({t["resolved_title"] for t in resolved if norm(t["resolved_title"]) not in existing})
    print(f"Resolved {len(resolved)}/{len(topics)}; already in the corpus: "
          f"{len({t['resolved_title'] for t in resolved}) - len(new_titles)}; new articles: {len(new_titles)}",
          flush=True)

    with AUDIT_FILE.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["discipline", "subject_id", "subject_name", "topic", "reason", "resolved_title",
                         "score", "status", "runner_up", "rejected_candidates", "reviewer_ok"])
        for t in topics:
            title = t.get("resolved_title")
            status = ("unresolved" if not title else
                      "already in corpus" if norm(title) not in {norm(n) for n in new_titles} else "added")
            writer.writerow([t["discipline"], t["subject_id"], t["subject_name"], t["topic"], t["reason"],
                             title or "", t.get("score"), status, t.get("runner_up", ""), t.get("rejected", ""), ""])
    print("Audit:", AUDIT_FILE, flush=True)
    if args.dry_run or not new_titles:
        print("Dry run: nothing changed." if args.dry_run else "Nothing new to add.")
        return

    fetch_articles(client, article_cache, new_titles)
    records = {}
    for t in resolved:
        title = t["resolved_title"]
        if title not in new_titles:
            continue
        mapping = {k: t[k] for k in ("discipline", "subject_id", "subject_name", "topic")}
        raw = article_cache.get(title)
        key = raw["title"] if raw else title
        if key in records:
            if mapping not in records[key]["subjects"]:
                records[key]["subjects"].append(mapping)
            continue
        sections, _ = clean_article(tidy_formula_wrappers(raw["text"])) if raw and raw["text"] else ([], None)
        if not sections or sum(len(s["text"]) for s in sections) < MIN_ARTICLE_CHARS:
            print(f"  skipped (too short after cleaning): {title}")
            continue
        sections, was_truncated = truncate_sections(sections, MAX_ARTICLE_CHARS)
        records[key] = {
            "title": raw["title"], "text": "\n\n".join(s["text"] for s in sections), "sections": sections,
            "source": "Wikipedia", "url": raw["url"], "page_id": raw["page_id"],
            "revision_id": raw["revision_id"], "revision_timestamp": raw["revision_timestamp"],
            "retrieved_at": raw["retrieved_at"],
            "license_note": "Wikipedia text is generally available under CC BY-SA and GFDL; "
                            "verify applicable attribution and licensing.",
            "content_cleanup_status": "engineering_v1_supplement" + ("_truncated" if was_truncated else ""),
            "subjects": [mapping],
        }
    new_articles = []
    for record in records.values():
        primary = record["subjects"][0]
        record.update(discipline=primary["discipline"], subject_id=primary["subject_id"],
                      subject_name=primary["subject_name"], topic=primary["topic"],
                      disciplines=sorted({s["discipline"] for s in record["subjects"]}),
                      subject_ids=sorted({s["subject_id"] for s in record["subjects"]}))
        new_articles.append(record)
    if not new_articles:
        print("No usable new articles.")
        return

    # ---------------------------------------------------- chunk + embed
    paths = dataset_paths("engineering_v1")
    tokenizer = AutoTokenizer.from_pretrained(settings.EMBEDDING_MODEL_NAME)
    metadata = load_jsonl(paths["metadata"])
    next_vector = len(metadata)
    chunks = []
    for article_number, article in enumerate(new_articles, start=1):
        pieces = []
        for section in merge_short_sections(article["sections"], tokenizer):
            for piece in chunk_section(section["text"], tokenizer):
                pieces.append((section["heading_path"], piece))
        for chunk_number, (heading_path, piece) in enumerate(pieces, start=1):
            chunks.append({
                "chunk_id": f"ENG_V1S_{article_number:04d}_CHUNK_{chunk_number:04d}",
                "article_title": article["title"],
                "section": " > ".join(heading_path) if heading_path else "Introduction",
                "subject_id": article["subject_id"], "subject_name": article["subject_name"],
                "topic": article["topic"], "discipline": article["discipline"],
                "disciplines": article["disciplines"], "subject_ids": article["subject_ids"],
                "source": "Wikipedia", "source_url": article["url"], "page_id": article["page_id"],
                "revision_id": article["revision_id"], "revision_timestamp": article["revision_timestamp"],
                "license_note": article["license_note"], "review_status": article["content_cleanup_status"],
                "text": piece["text"],
                "embed_text": context_prefix(article["title"], heading_path, tokenizer) + piece["text"],
                "token_count": piece["token_count"], "chunk_index": chunk_number,
                "total_article_chunks": len(pieces),
            })
    print(f"New articles: {len(new_articles)}; new passages: {len(chunks)}; embedding...", flush=True)
    t0 = time.time()
    vectors = model.encode([c["embed_text"] for c in chunks], batch_size=32, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    embed_seconds = round(time.time() - t0, 1)
    if vectors.shape != (len(chunks), EMBEDDING_DIMENSION) or not np.isfinite(vectors).all():
        raise ValueError(f"Unexpected embeddings {vectors.shape}")

    # ------------------------------------------------- append, with backup
    for path in (paths["index"], paths["metadata"], OUTPUT_FILE):
        shutil.copy2(path, str(path) + ".bak")
    index = faiss.read_index(str(paths["index"]))
    before = index.ntotal
    if before != next_vector:
        raise RuntimeError(f"Index ({before}) and metadata ({next_vector}) are out of step; nothing changed.")
    index.add(vectors)
    faiss.write_index(index, str(paths["index"]))
    for offset, chunk in enumerate(chunks):
        record = {"vector_id": next_vector + offset, **chunk}
        record.pop("embed_text")
        metadata.append(record)
    write_jsonl(paths["metadata"], metadata)
    write_jsonl(OUTPUT_FILE, corpus + new_articles)

    reloaded = faiss.read_index(str(paths["index"]))
    if reloaded.ntotal != len(metadata) or reloaded.ntotal != before + len(chunks):
        raise RuntimeError("Verification failed: restore the *.bak files.")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "gap_topics": len(topics),
        "resolved": len(resolved), "new_articles": [a["title"] for a in new_articles],
        "new_passages": len(chunks), "embedding_seconds": embed_seconds,
        "index_vectors_before": before, "index_vectors_after": reloaded.ntotal,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(REPORT_FILE, report)
    print(json.dumps({k: v for k, v in report.items() if k != "new_articles"}, indent=1))
    print("Backups: *.bak next to the index, metadata and corpus files.")


if __name__ == "__main__":
    main()
