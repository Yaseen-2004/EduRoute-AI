"""
EduRoute AI - V4 knowledge-base preparation.

Fixes found while auditing V3 (see docs/PROGRESS.md):
  1. Seven V3 records are the wrong Wikipedia page: "CNN" is the news network,
     "Lime" is the fruit, and "Spark", "RNN", "Regression", "Clustering" and
     "Transparency" are disambiguation pages. They are removed and replaced
     with the intended technical articles.
  2. The two integrated Transformer articles were filed under AIML009 (Big
     Data Analytics); they belong to AIML006 / Transformers.
  3. Reference-style sections are removed with the section-aware cleaner
     (src/rag/text_cleaning.py) instead of truncate-at-first-heading.
  4. Section structure is preserved for chunk metadata.

Run from the project root:
    venv\\Scripts\\python -m src.data_collection.prepare_wikipedia_v4
Fetched replacements are cached in data/raw/wikipedia, so reruns are offline.
"""

import time
from collections import Counter
from datetime import datetime, timezone

from config.settings import BASE_DIR
from src.rag.text_cleaning import clean_article
from src.utils.io import load_jsonl, write_jsonl, write_json

V3_FILE = BASE_DIR / "data" / "processed" / "wikipedia" / "aiml_wikipedia_v3.jsonl"
REPLACEMENT_CACHE = BASE_DIR / "data" / "raw" / "wikipedia" / "aiml_v4_replacements.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "wikipedia" / "aiml_wikipedia_v4.jsonl"
REPORT_FILE = BASE_DIR / "data" / "metadata" / "aiml_v4_preparation_report.json"

API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "EduRouteAI/1.0 (B.Tech educational research project)"}
REQUEST_DELAY = 1.0

WRONG_PAGES = {
    "cnn": "Cable News Network article, not convolutional neural networks",
    "lime": "Disambiguation page (fruit/material), not the LIME explainer",
    "spark": "Disambiguation page, not Apache Spark",
    "rnn": "Disambiguation page",
    "regression": "Disambiguation page",
    "clustering": "Disambiguation page",
    "transparency": "Disambiguation page (optics etc.)",
}

SUBJECTS = {
    "AIML002": "Machine Learning",
    "AIML005": "Deep Learning",
    "AIML006": "Natural Language Processing",
    "AIML007": "Computer Vision",
    "AIML009": "Big Data Analytics",
    "AIML012": "Explainable Artificial Intelligence",
    "AIML013": "AI Ethics and Responsible AI",
}

# title -> (subject_id, topic)
REPLACEMENTS = {
    "Regression analysis": ("AIML002", "Regression"),
    "Cluster analysis": ("AIML002", "Clustering"),
    "Statistical classification": ("AIML002", "Classification"),
    "Long short-term memory": ("AIML005", "RNN"),
    "Gradient descent": ("AIML005", "Optimization"),
    "Attention (machine learning)": ("AIML006", "Transformers"),
    "Lexical analysis": ("AIML006", "Tokenization"),
    "Image segmentation": ("AIML007", "Segmentation"),
    "Apache Spark": ("AIML009", "Spark"),
    "Shapley value": ("AIML012", "SHAP"),
    "Algorithmic transparency": ("AIML013", "Transparency"),
}

SUBJECT_FIXES = {
    "Transformer (deep learning)": ("AIML006", "Transformers"),
    "Attention Is All You Need": ("AIML006", "Transformers"),
}


def _session():
    # Use the Windows certificate store: on this machine the bundled certifi
    # store fails TLS verification (HTTPS inspection), so verification is kept
    # on but delegated to the OS trust store.
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    import requests

    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def fetch_article(session, title):
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|info|revisions|pageprops",
        "ppprop": "disambiguation",
        "titles": title,
        "explaintext": 1,
        "inprop": "url",
        "rvprop": "ids|timestamp",
        "redirects": 1,
        "maxlag": 5,
    }
    for attempt in range(4):
        response = session.get(API_URL, params=params, timeout=30)
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(float(response.headers.get("Retry-After", 2 ** (attempt + 1))))
            continue
        response.raise_for_status()
        break
    else:
        raise RuntimeError(f"Failed to fetch {title}")

    page = next(iter(response.json()["query"]["pages"].values()))
    if "missing" in page:
        raise ValueError(f"Wikipedia page missing: {title}")
    if "disambiguation" in page.get("pageprops", {}):
        raise ValueError(f"Refusing disambiguation page: {title}")

    revision = page.get("revisions", [{}])[0]
    return {
        "title": page["title"],
        "text": page.get("extract", "").strip(),
        "source": "Wikipedia",
        "url": page.get("fullurl"),
        "page_id": page.get("pageid"),
        "revision_id": revision.get("revid"),
        "revision_timestamp": revision.get("timestamp"),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "license_note": (
            "Wikipedia text is generally available under CC BY-SA and GFDL; "
            "verify applicable attribution and licensing."
        ),
    }


def load_replacements():
    cached = {}
    if REPLACEMENT_CACHE.exists():
        cached = {r["requested_title"]: r for r in load_jsonl(REPLACEMENT_CACHE)}

    missing = [t for t in REPLACEMENTS if t not in cached]
    if missing:
        session = _session()
        for title in missing:
            print(f"[FETCH] {title}")
            record = fetch_article(session, title)
            record["requested_title"] = title
            cached[title] = record
            time.sleep(REQUEST_DELAY)
        write_jsonl(REPLACEMENT_CACHE, list(cached.values()))

    articles = []
    for title, (subject_id, topic) in REPLACEMENTS.items():
        record = dict(cached[title])
        record.pop("requested_title", None)
        record.update({
            "discipline": "AIML",
            "subject_id": subject_id,
            "subject_name": SUBJECTS[subject_id],
            "topic": topic,
            "cleaning_status": "v4_replacement",
        })
        articles.append(record)
    return articles


def main():
    print("\n========== EDUROUTE AI V4 DATA PREPARATION ==========\n")

    v3_articles = load_jsonl(V3_FILE)
    removed = []
    articles = []

    for article in v3_articles:
        key = article["title"].strip().lower()
        if key in WRONG_PAGES:
            removed.append({"title": article["title"], "reason": WRONG_PAGES[key]})
            continue
        article = dict(article)
        if article["title"] in SUBJECT_FIXES:
            subject_id, topic = SUBJECT_FIXES[article["title"]]
            article.update(subject_id=subject_id, subject_name=SUBJECTS[subject_id], topic=topic)
        articles.append(article)

    replacements = load_replacements()
    existing = {a["title"].lower() for a in articles}
    added = [a for a in replacements if a["title"].lower() not in existing]
    articles.extend(added)

    prepared = []
    cleaning_log = {}
    for article in articles:
        sections, dropped = clean_article(article["text"])
        if not sections:
            raise ValueError(f"No content left after cleaning: {article['title']}")

        record = {k: v for k, v in article.items() if k not in ("text", "content_cleanup_status")}
        record["sections"] = [
            {"heading_path": s["heading_path"], "text": s["text"]} for s in sections
        ]
        record["text"] = "\n\n".join(s["text"] for s in sections)
        record["content_cleanup_status"] = "v4_section_aware"
        prepared.append(record)

        if dropped["sections"] or dropped["citation_paragraphs"]:
            cleaning_log[article["title"]] = dropped

    titles = [a["title"].lower() for a in prepared]
    if len(titles) != len(set(titles)):
        raise ValueError("Duplicate titles in V4 dataset.")

    write_jsonl(OUTPUT_FILE, prepared)

    lengths = [len(a["text"]) for a in prepared]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": "V4",
        "source_file": str(V3_FILE.relative_to(BASE_DIR)),
        "output_file": str(OUTPUT_FILE.relative_to(BASE_DIR)),
        "v3_articles": len(v3_articles),
        "removed_wrong_pages": removed,
        "added_replacements": [a["title"] for a in added],
        "subject_fixes": list(SUBJECT_FIXES),
        "final_articles": len(prepared),
        "min_text_chars": min(lengths),
        "max_text_chars": max(lengths),
        "dropped_by_article": cleaning_log,
        "subject_distribution": dict(Counter(a["subject_id"] for a in prepared)),
    }
    write_json(REPORT_FILE, report)

    print("V3 articles:", len(v3_articles))
    print("Removed wrong pages:", len(removed))
    for item in removed:
        print("  -", item["title"], "->", item["reason"])
    print("Added replacements:", len(added))
    for title in report["added_replacements"]:
        print("  +", title)
    print("Final V4 articles:", len(prepared))
    print("Articles with non-content removed:", len(cleaning_log))
    print("\nOutput:", OUTPUT_FILE)
    print("Report:", REPORT_FILE)


if __name__ == "__main__":
    main()
