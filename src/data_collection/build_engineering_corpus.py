"""
EduRoute AI - engineering-wide knowledge base collection (all 26 catalogue groups).

For every topic in the catalogue (config/common_subjects.json and
config/branches/*.json; AIML uses the curated V4 articles):

  1. Exact titles  introductions for the topic title and its hint from
                   config/topic_hints.json, 20 pages per request.
  2. Search        only for topics whose exact page is missing, a
                   disambiguation page, or a weak match: one
                   generator=search request returns the top 5 results WITH
                   their introductions. A plain-topic search follows only if
                   that is still weak.
  3. Scoring       cosine similarity between "<topic> (<subject>, <branch>)"
                   and "<title>. <intro>" (MiniLM), +0.15 for the exact/hinted
                   title. Disambiguation, list/outline/glossary, missing and
                   stub pages are rejected. This guards against the V3 failure
                   where "CNN" resolved to the news channel.
  4. Articles      raw wikitext for 20 pages per request, converted locally
                   (src/data_collection/wikitext.py), then the same
                   section-aware cleaning as V4. Long articles keep their first
                   MAX_ARTICLE_CHARS characters of content sections.
  5. Merge         one record per article listing every subject/topic that maps
                   to it; the curated AIML V4 articles are merged in.

The Wikimedia API rate-limits this client (HTTP 429), so the design minimises
requests and always honours Retry-After. All responses are cached in
data/raw/wikipedia/engineering/; reruns are offline and interrupted runs resume.

Outputs
  data/processed/wikipedia/engineering_wikipedia_v1.jsonl
  data/metadata/engineering_corpus_report_v1.json
  data/metadata/engineering_topic_resolution_v1.csv   <- audit this

Run (project root):
    venv\\Scripts\\python -m src.data_collection.build_engineering_corpus
"""

import argparse
import csv
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone

from config import settings
from src.data_collection.wikitext import tidy_formula_wrappers, wikitext_to_text
from src.rag.text_cleaning import clean_article
from src.utils.catalogue_loader import load_catalogue
from src.utils.io import load_jsonl, write_json, write_jsonl

API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "EduRouteAI/1.1 (B.Tech educational research project)"}
CACHE_DIR = settings.WIKIPEDIA_RAW_DIR / "engineering"
OUTPUT_FILE = settings.WIKIPEDIA_PROCESSED_DIR / "engineering_wikipedia_v1.jsonl"
REPORT_FILE = settings.METADATA_DIR / "engineering_corpus_report_v1.json"
AUDIT_FILE = settings.METADATA_DIR / "engineering_topic_resolution_v1.csv"
AIML_V4_FILE = settings.WIKIPEDIA_PROCESSED_DIR / "aiml_wikipedia_v4.jsonl"
HINTS_FILE = settings.BASE_DIR / "config" / "topic_hints.json"

MIN_ACCEPT_SCORE = 0.50      # below this a topic is left unresolved (better than a wrong page)
EXACT_ACCEPT_SCORE = 0.60    # exact/hinted page this good -> no search needed
EXACT_TITLE_BONUS = 0.15
MIN_INTRO_CHARS = 60        # short leads are normal; real stubs are caught by MIN_ARTICLE_CHARS
MAX_ARTICLE_CHARS = 30000
MIN_ARTICLE_CHARS = 1000    # cleaned article shorter than this = stub, not indexed
ARTICLE_BATCH = 20
REJECT_TITLE = re.compile(r"^(list|lists|outline|index|glossary|timeline) of\b", re.IGNORECASE)


# ----------------------------------------------------------------- caching

class JsonlCache:
    """Append-only key/value cache so long runs can resume."""

    def __init__(self, name):
        self.path = CACHE_DIR / f"{name}.jsonl"
        self.data = {}
        if self.path.exists():
            for row in load_jsonl(self.path):
                self.data[row["key"]] = row["value"]

    def __contains__(self, key):
        return key in self.data

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        self.data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


class WikipediaClient:
    """Serial client that honours Retry-After / maxlag (Wikimedia API etiquette)."""

    def __init__(self, delay=1.0):
        try:
            import truststore
            truststore.inject_into_ssl()
        except ImportError:
            pass
        import requests

        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.delay = delay
        self.requests = 0
        self.rate_limited_seconds = 0.0

    def get(self, params):
        params = {**params, "format": "json", "formatversion": 2, "maxlag": 5}
        for attempt in range(12):
            try:
                response = self.session.get(API_URL, params=params, timeout=60)
            except Exception:
                time.sleep(min(60, 2 ** attempt))
                continue
            self.requests += 1
            if response.status_code == 429 or response.status_code >= 500:
                wait = float(response.headers.get("Retry-After", min(60, 2 ** (attempt + 1))))
                self.rate_limited_seconds += wait
                print(f"    HTTP {response.status_code}; waiting {wait:.0f} s as requested", flush=True)
                time.sleep(wait)
                continue
            data = response.json()
            if data.get("error", {}).get("code") == "maxlag":
                time.sleep(float(response.headers.get("Retry-After", 5)))
                continue
            time.sleep(self.delay)
            return data
        raise RuntimeError(f"Wikipedia request failed repeatedly: {params.get('titles') or params}")


# ------------------------------------------------------------------ helpers

def norm(title):
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


def catalogue_topics():
    catalogue = load_catalogue()
    hints = json.loads(HINTS_FILE.read_text(encoding="utf-8")) if HINTS_FILE.exists() else {}
    names = {g["code"]: g["name"] for g in catalogue["metadata"]["groups"]}
    topics = []
    for code, subjects in catalogue["groups"].items():
        if code == "AIML":
            continue  # curated V4 AIML articles are merged separately
        for subject in subjects:
            for topic in subject["topics"]:
                hint = hints.get(f"{subject['id']}|{topic}")
                topics.append({
                    "discipline": code, "branch_name": names[code], "subject_id": subject["id"],
                    "subject_name": subject["name"], "year": subject["year"], "topic": topic,
                    "hint": hint, "candidates": list(dict.fromkeys(c for c in (hint, topic) if c)),
                })
    return topics


def _page_entry(page):
    return {
        "title": page.get("title"),
        "pageid": page.get("pageid"),
        "missing": bool(page.get("missing")) or "pageid" not in page,
        "disambiguation": "disambiguation" in page.get("pageprops", {}),
        "intro": (page.get("extract") or "")[:1500],
    }


def fetch_intros(client, cache, titles):
    """Intro cache: requested title -> page entry (redirects resolved)."""
    pending = [t for t in dict.fromkeys(titles) if t not in cache]
    for start in range(0, len(pending), 20):
        batch = pending[start:start + 20]
        data = client.get({
            "action": "query", "prop": "extracts|pageprops", "exintro": 1, "explaintext": 1,
            "exlimit": 20, "ppprop": "disambiguation", "redirects": 1, "titles": "|".join(batch),
        })
        query = data.get("query", {})
        mapping = {t: t for t in batch}
        for key in ("normalized", "redirects"):
            for item in query.get(key, []):
                for requested, current in mapping.items():
                    if current == item["from"]:
                        mapping[requested] = item["to"]
        pages = {p.get("title"): p for p in query.get("pages", [])}
        for requested, resolved in mapping.items():
            page = pages.get(resolved, {"title": resolved, "missing": True})
            cache.put(requested, _page_entry(page))
        print(f"  intros {min(start + 20, len(pending))}/{len(pending)}", flush=True)


def search_with_intros(client, search_cache, intro_cache, query, limit=5):
    """One request: search results plus their introductions."""
    if query in search_cache:
        return search_cache.get(query)
    data = client.get({
        "action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": limit,
        "prop": "extracts|pageprops", "exintro": 1, "explaintext": 1, "exlimit": limit,
        "ppprop": "disambiguation",
    })
    pages = sorted(data.get("query", {}).get("pages", []), key=lambda p: p.get("index", 99))
    titles = []
    for page in pages:
        titles.append(page["title"])
        if page["title"] not in intro_cache:
            intro_cache.put(page["title"], _page_entry(page))
    search_cache.put(query, titles)
    return titles


class Scorer:
    def __init__(self, model):
        self.model = model
        self.page_vecs = {}

    def _encode(self, texts):
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=False)

    def ensure_pages(self, intro_cache, titles):
        new = {}
        for c in titles:
            info = intro_cache.get(c)
            if info and not info["missing"] and info["title"] not in self.page_vecs:
                new[info["title"]] = f"{info['title']}. {info['intro'][:1000]}"
        if new:
            for title, vec in zip(new, self._encode(list(new.values()))):
                self.page_vecs[title] = vec

    def score_topics(self, topics, intro_cache):
        self.ensure_pages(intro_cache, {c for t in topics for c in t["candidates"]})
        contexts = self._encode([f"{t['topic']} ({t['subject_name']}, {t['branch_name']})" for t in topics])
        for t, vec in zip(topics, contexts):
            scored, rejected = [], []
            for c in t["candidates"]:
                info = intro_cache.get(c)
                if not info or info["missing"]:
                    continue
                title = info["title"]
                if info["disambiguation"]:
                    rejected.append(f"{title} (disambiguation)")
                    continue
                if REJECT_TITLE.search(title):
                    rejected.append(f"{title} (list/outline)")
                    continue
                if len(info["intro"]) < MIN_INTRO_CHARS:
                    rejected.append(f"{title} (stub intro)")
                    continue
                exact = norm(c) in (norm(t["topic"]), norm(t["hint"]))
                score = float(self.page_vecs[title] @ vec) + (EXACT_TITLE_BONUS if exact else 0.0)
                scored.append((score, title, exact))
            scored.sort(reverse=True)
            # Deduplicate (several candidates can redirect to the same page).
            seen, unique = set(), []
            for item in scored:
                if item[1] not in seen:
                    seen.add(item[1])
                    unique.append(item)
            # A human-written hint is authoritative when its page is valid.
            hint_info = intro_cache.get(t["hint"]) if t["hint"] else None
            hinted = [u for u in unique if hint_info and u[1] == hint_info["title"]]
            best = hinted[0] if hinted else (unique[0] if unique else None)
            if hinted:
                unique = hinted + [u for u in unique if u is not hinted[0]]
            t["best_score"] = best[0] if best else 0.0
            t["resolved_title"] = best[1] if best and (best[0] >= MIN_ACCEPT_SCORE or hinted) else None
            t["score"] = round(best[0], 3) if best else None
            t["exact_title"] = bool(best and best[2])
            t["runner_up"] = f"{unique[1][1]} ({unique[1][0]:.2f})" if len(unique) > 1 else ""
            t["rejected"] = "; ".join(dict.fromkeys(rejected))


def truncate_sections(sections, limit):
    kept, total = [], 0
    for section in sections:
        if total >= limit:
            break
        text = section["text"][: max(0, limit - total)]
        kept.append({**section, "text": text})
        total += len(text)
    return kept, total >= limit


def fetch_articles(client, cache, titles):
    pending = [t for t in titles if t not in cache]
    for start in range(0, len(pending), ARTICLE_BATCH):
        batch = pending[start:start + ARTICLE_BATCH]
        data = client.get({
            "action": "query", "prop": "revisions|info", "rvprop": "content|ids|timestamp",
            "rvslots": "main", "inprop": "url", "redirects": 1, "titles": "|".join(batch),
        })
        pages = {p.get("title"): p for p in data.get("query", {}).get("pages", [])}
        for title in batch:
            page = pages.get(title, {})
            revision = (page.get("revisions") or [{}])[0]
            wikitext = revision.get("slots", {}).get("main", {}).get("content", "")
            cache.put(title, {
                "title": page.get("title", title),
                "text": wikitext_to_text(wikitext) if wikitext else "",
                "text_source": "wikitext",
                "url": page.get("fullurl"),
                "page_id": page.get("pageid"),
                "revision_id": revision.get("revid"),
                "revision_timestamp": revision.get("timestamp"),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            })
        print(f"  articles {min(start + ARTICLE_BATCH, len(pending))}/{len(pending)}", flush=True)


# --------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-topics", type=int, help="debug: only the first N topics")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    started = time.time()
    client = WikipediaClient()
    search_cache = JsonlCache("search")
    intro_cache = JsonlCache("intros")
    article_cache = JsonlCache("articles")
    scorer = Scorer(SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device="cpu"))

    topics = catalogue_topics()
    if args.limit_topics:
        topics = topics[: args.limit_topics]
    print(f"Topics to resolve (non-AIML): {len(topics)}", flush=True)

    print("1/4 exact titles and hints", flush=True)
    fetch_intros(client, intro_cache, [c for t in topics for c in t["candidates"]])
    scorer.score_topics(topics, intro_cache)

    weak = [t for t in topics if t["best_score"] < EXACT_ACCEPT_SCORE]
    print(f"2/4 searching for {len(weak)} topics without a confident exact page", flush=True)
    for n, t in enumerate(weak, start=1):
        found = search_with_intros(client, search_cache, intro_cache, f"{t['topic']} {t['subject_name']}")
        t["candidates"] = list(dict.fromkeys(t["candidates"] + found))
        if n % 25 == 0:
            print(f"  search {n}/{len(weak)}", flush=True)
    fetch_intros(client, intro_cache, [c for t in weak for c in t["candidates"]])  # older cached searches
    scorer.score_topics(weak, intro_cache)

    still_weak = [t for t in weak if t["best_score"] < EXACT_ACCEPT_SCORE]
    print(f"   second search for {len(still_weak)} topics", flush=True)
    for t in still_weak:
        found = search_with_intros(client, search_cache, intro_cache, t["topic"], limit=3)
        t["candidates"] = list(dict.fromkeys(t["candidates"] + found))
    fetch_intros(client, intro_cache, [c for t in still_weak for c in t["candidates"]])
    scorer.score_topics(still_weak, intro_cache)

    chosen = sorted({t["resolved_title"] for t in topics if t["resolved_title"]})
    print(f"3/4 fetching {len(chosen)} articles ({ARTICLE_BATCH} per request)", flush=True)
    fetch_articles(client, article_cache, chosen)

    print("4/4 merging", flush=True)
    records = {}
    for article in load_jsonl(AIML_V4_FILE):
        records[article["title"]] = {
            **article,
            "subjects": [{"discipline": "AIML", "subject_id": article["subject_id"],
                          "subject_name": article["subject_name"], "topic": article["topic"]}],
        }

    truncated, empty = 0, []
    for t in topics:
        title = t["resolved_title"]
        if not title:
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
            empty.append(title)
            continue
        sections, was_truncated = truncate_sections(sections, MAX_ARTICLE_CHARS)
        truncated += was_truncated
        records[key] = {
            "title": raw["title"],
            "text": "\n\n".join(s["text"] for s in sections),
            "sections": sections,
            "source": "Wikipedia",
            "url": raw["url"],
            "page_id": raw["page_id"],
            "revision_id": raw["revision_id"],
            "revision_timestamp": raw["revision_timestamp"],
            "retrieved_at": raw["retrieved_at"],
            "license_note": "Wikipedia text is generally available under CC BY-SA and GFDL; "
                            "verify applicable attribution and licensing.",
            "content_cleanup_status": "engineering_v1_section_aware" + ("_truncated" if was_truncated else ""),
            "subjects": [mapping],
        }

    output = []
    for record in records.values():
        primary = record["subjects"][0]
        record.update(
            discipline=primary["discipline"], subject_id=primary["subject_id"],
            subject_name=primary["subject_name"], topic=primary["topic"],
            disciplines=sorted({s["discipline"] for s in record["subjects"]}),
            subject_ids=sorted({s["subject_id"] for s in record["subjects"]}),
        )
        output.append(record)
    write_jsonl(OUTPUT_FILE, output)

    with AUDIT_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["discipline", "subject_id", "subject_name", "topic", "hint", "resolved_title",
                         "score", "exact_title", "runner_up", "rejected_candidates", "reviewer_ok"])
        for t in topics:
            writer.writerow([t["discipline"], t["subject_id"], t["subject_name"], t["topic"], t["hint"] or "",
                             t["resolved_title"] or "UNRESOLVED", t["score"], t["exact_title"],
                             t["runner_up"], t["rejected"], ""])

    resolved = [t for t in topics if t["resolved_title"]]
    subjects_covered = Counter(t["subject_id"] for t in resolved)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "engineering_v1",
        "topics_total_non_aiml": len(topics),
        "topics_resolved": len(resolved),
        "topics_unresolved": [f"{t['subject_id']}: {t['topic']}" for t in topics if not t["resolved_title"]],
        "resolved_by_exact_title_or_hint": sum(t["exact_title"] for t in resolved),
        "low_confidence_resolutions": [
            f"{t['subject_id']}: {t['topic']} -> {t['resolved_title']} ({t['score']})"
            for t in resolved if t["score"] < 0.60
        ],
        "subjects_without_any_article": sorted(
            {t["subject_id"] for t in topics} - set(subjects_covered)),
        "articles_total": len(output),
        "articles_truncated": truncated,
        "articles_empty_after_cleaning": empty,
        "articles_per_discipline": dict(sorted(Counter(d for r in output for d in r["disciplines"]).items())),
        "acceptance_threshold": MIN_ACCEPT_SCORE,
        "max_article_chars": MAX_ARTICLE_CHARS,
        "wikipedia_requests_this_run": client.requests,
        "seconds_waiting_for_rate_limit": round(client.rate_limited_seconds),
        "runtime_seconds": round(time.time() - started),
    }
    write_json(REPORT_FILE, report)

    print(f"\nResolved {len(resolved)}/{len(topics)} topics "
          f"({report['resolved_by_exact_title_or_hint']} by exact title/hint)")
    print(f"Articles: {len(output)} (truncated {truncated}); per discipline: {report['articles_per_discipline']}")
    print(f"Unresolved: {len(report['topics_unresolved'])}; low confidence: "
          f"{len(report['low_confidence_resolutions'])}; subjects with no article: "
          f"{report['subjects_without_any_article']}")
    print(f"Wikipedia requests: {client.requests}; rate-limit waits {report['seconds_waiting_for_rate_limit']} s; "
          f"total {report['runtime_seconds']} s")
    print("Audit sheet:", AUDIT_FILE)


if __name__ == "__main__":
    main()
