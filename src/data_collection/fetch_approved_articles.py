
import json
import time
import requests

from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import quote

API_URL = "https://en.wikipedia.org/w/api.php"

APPROVED_FILE = Path(
    "data/processed/wikipedia/aiml_approved_articles.json"
)

EXISTING_FILE = Path(
    "data/processed/wikipedia/aiml_wikipedia_clean.jsonl"
)

OUTPUT_FILE = Path(
    "data/processed/wikipedia/aiml_replacement_articles.jsonl"
)

HEADERS = {
    "User-Agent": "EduRouteAI/1.0 (B.Tech educational research project)"
}

# Reuse existing subject IDs.
SUBJECT_MAP = {
    "Explainable AI": ("AIML001", "Introduction to Artificial Intelligence"),
    "Machine Learning": ("AIML002", "Machine Learning"),
    "Deep Learning": ("AIML005", "Deep Learning"),
    "Natural Language Processing": ("AIML006", "Natural Language Processing"),
    "Computer Vision": ("AIML007", "Computer Vision"),
    "Reinforcement Learning": ("AIML008", "Reinforcement Learning"),
    "Generative AI": ("AIML010", "Generative Artificial Intelligence"),
    "MLOps": ("AIML011", "MLOps and Model Deployment"),
}


def load_existing_titles():
    titles = set()

    if EXISTING_FILE.exists():
        with EXISTING_FILE.open(
            "r", encoding="utf-8"
        ) as file:
            for line in file:
                if line.strip():
                    article = json.loads(line)
                    titles.add(article["title"].casefold())

    return titles


def fetch_article(session, title):
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|info|revisions",
        "titles": title,
        "explaintext": 1,
        "exsectionformat": "wiki",
        "inprop": "url",
        "rvprop": "ids|timestamp",
        "rvlimit": 1,
        "redirects": 1,
    }

    for attempt in range(4):
        try:
            response = session.get(
                API_URL,
                params=params,
                timeout=45
            )

            if response.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()

            data = response.json()
            pages = data["query"]["pages"]
            page = next(iter(pages.values()))

            if "missing" in page:
                return None, "Article not found"

            text = page.get("extract", "").strip()

            if len(text) < 500:
                return None, "Article text too short"

            revision = page.get("revisions", [{}])[0]

            return {
                "title": page.get("title", title),
                "text": text,
                "source": "Wikipedia",
                "url": page.get(
                    "fullurl",
                    "https://en.wikipedia.org/wiki/"
                    + quote(title.replace(" ", "_"))
                ),
                "page_id": str(page.get("pageid", "")),
                "revision_id": str(revision.get("revid", "")),
                "revision_timestamp": revision.get("timestamp", ""),
            }, None

        except (
            requests.RequestException,
            ValueError,
            KeyError,
            StopIteration
        ) as error:
            if attempt == 3:
                return None, str(error)

            time.sleep(3 * (attempt + 1))

    return None, "Maximum retries exceeded"


def main():
    approved = json.loads(
        APPROVED_FILE.read_text(encoding="utf-8")
    )

    existing_titles = load_existing_titles()

    session = requests.Session()
    session.headers.update(HEADERS)

    fetched = []
    failed = []
    seen = set()

    print("\n========== EDUROUTE AI ARTICLE FETCHING ==========")
    print("Approved articles:", len(approved))

    for index, candidate in enumerate(approved, 1):
        title = candidate["title"]
        subject = candidate["subject"]

        print(f"\n[{index}/{len(approved)}] Fetching: {title}")

        key = title.casefold()

        if key in existing_titles or key in seen:
            print("Skipped: duplicate")
            continue

        if subject not in SUBJECT_MAP:
            failed.append({
                "title": title,
                "reason": "Unknown subject classification"
            })
            continue

        article, error = fetch_article(session, title)

        if error:
            print("Failed:", error)
            failed.append({
                "title": title,
                "reason": error
            })
            continue

        subject_id, subject_name = SUBJECT_MAP[subject]

        record = {
            **article,
            "retrieved_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "license_note": (
                "Wikipedia content; verify article-specific "
                "license and attribution requirements before "
                "redistribution."
            ),
            "discipline": "AIML",
            "subject_id": subject_id,
            "subject_name": subject_name,
            "topic": subject,
            "cleaning_status": "fetched_pending_review"
        }

        fetched.append(record)
        seen.add(key)

        print(
            f"Success | {len(article['text']):,} characters "
            f"| {subject_id}"
        )

        time.sleep(2)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w", encoding="utf-8"
    ) as file:
        for record in fetched:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    report = {
        "approved": len(approved),
        "fetched": len(fetched),
        "failed": len(failed),
        "output_file": str(OUTPUT_FILE),
        "failures": failed,
    }

    report_path = Path(
        "data/metadata/aiml_fetch_report.json"
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n========== FETCH SUMMARY ==========")
    print("Approved:", len(approved))
    print("Fetched:", len(fetched))
    print("Failed:", len(failed))
    print("Output:", OUTPUT_FILE.resolve())
    print("Report:", report_path.resolve())


if __name__ == "__main__":
    main()
