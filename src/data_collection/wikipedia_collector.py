
import json
import time
import requests

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from config.settings import (
    WIKIPEDIA_API_URL,
    WIKIPEDIA_PROCESSED_DIR,
    REQUEST_TIMEOUT
)

from src.utils.catalogue_loader import load_catalogue


HEADERS = {
    "User-Agent": "EduRouteAI/1.0 (B.Tech educational research project)"
}

REQUEST_DELAY = 2.0
MAX_RETRIES = 4

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def api_request(params):
    """Make a Wikipedia API request with retry and backoff."""

    params = {
        **params,
        "maxlag": 5
    }

    for attempt in range(MAX_RETRIES):

        try:
            response = SESSION.get(
                WIKIPEDIA_API_URL,
                params=params,
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 429 or response.status_code >= 500:

                retry_after = response.headers.get("Retry-After")

                if retry_after:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        try:
                            retry_date = parsedate_to_datetime(retry_after)
                            wait = max(
                                0,
                                (retry_date - datetime.now(timezone.utc)).total_seconds()
                            )
                        except (TypeError, ValueError):
                            wait = 2 ** (attempt + 1)
                else:
                    wait = 2 ** (attempt + 1)

                if attempt == MAX_RETRIES - 1:
                    response.raise_for_status()

                print(
                    f"[RETRY] HTTP {response.status_code}; "
                    f"waiting {wait:.1f} seconds"
                )

                time.sleep(wait)
                continue

            response.raise_for_status()

            time.sleep(REQUEST_DELAY)

            return response.json()

        except requests.RequestException:

            if attempt == MAX_RETRIES - 1:
                raise

            wait = 2 ** (attempt + 1)

            print(f"[RETRY] Request failed; waiting {wait} seconds")
            time.sleep(wait)

    return None


def search_wikipedia(topic, limit=3):
    """Search Wikipedia using the actual topic."""

    params = {
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "srlimit": limit,
        "srwhat": "text",
        "format": "json"
    }

    data = api_request(params)

    if not data:
        return []

    return data.get("query", {}).get("search", [])


def fetch_article(title):
    """Retrieve article text and metadata."""

    params = {
        "action": "query",
        "prop": "extracts|info|revisions",
        "titles": title,
        "explaintext": True,
        "inprop": "url",
        "rvprop": "ids|timestamp",
        "format": "json"
    }

    data = api_request(params)

    if not data:
        return None

    pages = data.get("query", {}).get("pages", {})

    if not pages:
        return None

    page = next(iter(pages.values()))

    article_text = page.get("extract", "").strip()

    if page.get("missing") is not None or not article_text:
        return None

    revision = page.get("revisions", [{}])[0]

    return {
        "title": page.get("title"),
        "text": article_text,
        "source": "Wikipedia",
        "url": page.get("fullurl"),
        "page_id": page.get("pageid"),
        "revision_id": revision.get("revid"),
        "revision_timestamp": revision.get("timestamp"),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "license_note": (
            "Wikipedia text is generally available under CC BY-SA "
            "and GFDL; verify applicable attribution and licensing."
        )
    }


def collect_branch(branch_code, max_articles_per_topic=1):

    catalogue = load_catalogue()

    branch_code = branch_code.upper()

    if branch_code not in catalogue["groups"]:
        raise ValueError(f"Unknown branch: {branch_code}")

    subjects = catalogue["groups"][branch_code]

    WIKIPEDIA_PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        WIKIPEDIA_PROCESSED_DIR /
        f"{branch_code.lower()}_wikipedia_v2.jsonl"
    )

    seen_titles = set()
    collected = 0
    failed_topics = []

    with open(output_file, "w", encoding="utf-8") as file:

        for subject in subjects:

            for topic in subject["topics"]:

                print(f"\n[SEARCHING] {topic}")

                try:
                    results = search_wikipedia(
                        topic,
                        limit=3
                    )

                    if not results:
                        print(f"[NO RESULTS] {topic}")
                        continue

                    saved_for_topic = 0

                    for result in results:

                        title = result.get("title", "")

                        if not title or title in seen_titles:
                            continue

                        article = fetch_article(title)

                        if not article:
                            continue

                        article.update({
                            "discipline": branch_code,
                            "subject_id": subject["id"],
                            "subject_name": subject["name"],
                            "topic": topic
                        })

                        file.write(
                            json.dumps(article, ensure_ascii=False) + "\n"
                        )

                        file.flush()

                        seen_titles.add(title)
                        collected += 1
                        saved_for_topic += 1

                        print(f"[COLLECTED] {topic} -> {title}")

                        if saved_for_topic >= max_articles_per_topic:
                            break

                except requests.RequestException as error:

                    failed_topics.append(topic)
                    print(f"[FAILED] {topic}: {error}")

    print("\n========== COLLECTION SUMMARY ==========")
    print("Branch:", branch_code)
    print("Subjects:", len(subjects))
    print("Unique articles:", collected)
    print("Failed topics:", len(failed_topics))
    print("Output:", output_file)

    if failed_topics:
        print("Failed topic names:", failed_topics)

    return output_file


if __name__ == "__main__":
    collect_branch("AIML", max_articles_per_topic=1)
