
import json
import time
import requests

from pathlib import Path
from datetime import datetime, timezone

API_URL = "https://en.wikipedia.org/w/api.php"

OUTPUT_FILE = Path(
    "data/processed/wikipedia/aiml_replacement_articles.jsonl"
)

REPORT_FILE = Path(
    "data/metadata/aiml_fetch_report.json"
)

TITLE = "Graph neural network"

HEADERS = {
    "User-Agent": "EduRouteAI/1.0 (B.Tech educational research project)"
}

SUBJECT_ID = "AIML005"
SUBJECT_NAME = "Deep Learning"


def main():
    existing = []

    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open("r", encoding="utf-8") as file:
            existing = [
                json.loads(line)
                for line in file if line.strip()
            ]

    if any(x["title"].casefold() == TITLE.casefold()
           for x in existing):
        print("Article already exists. No changes made.")
        return

    session = requests.Session()
    session.headers.update(HEADERS)

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|info|revisions",
        "titles": TITLE,
        "explaintext": 1,
        "inprop": "url",
        "rvprop": "ids|timestamp",
        "rvlimit": 1,
        "redirects": 1
    }

    print("Waiting 60 seconds before retry...")
    time.sleep(60)

    for attempt in range(1, 7):
        print(f"\nAttempt {attempt}/6")

        try:
            response = session.get(
                API_URL,
                params=params,
                timeout=45
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait = (
                    int(retry_after)
                    if retry_after and retry_after.isdigit()
                    else min(30 * (2 ** (attempt - 1)), 300)
                )

                print(f"Rate limited. Waiting {wait} seconds.")
                time.sleep(wait)
                continue

            response.raise_for_status()

            data = response.json()
            page = next(iter(
                data["query"]["pages"].values()
            ))

            if "missing" in page:
                raise ValueError("Wikipedia article not found")

            text = page.get("extract", "").strip()

            if len(text) < 500:
                raise ValueError("Article text too short")

            revision = page.get("revisions", [{}])[0]

            record = {
                "title": page.get("title", TITLE),
                "text": text,
                "source": "Wikipedia",
                "url": page.get("fullurl", ""),
                "page_id": str(page.get("pageid", "")),
                "revision_id": str(revision.get("revid", "")),
                "revision_timestamp": revision.get("timestamp", ""),
                "retrieved_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "license_note": (
                    "Wikipedia content; verify article-specific "
                    "license and attribution requirements before "
                    "redistribution."
                ),
                "discipline": "AIML",
                "subject_id": SUBJECT_ID,
                "subject_name": SUBJECT_NAME,
                "topic": "Deep Learning",
                "cleaning_status": "fetched_pending_review"
            }

            with OUTPUT_FILE.open(
                "a", encoding="utf-8"
            ) as file:
                file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

            if REPORT_FILE.exists():
                report = json.loads(
                    REPORT_FILE.read_text(encoding="utf-8")
                )

                report["fetched"] = len(existing) + 1
                report["failed"] = 0
                report["failures"] = []

                REPORT_FILE.write_text(
                    json.dumps(
                        report,
                        indent=2,
                        ensure_ascii=False
                    ),
                    encoding="utf-8"
                )

            print("\nSUCCESS")
            print("Article:", record["title"])
            print("Characters:", len(text))
            print("Saved:", OUTPUT_FILE.resolve())
            return

        except (
            requests.RequestException,
            ValueError,
            KeyError,
            StopIteration
        ) as error:
            print("Error:", error)

            if attempt < 6:
                time.sleep(30)

    print("\nRetry unsuccessful. Existing dataset preserved.")


if __name__ == "__main__":
    main()
