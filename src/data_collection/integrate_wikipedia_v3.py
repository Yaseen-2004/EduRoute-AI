
import json
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]

V3_FILE = (
    ROOT / "data" / "processed" / "wikipedia"
    / "aiml_wikipedia_v3.jsonl"
)

INSPECTION_FILE = (
    ROOT / "data" / "raw" / "wikipedia"
    / "aiml_missing_articles_inspection.jsonl"
)

REPORT_FILE = (
    ROOT / "data" / "metadata"
    / "aiml_v3_integration_report.json"
)

ALLOWED_NEW_TITLES = {
    "transformer (deep learning)",
    "attention is all you need",
}


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main():
    print("\n========== EDUROUTE AI V3 INTEGRATION ==========\n")

    if not V3_FILE.exists():
        raise FileNotFoundError(V3_FILE)

    if not INSPECTION_FILE.exists():
        raise FileNotFoundError(INSPECTION_FILE)

    existing = load_jsonl(V3_FILE)
    inspected = load_jsonl(INSPECTION_FILE)

    existing_titles = {
        article["title"].strip().lower()
        for article in existing
    }

    added = []
    skipped = []

    for article in inspected:
        title = article.get("title", "").strip()
        normalized = title.lower()

        if normalized not in ALLOWED_NEW_TITLES:
            skipped.append({
                "title": title,
                "reason": "Not in approved article list"
            })
            continue

        if normalized in existing_titles:
            skipped.append({
                "title": title,
                "reason": "Duplicate article"
            })
            continue

        if not article.get("text", "").strip():
            skipped.append({
                "title": title,
                "reason": "Empty article text"
            })
            continue

        article = article.copy()

        article["content_cleanup_status"] = (
            "targeted_article_enrichment"
        )

        existing.append(article)
        existing_titles.add(normalized)
        added.append(title)

    # Validate before writing.
    titles = [
        article["title"].strip().lower()
        for article in existing
    ]

    if len(titles) != len(set(titles)):
        raise ValueError("Duplicate titles detected")

    for article in existing:
        for field in ["title", "text", "url", "subject_id"]:
            if not article.get(field):
                raise ValueError(
                    f"Missing {field} in {article.get('title')}"
                )

    # Write to a temporary file first.
    temporary_file = V3_FILE.with_suffix(".tmp")

    with temporary_file.open("w", encoding="utf-8") as file:
        for article in existing:
            file.write(
                json.dumps(article, ensure_ascii=False) + "\n"
            )

    temporary_file.replace(V3_FILE)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "final_article_count": len(existing),
        "added_articles": added,
        "skipped_articles": skipped,
        "unique_titles": len(set(titles)),
        "subject_distribution": dict(
            Counter(
                article.get("subject_id", "UNKNOWN")
                for article in existing
            )
        ),
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("Original V3 articles:", len(existing) - len(added))
    print("New articles added:", len(added))
    print("Final V3 articles:", len(existing))
    print("Unique titles:", len(set(titles)))

    print("\nAdded articles:")
    for title in added:
        print("-", title)

    print("\nSkipped articles:")
    for item in skipped:
        print("-", item["title"], "|", item["reason"])

    print("\nSubject distribution:")
    for subject, count in sorted(
        report["subject_distribution"].items()
    ):
        print(f"{subject}: {count}")

    print("\nUpdated dataset:", V3_FILE)
    print("Integration report:", REPORT_FILE)

    print("\nV3 INTEGRATION COMPLETE")


if __name__ == "__main__":
    main()
