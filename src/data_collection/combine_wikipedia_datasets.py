
import json
from pathlib import Path
from collections import Counter

ORIGINAL_FILE = Path(
    "data/processed/wikipedia/aiml_wikipedia_clean.jsonl"
)

REPLACEMENT_FILE = Path(
    "data/processed/wikipedia/aiml_replacement_articles.jsonl"
)

OUTPUT_FILE = Path(
    "data/processed/wikipedia/aiml_wikipedia_combined.jsonl"
)

REPORT_FILE = Path(
    "data/metadata/aiml_combined_report.json"
)

REQUIRED_FIELDS = [
    "title", "text", "source", "url", "page_id",
    "revision_id", "revision_timestamp", "retrieved_at",
    "license_note", "discipline", "subject_id",
    "subject_name", "topic", "cleaning_status"
]


def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main():
    original = load_jsonl(ORIGINAL_FILE)
    replacements = load_jsonl(REPLACEMENT_FILE)

    records = original + replacements

    # Correct the Explainable AI subject mapping.
    for record in records:
        if record["title"].casefold() == (
            "explainable artificial intelligence"
        ).casefold():
            record["subject_id"] = "AIML012"
            record["subject_name"] = (
                "Explainable Artificial Intelligence"
            )
            record["topic"] = "Explainable AI"

    # Validate required fields.
    for index, record in enumerate(records, start=1):
        missing = [
            field for field in REQUIRED_FIELDS
            if not record.get(field)
        ]

        if missing:
            raise ValueError(
                f"Record {index} has missing fields: {missing}"
            )

    # Detect duplicate titles.
    title_counts = Counter(
        record["title"].casefold()
        for record in records
    )

    duplicates = [
        title for title, count in title_counts.items()
        if count > 1
    ]

    if duplicates:
        raise ValueError(
            f"Duplicate article titles found: {duplicates}"
        )

    # Verify text and URL.
    for record in records:
        if not record["text"].strip():
            raise ValueError(
                f"Empty text: {record['title']}"
            )

        if not record["url"].startswith("https://"):
            raise ValueError(
                f"Invalid source URL: {record['title']}"
            )

    # Save combined dataset without overwriting inputs.
    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w", encoding="utf-8"
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    subject_counts = Counter(
        record["subject_id"]
        for record in records
    )

    status_counts = Counter(
        record["cleaning_status"]
        for record in records
    )

    report = {
        "total_articles": len(records),
        "unique_titles": len(title_counts),
        "duplicate_titles": len(duplicates),
        "subject_distribution": dict(subject_counts),
        "status_distribution": dict(status_counts),
        "explainable_ai_mapping_corrected": True,
        "output_file": str(OUTPUT_FILE),
        "note": (
            "Newly fetched articles remain pending review. "
            "Basic structural validation does not establish "
            "factual accuracy."
        )
    }

    REPORT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=4,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n========== EDUROUTE AI DATASET MERGE ==========")
    print("Original articles:", len(original))
    print("Replacement articles:", len(replacements))
    print("Combined articles:", len(records))
    print("Unique titles:", len(title_counts))
    print("Duplicate titles:", len(duplicates))

    print("\nSubject distribution:")
    for subject, count in sorted(subject_counts.items()):
        print(f"{subject}: {count}")

    print("\nReview status:")
    for status, count in status_counts.items():
        print(f"{status}: {count}")

    print("\nCombined dataset:", OUTPUT_FILE)
    print("Report:", REPORT_FILE)
    print("\nDataset merge completed successfully.")


if __name__ == "__main__":
    main()
