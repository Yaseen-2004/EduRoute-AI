
import json
import re
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "wikipedia"
    / "aiml_wikipedia_v2.jsonl"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "wikipedia"
    / "aiml_wikipedia_clean.jsonl"
)

REPORT_FILE = (
    BASE_DIR
    / "data"
    / "metadata"
    / "aiml_cleaning_report.json"
)

# Articles excluded because their titles or contexts
# are clearly unsuitable for the assigned AIML topics.
EXCLUDED_TITLES = {
    "AI bubble",
    "Köppen climate classification",
    "PAQ",
    "Tokenization (data security)",
    "V-model",
    "Input hypothesis",
    "List of large language models",
    "Segment",
}


def clean_text(text):
    """Normalize whitespace without removing technical content."""
    if not isinstance(text, str):
        return ""

    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                print(f"Skipping invalid line {line_number}: {error}")

    return records


def main():
    records = load_jsonl(INPUT_FILE)

    cleaned_records = []
    excluded_records = []
    seen_titles = set()

    for record in records:
        title = str(record.get("title", "")).strip()

        if not title:
            excluded_records.append({
                "title": "Missing title",
                "reason": "Missing article title"
            })
            continue

        if title.casefold() in {
            item.casefold() for item in EXCLUDED_TITLES
        }:
            excluded_records.append({
                "title": title,
                "reason": "Manually excluded for topic relevance"
            })
            continue

        normalized_title = title.casefold()

        if normalized_title in seen_titles:
            excluded_records.append({
                "title": title,
                "reason": "Duplicate article title"
            })
            continue

        text = clean_text(record.get("text", ""))

        if len(text) < 300:
            excluded_records.append({
                "title": title,
                "reason": "Article text too short"
            })
            continue

        # Preserve the original metadata and source information.
        cleaned_record = record.copy()
        cleaned_record["title"] = title
        cleaned_record["text"] = text
        cleaned_record["cleaning_status"] = "passed_basic_cleaning"

        cleaned_records.append(cleaned_record)
        seen_titles.add(normalized_title)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        for record in cleaned_records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    subject_counts = Counter(
        record.get("subject_name", "Unknown")
        for record in cleaned_records
    )

    report = {
        "input_file": str(INPUT_FILE.relative_to(BASE_DIR)),
        "output_file": str(OUTPUT_FILE.relative_to(BASE_DIR)),
        "original_article_count": len(records),
        "cleaned_article_count": len(cleaned_records),
        "excluded_article_count": len(excluded_records),
        "excluded_articles": excluded_records,
        "articles_per_subject": dict(subject_counts),
        "cleaning_operations": [
            "Whitespace normalization",
            "Duplicate title removal",
            "Short article removal",
            "Manual relevance exclusions"
        ]
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\n========== EDUROUTE DATA CLEANING ==========")
    print("Original articles:", len(records))
    print("Cleaned articles:", len(cleaned_records))
    print("Excluded articles:", len(excluded_records))

    print("\nExcluded titles:")
    for item in excluded_records:
        print(f"- {item['title']}: {item['reason']}")

    print("\nArticles per subject:")
    for subject, count in sorted(subject_counts.items()):
        print(f"- {subject}: {count}")

    print("\nClean dataset:", OUTPUT_FILE)
    print("Cleaning report:", REPORT_FILE)
    print("\nDATA CLEANING COMPLETED")


if __name__ == "__main__":
    main()
