
import json
import re
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed" / "wikipedia"
    / "aiml_wikipedia_combined.jsonl"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed" / "wikipedia"
    / "aiml_wikipedia_v3.jsonl"
)

REPORT_FILE = (
    ROOT / "data" / "metadata"
    / "aiml_v3_preparation_report.json"
)

# These articles are unrelated to the intended AIML concepts.
EXCLUDED_TITLES = {
    "shap",
    "transformers",
}

# Remove these sections and everything following them.
REFERENCE_HEADINGS = re.compile(
    r"={2,}\s*(?:"
    r"references|"
    r"further reading|"
    r"external links|"
    r"bibliography|"
    r"notes"
    r")\s*={2,}",
    re.IGNORECASE,
)


def clean_article_text(text):
    text = text.strip()

    match = REFERENCE_HEADINGS.search(text)

    if match:
        text = text[:match.start()]

    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    print("\n========== EDUROUTE AI V3 DATA PREPARATION ==========\n")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    articles = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                articles.append(json.loads(line))

    print("Original articles:", len(articles))

    cleaned_articles = []
    excluded_articles = []
    shortened_articles = []

    for article in articles:
        title = article.get("title", "").strip()

        if title.lower() in EXCLUDED_TITLES:
            excluded_articles.append(title)
            continue

        original_text = article.get("text", "")
        cleaned_text = clean_article_text(original_text)

        if len(cleaned_text) < 300:
            excluded_articles.append(
                f"{title} (insufficient text after cleanup)"
            )
            continue

        updated = article.copy()
        updated["text"] = cleaned_text
        updated["content_cleanup_status"] = (
            "reference_sections_removed"
            if len(cleaned_text) < len(original_text.strip())
            else "no_reference_section_detected"
        )

        if len(cleaned_text) < len(original_text.strip()):
            shortened_articles.append(title)

        cleaned_articles.append(updated)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for article in cleaned_articles:
            file.write(
                json.dumps(article, ensure_ascii=False) + "\n"
            )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_file": str(INPUT_FILE.relative_to(ROOT)),
        "output_file": str(OUTPUT_FILE.relative_to(ROOT)),
        "original_articles": len(articles),
        "final_articles": len(cleaned_articles),
        "excluded_articles": excluded_articles,
        "articles_with_reference_sections_removed": shortened_articles,
        "subject_distribution": dict(
            Counter(
                article.get("subject_id", "UNKNOWN")
                for article in cleaned_articles
            )
        ),
        "unique_titles": len({
            article["title"].lower()
            for article in cleaned_articles
        }),
    }

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\n========== V3 PREPARATION SUMMARY ==========")
    print("Original articles:", len(articles))
    print("Final articles:", len(cleaned_articles))
    print("Excluded articles:", len(excluded_articles))
    print(
        "Articles with reference sections removed:",
        len(shortened_articles)
    )
    print("Unique titles:", report["unique_titles"])

    print("\nExcluded:")
    for title in excluded_articles:
        print("-", title)

    print("\nOutput:", OUTPUT_FILE)
    print("Report:", REPORT_FILE)

    print("\nV3 ARTICLE DATASET PREPARATION COMPLETE")


if __name__ == "__main__":
    main()
