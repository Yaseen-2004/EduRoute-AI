
import json
import statistics
from pathlib import Path
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR / "data" / "processed" / "wikipedia"
    / "aiml_wikipedia_v3.jsonl"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "processed" / "wikipedia"
    / "aiml_chunks_v3.jsonl"
)

REPORT_FILE = (
    BASE_DIR / "data" / "metadata"
    / "aiml_chunking_report_v3.json"
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 220
CHUNK_OVERLAP = 30

EXPECTED_ARTICLES = 92


def load_jsonl(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {file_path}"
        )

    records = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {error}"
                ) from error

    return records


def split_into_chunks(text, tokenizer, chunk_size, overlap):
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False
    )

    chunks = []

    if not token_ids:
        return chunks

    step = chunk_size - overlap

    for start in range(0, len(token_ids), step):
        end = min(start + chunk_size, len(token_ids))

        current_tokens = token_ids[start:end]

        chunk_text = tokenizer.decode(
            current_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()

        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "token_count": len(current_tokens)
            })

        if end >= len(token_ids):
            break

    return chunks


def validate_chunks(chunks):
    if not chunks:
        raise ValueError("No chunks were generated.")

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    for chunk in chunks:
        if not chunk["text"].strip():
            raise ValueError(
                f"Empty chunk detected: {chunk['chunk_id']}"
            )

        if chunk["token_count"] > CHUNK_SIZE:
            raise ValueError(
                f"Chunk exceeds token limit: {chunk['chunk_id']}"
            )

        if not chunk["article_title"]:
            raise ValueError("Missing article title.")

        if not chunk["source_url"]:
            raise ValueError(
                f"Missing source URL: {chunk['chunk_id']}"
            )


def main():
    print("\n========== EDUROUTE AI CHUNKING V3 ==========\n")

    records = load_jsonl(INPUT_FILE)

    print("Loaded articles:", len(records))

    if len(records) != EXPECTED_ARTICLES:
        raise ValueError(
            f"Expected {EXPECTED_ARTICLES} articles, "
            f"but found {len(records)}. "
            "Check the V3 dataset before continuing."
        )

    titles = [
        record.get("title", "").strip().lower()
        for record in records
    ]

    if len(titles) != len(set(titles)):
        raise ValueError("Duplicate article titles detected.")

    print("Unique article titles:", len(set(titles)))
    print("Loading tokenizer:", EMBEDDING_MODEL)

    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL
    )

    model_limit = tokenizer.model_max_length

    print("Tokenizer maximum sequence length:", model_limit)

    if CHUNK_SIZE + tokenizer.num_special_tokens_to_add() > model_limit:
        raise ValueError(
            "Chunk size exceeds the model's supported sequence length."
        )

    all_chunks = []
    article_chunk_counts = {}

    for article_number, record in enumerate(records, start=1):

        article_title = record.get("title", "Unknown")
        article_text = record.get("text", "")

        if not article_text.strip():
            print("[SKIPPED EMPTY]", article_title)
            continue

        text_chunks = split_into_chunks(
            article_text,
            tokenizer,
            CHUNK_SIZE,
            CHUNK_OVERLAP
        )

        article_chunk_counts[article_title] = len(text_chunks)

        for chunk_number, chunk in enumerate(text_chunks, start=1):

            chunk_record = {
                "chunk_id": (
                    f"AIML_V3_{article_number:03d}_"
                    f"CHUNK_{chunk_number:04d}"
                ),
                "article_title": article_title,
                "subject_id": record.get("subject_id", ""),
                "subject_name": record.get("subject_name", ""),
                "topic": record.get("topic", ""),
                "discipline": record.get("discipline", "AIML"),
                "source": record.get("source", "Wikipedia"),
                "source_url": record.get("url", ""),
                "page_id": record.get("page_id"),
                "revision_id": record.get("revision_id"),
                "revision_timestamp": record.get(
                    "revision_timestamp"
                ),
                "review_status": record.get(
                    "content_cleanup_status",
                    record.get("cleaning_status", "unknown")
                ),
                "text": chunk["text"],
                "token_count": chunk["token_count"],
                "chunk_index": chunk_number,
                "total_article_chunks": len(text_chunks)
            }

            all_chunks.append(chunk_record)

        print(
            f"[{article_number}/{len(records)}] "
            f"{article_title}: {len(text_chunks)} chunks"
        )

    validate_chunks(all_chunks)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    temporary_file = OUTPUT_FILE.with_suffix(".tmp")

    with temporary_file.open("w", encoding="utf-8") as file:
        for chunk in all_chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    temporary_file.replace(OUTPUT_FILE)

    token_counts = [
        chunk["token_count"] for chunk in all_chunks
    ]

    review_distribution = {}

    for chunk in all_chunks:
        status = chunk["review_status"]
        review_distribution[status] = (
            review_distribution.get(status, 0) + 1
        )

    report = {
        "dataset_version": "V3",
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "original_articles": len(records),
        "processed_articles": len(article_chunk_counts),
        "total_chunks": len(all_chunks),
        "unique_chunk_ids": len({
            chunk["chunk_id"] for chunk in all_chunks
        }),
        "minimum_chunk_tokens": min(token_counts),
        "maximum_chunk_tokens": max(token_counts),
        "average_chunk_tokens": round(
            statistics.mean(token_counts), 2
        ),
        "review_status_distribution": review_distribution,
        "article_chunk_counts": article_chunk_counts,
        "output_file": str(
            OUTPUT_FILE.relative_to(BASE_DIR)
        )
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n========== V3 CHUNKING SUMMARY ==========")
    print("Articles loaded:", len(records))
    print("Articles processed:", len(article_chunk_counts))
    print("Total chunks:", len(all_chunks))
    print("Unique chunk IDs:", report["unique_chunk_ids"])
    print("Chunk size:", CHUNK_SIZE)
    print("Overlap:", CHUNK_OVERLAP)
    print("Minimum tokens:", min(token_counts))
    print("Maximum tokens:", max(token_counts))
    print("Average tokens:", report["average_chunk_tokens"])

    print("\nReview status distribution:")
    for status, count in review_distribution.items():
        print(f"{status}: {count}")

    print("\nChunk dataset:", OUTPUT_FILE)
    print("Chunking report:", REPORT_FILE)

    print("\nV3 CHUNKING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
