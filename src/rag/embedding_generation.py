
import json
import time
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[2]

CHUNKS_FILE = (
    BASE_DIR / "data" / "processed" / "wikipedia"
    / "aiml_chunks_v3.jsonl"
)

EMBEDDINGS_DIR = BASE_DIR / "models" / "embeddings"

EMBEDDINGS_FILE = (
    EMBEDDINGS_DIR / "aiml_embeddings_v3.npy"
)

METADATA_FILE = (
    BASE_DIR / "data" / "metadata"
    / "aiml_embeddings_metadata_v3.jsonl"
)

REPORT_FILE = (
    BASE_DIR / "data" / "metadata"
    / "aiml_embedding_report_v3.json"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 32
EXPECTED_DIMENSION = 384
EXPECTED_CHUNKS = 2410


def load_chunks(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Chunk dataset not found: {file_path}"
        )

    chunks = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {error}"
                ) from error

    return chunks


def main():
    print("\n========== EDUROUTE AI EMBEDDING GENERATION V3 ==========\n")

    chunks = load_chunks(CHUNKS_FILE)

    if not chunks:
        raise ValueError("No chunks found.")

    if len(chunks) != EXPECTED_CHUNKS:
        raise ValueError(
            f"Expected {EXPECTED_CHUNKS} chunks, "
            f"received {len(chunks)}."
        )

    texts = [chunk["text"] for chunk in chunks]

    if any(not text.strip() for text in texts):
        raise ValueError("Empty text found in chunk dataset.")

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    for chunk in chunks:
        if not chunk.get("article_title"):
            raise ValueError(
                f"Missing article title: {chunk['chunk_id']}"
            )

        if not chunk.get("source_url"):
            raise ValueError(
                f"Missing source URL: {chunk['chunk_id']}"
            )

    print("Total chunks:", len(chunks))
    print("Loading embedding model:", MODEL_NAME)

    start_time = time.time()

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    print("Model loaded successfully.")
    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    generation_time = round(
        time.time() - start_time, 2
    )

    # Validate embedding matrix.
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D matrix.")

    if embeddings.shape[0] != len(chunks):
        raise ValueError(
            "Embedding count does not match chunk count."
        )

    if embeddings.shape[1] != EXPECTED_DIMENSION:
        raise ValueError(
            f"Expected {EXPECTED_DIMENSION} dimensions, "
            f"received {embeddings.shape[1]}."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "NaN or infinite values found in embeddings."
        )

    vector_norms = np.linalg.norm(
        embeddings, axis=1
    )

    if not np.allclose(
        vector_norms, 1.0, atol=1e-4
    ):
        raise ValueError(
            "Embedding normalization failed."
        )

    EMBEDDINGS_DIR.mkdir(
        parents=True, exist_ok=True
    )

    METADATA_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    # Save embedding matrix.
    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )

    # Save metadata in exactly the same order.
    with METADATA_FILE.open(
        "w", encoding="utf-8"
    ) as file:

        for index, chunk in enumerate(chunks):

            metadata = {
                "vector_id": index,
                "chunk_id": chunk["chunk_id"],
                "article_title": chunk["article_title"],
                "subject_id": chunk.get("subject_id", ""),
                "subject_name": chunk.get("subject_name", ""),
                "topic": chunk.get("topic", ""),
                "discipline": chunk.get("discipline", ""),
                "source": chunk.get("source", ""),
                "source_url": chunk.get("source_url", ""),
                "page_id": chunk.get("page_id"),
                "revision_id": chunk.get("revision_id"),
                "revision_timestamp": chunk.get(
                    "revision_timestamp"
                ),
                "review_status": chunk.get(
                    "review_status", "unknown"
                ),
                "text": chunk["text"]
            }

            file.write(
                json.dumps(
                    metadata,
                    ensure_ascii=False
                ) + "\n"
            )

    report = {
        "dataset_version": "V3",
        "model_name": MODEL_NAME,
        "device": "cpu",
        "total_chunks": len(chunks),
        "embedding_count": int(embeddings.shape[0]),
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_dtype": str(embeddings.dtype),
        "batch_size": BATCH_SIZE,
        "normalized": True,
        "similarity_metric_recommended": "cosine_similarity",
        "generation_time_seconds": generation_time,
        "review_status_distribution": {},
        "embedding_file": str(
            EMBEDDINGS_FILE.relative_to(BASE_DIR)
        ),
        "metadata_file": str(
            METADATA_FILE.relative_to(BASE_DIR)
        )
    }

    for chunk in chunks:
        status = chunk.get(
            "review_status", "unknown"
        )

        report["review_status_distribution"][status] = (
            report["review_status_distribution"].get(status, 0) + 1
        )

    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=4,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n========== V3 EMBEDDING SUMMARY ==========")
    print("Embedding model:", MODEL_NAME)
    print("Total vectors:", embeddings.shape[0])
    print("Vector dimensions:", embeddings.shape[1])
    print("Data type:", embeddings.dtype)
    print("Normalized:", True)
    print("Generation time:", generation_time, "seconds")

    print("\nReview status distribution:")
    for status, count in report[
        "review_status_distribution"
    ].items():
        print(f"{status}: {count}")

    print("\nSaved embeddings:", EMBEDDINGS_FILE)
    print("Saved metadata:", METADATA_FILE)
    print("Saved report:", REPORT_FILE)

    print("\nV3 EMBEDDING GENERATION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
