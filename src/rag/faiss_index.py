
import json
import time
import numpy as np
import faiss

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# V3 input files
EMBEDDINGS_FILE = (
    BASE_DIR / "models" / "embeddings"
    / "aiml_embeddings_v3.npy"
)

METADATA_FILE = (
    BASE_DIR / "data" / "metadata"
    / "aiml_embeddings_metadata_v3.jsonl"
)

CHUNKS_FILE = (
    BASE_DIR / "data" / "processed" / "wikipedia"
    / "aiml_chunks_v3.jsonl"
)

# V3 output files
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

INDEX_FILE = VECTOR_STORE_DIR / "aiml_faiss_v3.index"

INDEX_METADATA_FILE = (
    VECTOR_STORE_DIR / "aiml_faiss_v3_metadata.jsonl"
)

REPORT_FILE = (
    BASE_DIR / "data" / "metadata"
    / "aiml_faiss_report_v3.json"
)

EXPECTED_DIMENSION = 384
EXPECTED_VECTORS = 2410


def load_jsonl(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
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


def main():
    print("\n========== EDUROUTE AI FAISS INDEXING V3 ==========")

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {EMBEDDINGS_FILE}"
        )

    print("Loading V3 embedding matrix...")
    embeddings = np.load(EMBEDDINGS_FILE)

    if embeddings.ndim != 2:
        raise ValueError(
            "Embedding matrix must be two-dimensional."
        )

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    total_vectors, dimension = embeddings.shape

    print("Embedding shape:", embeddings.shape)

    if total_vectors != EXPECTED_VECTORS:
        raise ValueError(
            f"Expected {EXPECTED_VECTORS} vectors, "
            f"found {total_vectors}."
        )

    if dimension != EXPECTED_DIMENSION:
        raise ValueError(
            f"Expected {EXPECTED_DIMENSION} dimensions, "
            f"found {dimension}."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings contain invalid numerical values."
        )

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError(
            "Embeddings are not normalized."
        )

    print("Loading V3 metadata...")
    metadata = load_jsonl(METADATA_FILE)

    if len(metadata) != total_vectors:
        raise ValueError(
            f"Vector count ({total_vectors}) and metadata "
            f"count ({len(metadata)}) do not match."
        )

    vector_ids = [
        item.get("vector_id") for item in metadata
    ]

    if vector_ids != list(range(total_vectors)):
        raise ValueError(
            "Metadata vector IDs are missing or out of order."
        )

    chunk_ids = [
        item.get("chunk_id") for item in metadata
    ]

    if (
        any(not chunk_id for chunk_id in chunk_ids)
        or len(set(chunk_ids)) != total_vectors
    ):
        raise ValueError(
            "Duplicate or missing chunk IDs detected."
        )

    print("Loading V3 chunk dataset...")
    chunks = load_jsonl(CHUNKS_FILE)

    if len(chunks) != total_vectors:
        raise ValueError(
            "Chunk count does not match embedding count."
        )

    for index_number, (chunk, item) in enumerate(
        zip(chunks, metadata)
    ):
        if chunk.get("chunk_id") != item.get("chunk_id"):
            raise ValueError(
                f"Metadata order mismatch at vector {index_number}."
            )

    print("Metadata order verified successfully.")

    VECTOR_STORE_DIR.mkdir(
        parents=True, exist_ok=True
    )

    print("\nCreating FAISS IndexFlatIP...")
    start_time = time.time()

    # Normalized vectors + inner product = cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    elapsed_time = round(
        time.time() - start_time, 3
    )

    if index.ntotal != total_vectors:
        raise RuntimeError(
            "FAISS index count does not match embedding count."
        )

    print("Saving V3 FAISS index...")
    faiss.write_index(index, str(INDEX_FILE))

    with INDEX_METADATA_FILE.open(
        "w", encoding="utf-8"
    ) as file:
        for item in metadata:
            file.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )

    report = {
        "dataset_version": "V3",
        "index_type": "IndexFlatIP",
        "similarity_metric": "cosine_similarity",
        "total_vectors": int(index.ntotal),
        "vector_dimension": int(dimension),
        "embedding_model": (
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        "embedding_normalized": True,
        "metadata_order_verified": True,
        "chunk_dataset": str(
            CHUNKS_FILE.relative_to(BASE_DIR)
        ),
        "index_file": str(
            INDEX_FILE.relative_to(BASE_DIR)
        ),
        "metadata_file": str(
            INDEX_METADATA_FILE.relative_to(BASE_DIR)
        ),
        "indexing_time_seconds": elapsed_time
    }

    REPORT_FILE.write_text(
        json.dumps(
            report, indent=4, ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # Reload saved index to verify integrity.
    loaded_index = faiss.read_index(str(INDEX_FILE))

    if (
        loaded_index.ntotal != total_vectors
        or loaded_index.d != EXPECTED_DIMENSION
    ):
        raise RuntimeError(
            "Saved FAISS index verification failed."
        )

    print("\n========== V3 FAISS INDEX SUMMARY ==========")
    print("Index type:", type(index).__name__)
    print("Similarity metric: Cosine similarity")
    print("Total indexed vectors:", index.ntotal)
    print("Vector dimension:", index.d)
    print("Metadata order verified: True")
    print("Saved index reload verified: True")
    print("Indexing time:", elapsed_time, "seconds")

    print("\nFAISS index:", INDEX_FILE)
    print("Index metadata:", INDEX_METADATA_FILE)
    print("Index report:", REPORT_FILE)

    print("\nFAISS INDEXING V3 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
