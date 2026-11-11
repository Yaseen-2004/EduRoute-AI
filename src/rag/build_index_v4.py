"""
EduRoute AI - V4 chunking, embedding and FAISS indexing in one reproducible step.

Changes from V3:
  * Chunk text is sliced from the ORIGINAL article text using tokenizer offset
    mappings. V3 decoded token IDs back to text, which lowercased everything
    and produced WordPiece artifacts ("##ns", "dot - product", "�").
    That degraded text would have been shown to the LLM and in citations.
  * Chunks never cross section boundaries and carry section metadata.
  * Very short sections are merged with their neighbour so we don't index
    40-token fragments.
  * Each chunk is embedded with a short "Title > Section:" prefix
    (contextual chunk headers). The stored display text has no prefix.

Run from the project root:
    venv\\Scripts\\python -m src.rag.build_index_v4
"""

import statistics
import time
from datetime import datetime, timezone

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from config.settings import BASE_DIR, EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION
from src.utils.io import load_jsonl, write_jsonl, write_json

DOCUMENTS_FILE = BASE_DIR / "data" / "processed" / "documents" / "aiml_documents.jsonl"

# dataset key -> (input article file, file stem, chunk-id prefix)
DATASETS = {
    "v4": ("aiml_wikipedia_v4.jsonl", "aiml", "v4", "AIML_V4"),
    "engineering_v1": ("engineering_wikipedia_v1.jsonl", "engineering", "v1", "ENG_V1"),
}


def dataset_paths(dataset):
    """Artifact paths for a dataset key (the retriever uses the same scheme)."""
    source, stem, version, prefix = DATASETS[dataset]
    return {
        "input": BASE_DIR / "data" / "processed" / "wikipedia" / source,
        "chunks": BASE_DIR / "data" / "processed" / "wikipedia" / f"{stem}_chunks_{version}.jsonl",
        "embeddings": BASE_DIR / "models" / "embeddings" / f"{stem}_embeddings_{version}.npy",
        "index": BASE_DIR / "vector_store" / f"{stem}_faiss_{version}.index",
        "metadata": BASE_DIR / "vector_store" / f"{stem}_faiss_{version}_metadata.jsonl",
        "report": BASE_DIR / "data" / "metadata" / f"{stem}_index_report_{version}.json",
        "prefix": prefix,
    }

CHUNK_SIZE = 220
CHUNK_OVERLAP = 30
MIN_SECTION_TOKENS = 60
MAX_PREFIX_TOKENS = 30
BATCH_SIZE = 32


def merge_short_sections(sections, tokenizer):
    """Merge sections shorter than MIN_SECTION_TOKENS into the next one."""
    merged = []
    pending = None
    for section in sections:
        if pending is not None:
            section = {
                "heading_path": pending["heading_path"] or section["heading_path"],
                "text": pending["text"] + "\n" + section["text"],
            }
            pending = None
        n_tokens = len(tokenizer.encode(section["text"], add_special_tokens=False))
        if n_tokens < MIN_SECTION_TOKENS:
            pending = section
        else:
            merged.append(section)
    if pending is not None:
        if merged:
            merged[-1] = {
                "heading_path": merged[-1]["heading_path"],
                "text": merged[-1]["text"] + "\n" + pending["text"],
            }
        else:
            merged.append(pending)
    return merged


def _is_word_continuation(text, offsets, i):
    """True if token i continues the word of token i-1 (no gap between them)."""
    if i <= 0 or i >= len(offsets):
        return False
    prev_end = offsets[i - 1][1]
    start = offsets[i][0]
    return start == prev_end and text[start - 1:start].isalnum() and text[start:start + 1].isalnum()


def chunk_section(text, tokenizer):
    encoding = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True, truncation=False
    )
    offsets = encoding["offset_mapping"]
    n = len(offsets)
    if n == 0:
        return []

    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    start = 0
    while start < n:
        end = min(start + CHUNK_SIZE, n)
        # Do not cut in the middle of a word.
        s, e = start, end
        while s < e and _is_word_continuation(text, offsets, s):
            s += 1
        while e < n and e > s + 1 and _is_word_continuation(text, offsets, e):
            e -= 1
        chunk_text = text[offsets[s][0]:offsets[e - 1][1]].strip()
        if chunk_text:
            chunks.append({"text": chunk_text, "token_count": e - s})
        if end >= n:
            break
        start += step
    return chunks


def context_prefix(title, heading_path, tokenizer):
    prefix = title if not heading_path else f"{title} > {' > '.join(heading_path)}"
    ids = tokenizer.encode(prefix, add_special_tokens=False)
    if len(ids) > MAX_PREFIX_TOKENS:
        prefix = tokenizer.decode(ids[:MAX_PREFIX_TOKENS])
    return prefix + ": "


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="v4", choices=list(DATASETS))
    parser.add_argument("--include-documents", action="store_true",
                        help="Also index ingested PDFs (data/processed/documents)")
    args = parser.parse_args()

    paths = dataset_paths(args.dataset)
    CHUNKS_FILE, EMBEDDINGS_FILE = paths["chunks"], paths["embeddings"]
    INDEX_FILE, INDEX_METADATA_FILE, REPORT_FILE = paths["index"], paths["metadata"], paths["report"]

    print(f"\n========== EDUROUTE AI INDEX BUILD ({args.dataset}) ==========\n")
    articles = load_jsonl(paths["input"])
    if args.include_documents and DOCUMENTS_FILE.exists():
        documents = load_jsonl(DOCUMENTS_FILE)
        print("Ingested documents:", len(documents))
        articles.extend(documents)
    print("Articles:", len(articles))

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)

    chunks = []
    for article_number, article in enumerate(articles, start=1):
        sections = merge_short_sections(article["sections"], tokenizer)
        article_chunks = []
        for section in sections:
            for piece in chunk_section(section["text"], tokenizer):
                article_chunks.append((section["heading_path"], piece))

        for chunk_number, (heading_path, piece) in enumerate(article_chunks, start=1):
            chunks.append({
                "chunk_id": f"{paths['prefix']}_{article_number:04d}_CHUNK_{chunk_number:04d}",
                "article_title": article["title"],
                "section": " > ".join(heading_path) if heading_path else "Introduction",
                "subject_id": article.get("subject_id", ""),
                "subject_name": article.get("subject_name", ""),
                "topic": article.get("topic", ""),
                "discipline": article.get("discipline", "AIML"),
                # Every branch/subject that maps to this article (for filtering).
                "disciplines": article.get("disciplines") or [article.get("discipline", "AIML")],
                "subject_ids": article.get("subject_ids") or [article.get("subject_id", "")],
                "source": article.get("source", "Wikipedia"),
                "source_url": article.get("url", ""),
                "page_id": article.get("page_id"),
                "revision_id": article.get("revision_id"),
                "revision_timestamp": article.get("revision_timestamp"),
                "license_note": article.get("license_note"),
                "review_status": article.get("content_cleanup_status", "unknown"),
                "text": piece["text"],
                "embed_text": context_prefix(article["title"], heading_path, tokenizer) + piece["text"],
                "token_count": piece["token_count"],
                "chunk_index": chunk_number,
                "total_article_chunks": len(article_chunks),
            })

    ids = [c["chunk_id"] for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs.")
    if any(c["token_count"] > CHUNK_SIZE for c in chunks):
        raise ValueError("Chunk exceeds CHUNK_SIZE.")
    if any(not c["source_url"] for c in chunks):
        raise ValueError("Chunk without source URL.")

    write_jsonl(CHUNKS_FILE, chunks)
    token_counts = [c["token_count"] for c in chunks]
    print("Chunks:", len(chunks))

    print("Embedding (CPU, this takes a few minutes)...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    started = time.time()
    embeddings = model.encode(
        [c["embed_text"] for c in chunks],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    embedding_seconds = round(time.time() - started, 2)

    if embeddings.shape != (len(chunks), EMBEDDING_DIMENSION):
        raise ValueError(f"Unexpected embedding shape {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError("Non-finite embedding values.")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4):
        raise ValueError("Embeddings are not normalized.")

    EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_FILE, embeddings)

    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    index.add(embeddings)
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))

    metadata = []
    for vector_id, chunk in enumerate(chunks):
        record = {"vector_id": vector_id, **chunk}
        record.pop("embed_text")
        metadata.append(record)
    write_jsonl(INDEX_METADATA_FILE, metadata)

    reloaded = faiss.read_index(str(INDEX_FILE))
    if reloaded.ntotal != len(chunks) or reloaded.d != EMBEDDING_DIMENSION:
        raise RuntimeError("Saved FAISS index verification failed.")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": args.dataset,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "min_section_tokens": MIN_SECTION_TOKENS,
        "context_prefix": True,
        "articles": len(articles),
        "total_chunks": len(chunks),
        "minimum_chunk_tokens": min(token_counts),
        "maximum_chunk_tokens": max(token_counts),
        "average_chunk_tokens": round(statistics.mean(token_counts), 2),
        "embedding_dimension": EMBEDDING_DIMENSION,
        "embedding_seconds": embedding_seconds,
        "index_type": "IndexFlatIP",
        "similarity_metric": "cosine_similarity",
        "index_reload_verified": True,
        "files": {
            "chunks": str(CHUNKS_FILE.relative_to(BASE_DIR)),
            "embeddings": str(EMBEDDINGS_FILE.relative_to(BASE_DIR)),
            "index": str(INDEX_FILE.relative_to(BASE_DIR)),
            "metadata": str(INDEX_METADATA_FILE.relative_to(BASE_DIR)),
        },
    }
    write_json(REPORT_FILE, report)

    print(f"\n========== INDEX SUMMARY ({args.dataset}) ==========")
    for key in ("articles", "total_chunks", "minimum_chunk_tokens", "maximum_chunk_tokens",
                "average_chunk_tokens", "embedding_seconds"):
        print(f"{key}: {report[key]}")
    print("Report:", REPORT_FILE)


if __name__ == "__main__":
    main()
