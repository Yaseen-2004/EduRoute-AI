"""
Academic PDF ingestion (report §6 "Academic document ingestion", §11.1, §21).

Converts a permitted PDF (lecture notes, open textbook chapter, syllabus) into
the same section-aware article record used by the V4 knowledge base, so it
flows through src/rag/build_index_v4.py unchanged.

Provenance is mandatory: a document is refused unless its licence/permission
and subject are given (report §11.1: title, source, author/publisher,
date, licence must be preserved; §21: only permitted documents).

Usage:
    venv\\Scripts\\python -m src.data_collection.pdf_ingestion notes.pdf ^
        --title "Deep Learning Unit 3 Notes" --subject-id AIML005 --topic CNN ^
        --license "Course notes, used with instructor permission" --author "Dept. of AIML"
Output: data/processed/documents/aiml_documents.jsonl (one record per PDF, upserted)
Then rebuild the index with:  python -m src.rag.build_index_v4 --include-documents
"""

import argparse
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from config.settings import BASE_DIR
from src.rag.text_cleaning import clean_article
from src.utils.catalogue_loader import load_catalogue
from src.utils.io import load_jsonl, write_jsonl

DOCUMENTS_FILE = BASE_DIR / "data" / "processed" / "documents" / "aiml_documents.jsonl"
MAX_PDF_BYTES = 50 * 1024 * 1024
MIN_TEXT_CHARS = 300


def pdf_to_markdown(path):
    """Extract text with headings preserved as markdown (#, ##, ...)."""
    try:
        import pymupdf4llm
        return pymupdf4llm.to_markdown(str(path), show_progress=False)
    except ImportError:
        import pymupdf
        with pymupdf.open(str(path)) as document:
            return "\n\n".join(page.get_text() for page in document)


def markdown_to_wiki_headings(markdown):
    """Map markdown headings to the '== Heading ==' form the section cleaner parses."""
    def replace(match):
        level = min(len(match.group(1)) + 1, 6)
        heading = re.sub(r"[*_`]", "", match.group(2)).strip()
        return f"\n{'=' * level} {heading} {'=' * level}\n"
    text = re.sub(r"^(#{1,5})\s+(.+?)\s*#*\s*$", replace, markdown, flags=re.MULTILINE)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)          # image placeholders
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub(r"^\s*-{3,}\s*$", "", text, flags=re.MULTILINE)
    return text


def ingest_pdf(path, title, subject_id, topic, license_note, author=None, source_url=None,
               published=None):
    path = Path(path)
    if not path.exists() or path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {path}")
    if path.stat().st_size > MAX_PDF_BYTES:
        raise ValueError("PDF exceeds the 50 MB limit.")
    if not license_note or not license_note.strip():
        raise ValueError("A licence/permission note is required for every ingested document.")

    subjects = {s["id"]: s for s in load_catalogue()["subjects"]}
    if subject_id not in subjects:
        raise ValueError(f"Unknown subject_id {subject_id}")

    raw = path.read_bytes()
    text = markdown_to_wiki_headings(pdf_to_markdown(path))
    sections, dropped = clean_article(text)
    body = "\n\n".join(s["text"] for s in sections)
    if len(body) < MIN_TEXT_CHARS:
        raise ValueError(
            "Too little text extracted. The PDF may be scanned images; run OCR first."
        )

    digest = hashlib.sha256(raw).hexdigest()
    return {
        "title": title,
        "text": body,
        "sections": [{"heading_path": s["heading_path"], "text": s["text"]} for s in sections],
        "source": "Uploaded document",
        "url": source_url or f"document://{digest[:16]}/{path.name}",
        "page_id": None,
        "revision_id": digest[:16],
        "revision_timestamp": published,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "license_note": license_note.strip(),
        "author": author,
        "file_name": path.name,
        "sha256": digest,
        "discipline": subject_id.rstrip("0123456789"),
        "subject_id": subject_id,
        "subject_name": subjects[subject_id]["name"],
        "topic": topic,
        "content_cleanup_status": "pdf_section_aware",
        "dropped_sections": dropped["sections"],
    }


def upsert(record):
    records = load_jsonl(DOCUMENTS_FILE) if DOCUMENTS_FILE.exists() else []
    records = [r for r in records if r["sha256"] != record["sha256"] and r["title"] != record["title"]]
    records.append(record)
    write_jsonl(DOCUMENTS_FILE, records)
    return len(records)


def main():
    parser = argparse.ArgumentParser(description="Ingest a permitted academic PDF")
    parser.add_argument("pdf")
    parser.add_argument("--title", required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--license", required=True, help="Licence or permission statement")
    parser.add_argument("--author")
    parser.add_argument("--source-url")
    parser.add_argument("--published", help="Publication/revision date, e.g. 2025-08-01")
    args = parser.parse_args()

    record = ingest_pdf(args.pdf, args.title, args.subject_id, args.topic, args.license,
                        args.author, args.source_url, args.published)
    total = upsert(record)
    print(f"Ingested '{record['title']}': {len(record['sections'])} sections, "
          f"{len(record['text'])} characters. Documents in store: {total}")
    print("Rebuild the index: python -m src.rag.build_index_v4 --include-documents")


if __name__ == "__main__":
    main()
