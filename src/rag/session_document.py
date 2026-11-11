"""
Documents a student uploads into the chat (lecture notes, a textbook chapter,
a question paper): read, split into passages, embedded and searched alongside
the knowledge base for the rest of the conversation.

Supported: PDF (per page), Word .docx, plain text / Markdown. The document lives
only in the student's session (memory); it is never written to disk or logs
(report §21).
"""

import io
import re
import zipfile
from dataclasses import dataclass, field

import numpy as np

DOCUMENT_TYPES = ["pdf", "docx", "txt", "md"]
MAX_DOCUMENT_BYTES = 15 * 1024 * 1024
MAX_PAGES = 300
WINDOW_WORDS, OVERLAP_WORDS = 170, 30

# "summarise this document", "what is this pdf about", "key points of the notes"
SUMMARY_REQUEST = re.compile(
    r"\b(summar(y|ise|ize)|overview|key points|main points|what is (this|the) "
    r"(document|pdf|file|paper|notes?) about|outline of)\b", re.IGNORECASE)
DOCUMENT_REFERENCE = re.compile(
    r"\b(this|the|my|uploaded|attached|above) (document|pdf|file|paper|notes?|chapter|"
    r"question paper|handout|slides?)\b|\b(in|from|according to) (it|the document)\b", re.IGNORECASE)


class DocumentError(ValueError):
    pass


@dataclass
class SessionDocument:
    name: str
    pages: int
    passages: list = field(default_factory=list)   # dicts in the retriever's result format
    vectors: np.ndarray | None = None

    @property
    def words(self):
        return sum(len(p["text"].split()) for p in self.passages)


def _pdf_pages(data):
    import pymupdf
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            if doc.needs_pass:
                raise DocumentError("The PDF is password-protected.")
            return [page.get_text() for page in doc][:MAX_PAGES]
    except DocumentError:
        raise
    except Exception as error:
        raise DocumentError(f"Could not read the PDF: {error}") from error


def _docx_pages(data):
    """Paragraph text from word/document.xml; a .docx has no fixed pages, so
    every ~400 words is treated as one 'page' for references."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
    except Exception as error:
        raise DocumentError(f"Could not read the Word document: {error}") from error
    paragraphs = []
    for para in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.DOTALL):
        text = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", para, flags=re.DOTALL))
        text = (text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                .replace("&quot;", '"').replace("&apos;", "'"))
        if text.strip():
            paragraphs.append(text.strip())
    return _group_into_pages(paragraphs)


def _text_pages(data):
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    return _group_into_pages([p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()])


def _group_into_pages(paragraphs, words_per_page=400):
    pages, current, count = [], [], 0
    for para in paragraphs:
        current.append(para)
        count += len(para.split())
        if count >= words_per_page:
            pages.append("\n".join(current))
            current, count = [], 0
    if current:
        pages.append("\n".join(current))
    return pages[:MAX_PAGES]


def read_document(name, data):
    """Return the document's text per page. Raises DocumentError with a message
    that can be shown to the student."""
    if len(data) > MAX_DOCUMENT_BYTES:
        raise DocumentError(f"The file is larger than {MAX_DOCUMENT_BYTES // (1024 * 1024)} MB.")
    kind = name.rsplit(".", 1)[-1].lower()
    if kind == "pdf":
        pages = _pdf_pages(data)
    elif kind == "docx":
        pages = _docx_pages(data)
    elif kind in ("txt", "md"):
        pages = _text_pages(data)
    else:
        raise DocumentError(f"Unsupported file type .{kind}; use PDF, Word (.docx) or text.")
    pages = [re.sub(r"[ \t]+", " ", p).strip() for p in pages]
    if not any(len(p.split()) >= 5 for p in pages):
        raise DocumentError("No readable text was found. A scanned PDF needs to be converted "
                            "with OCR first, or attach the pages as images instead.")
    return pages


def build_session_document(name, data, embedding_model):
    pages = read_document(name, data)
    passages = []
    step = WINDOW_WORDS - OVERLAP_WORDS
    for page_no, page in enumerate(pages, start=1):
        words = page.split()
        for start in range(0, max(1, len(words) - OVERLAP_WORDS), step):
            chunk = " ".join(words[start:start + WINDOW_WORDS])
            if len(chunk.split()) < 12:
                continue
            passages.append({
                "chunk_id": f"DOC_{len(passages):04d}",
                "article_title": f"{name} (p. {page_no})",
                "section": None, "source_url": None, "source": "uploaded document",
                "page": page_no, "text": chunk,
            })
    if not passages:
        raise DocumentError("The document has too little text to search.")
    vectors = embedding_model.encode([p["text"] for p in passages], convert_to_numpy=True,
                                     normalize_embeddings=True, show_progress_bar=False)
    return SessionDocument(name=name, pages=len(pages), passages=passages, vectors=vectors)


def search_document(document, question, embedding_model, top_k=4):
    """Most relevant passages; for summary questions, passages spread over the
    whole document instead (a summary needs coverage, not one similar passage)."""
    if document is None or not document.passages:
        return []
    if SUMMARY_REQUEST.search(question or ""):
        n = len(document.passages)
        picks = sorted({round(i * (n - 1) / max(1, min(top_k + 2, n) - 1)) for i in range(min(top_k + 2, n))})
        chosen = [(i, None) for i in picks]
    else:
        q = embedding_model.encode([question], convert_to_numpy=True, normalize_embeddings=True,
                                   show_progress_bar=False)[0]
        semantic = document.vectors @ q
        # Exact words and numbers matter in notes ("experiment 3", "Table 2.1"):
        # the embedding barely separates "experiment 3" from "experiment 4".
        terms = _terms(question)
        lexical = np.array([len(terms & _terms(p["text"])) / len(terms) if terms else 0.0
                            for p in document.passages])
        scores = 0.75 * semantic + 0.25 * lexical
        chosen = [(int(i), float(semantic[i])) for i in np.argsort(scores)[::-1][:top_k]]
    return [{**document.passages[i], "similarity_score": round(score, 4) if score is not None else None,
             "rank": rank} for rank, (i, score) in enumerate(chosen, start=1)]


_STOP = set("the a an of in on to for and or is are was were what which how why does do with by "
            "this that these those it its from as at be used use".split())


def _terms(text):
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in _STOP and
            (len(w) > 2 or w.isdigit())}


def refers_to_document(question):
    return bool(question and (DOCUMENT_REFERENCE.search(question) or SUMMARY_REQUEST.search(question)))
