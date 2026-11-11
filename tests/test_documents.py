"""Uploaded-document reading and search (no language model needed)."""

import io
import zipfile

import numpy as np
import pytest

from src.rag.session_document import (DocumentError, build_session_document, read_document,
                                      refers_to_document, search_document)


class WordCountEmbedder:
    """Deterministic stand-in: a bag of a few topic words."""
    VOCAB = ["carnot", "efficiency", "transistor", "amplifier", "concrete", "curing", "engine", "heat"]

    def encode(self, texts, **_):
        v = np.array([[t.lower().count(w) for w in self.VOCAB] for t in texts], dtype=np.float32) + 1e-3
        return v / np.linalg.norm(v, axis=1, keepdims=True)


PAGES = [
    "The Carnot cycle is an ideal heat engine cycle. Its efficiency depends only on the "
    "temperatures of the hot and cold reservoirs. " * 6,
    "A transistor amplifier uses a small base current to control a larger collector current. "
    "Common emitter amplifiers give voltage gain. " * 6,
    "Curing of concrete keeps it moist so that hydration continues and the concrete gains strength. " * 6,
]


def make_pdf(pages):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    for text in pages:
        y = 800
        words = text.split()
        for i in range(0, len(words), 12):
            c.drawString(40, y, " ".join(words[i:i + 12]))
            y -= 14
        c.showPage()
    c.save()
    return buffer.getvalue()


def make_docx(paragraphs):
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    xml = ('<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxmlformats.org/'
           f'wordprocessingml/2006/main"><w:body>{body}</w:body></w:document>')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("word/document.xml", xml)
    return buffer.getvalue()


def test_pdf_pages_are_read_and_cited_by_page():
    doc = build_session_document("notes.pdf", make_pdf(PAGES), WordCountEmbedder())
    assert doc.pages == 3 and doc.passages
    hits = search_document(doc, "What is the efficiency of the Carnot cycle?", WordCountEmbedder(), top_k=2)
    assert hits[0]["article_title"] == "notes.pdf (p. 1)" and hits[0]["source_url"] is None
    hits = search_document(doc, "How does a transistor amplifier work?", WordCountEmbedder(), top_k=1)
    assert hits[0]["page"] == 2


def test_docx_and_text_files():
    pages = read_document("notes.docx", make_docx(["Curing of concrete matters.", "Keep it moist for 7 days."]))
    assert "Keep it moist" in pages[0]
    assert "Carnot" in read_document("notes.txt", PAGES[0].encode("utf-8"))[0]


def test_summary_questions_cover_the_whole_document():
    doc = build_session_document("notes.pdf", make_pdf(PAGES), WordCountEmbedder())
    hits = search_document(doc, "Summarise this document", WordCountEmbedder(), top_k=4)
    assert {h["page"] for h in hits} == {1, 2, 3}       # spread, not just the best match


def test_unreadable_documents_explain_why():
    with pytest.raises(DocumentError, match="No readable text"):
        read_document("scan.pdf", make_pdf([" "]))
    with pytest.raises(DocumentError, match="Unsupported"):
        read_document("slides.pptx", b"x")


def test_document_references():
    assert refers_to_document("Summarise this document")
    assert refers_to_document("What does the uploaded pdf say about curing?")
    assert not refers_to_document("Explain the Rankine cycle")


def test_numbers_in_the_question_pick_the_right_section():
    pages = ["Experiment 3: Ohm's law with a 220 ohm resistor and a regulated supply. " * 4,
             "Experiment 4: series and parallel resistor networks with a multimeter. " * 4]
    doc = build_session_document("manual.pdf", make_pdf(pages), WordCountEmbedder())
    hit = search_document(doc, "Which resistor is used in experiment 3?", WordCountEmbedder(), top_k=1)[0]
    assert hit["page"] == 1
