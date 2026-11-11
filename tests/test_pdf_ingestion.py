import pytest

pymupdf = pytest.importorskip("pymupdf")

from src.data_collection.pdf_ingestion import ingest_pdf  # noqa: E402

BODY = ("A convolutional layer applies learned filters across the input image to produce "
        "feature maps. Each filter detects a local pattern such as an edge. ") * 6


def make_pdf(path):
    document = pymupdf.open()
    page = document.new_page()
    y = 72
    for size, text in [(18, "Unit 3: Convolutional Networks"), (11, BODY),
                       (14, "Pooling layers"), (11, BODY.replace("convolutional layer", "pooling layer")),
                       (14, "References"), (11, "LeCun, Y. et al. (1998). Proc. IEEE. doi:10.1109/5.726791")]:
        rect = pymupdf.Rect(72, y, 520, y + (40 if size > 11 else 260))
        page.insert_textbox(rect, text, fontsize=size, fontname="hebo" if size > 11 else "helv")
        y = rect.y1 + 8
    document.save(str(path))


def test_pdf_ingestion_produces_v4_record(tmp_path):
    pdf = tmp_path / "unit3.pdf"
    make_pdf(pdf)
    record = ingest_pdf(pdf, "Unit 3 notes", "AIML005", "CNN", "CC BY 4.0", author="Dept. AIML")
    assert record["sections"] and "feature maps" in record["text"]
    assert "doi:10.1109" not in record["text"]          # reference section dropped
    assert record["license_note"] == "CC BY 4.0" and len(record["sha256"]) == 64
    assert record["subject_name"] == "Deep Learning"


def test_pdf_ingestion_requires_license(tmp_path):
    pdf = tmp_path / "unit3.pdf"
    make_pdf(pdf)
    with pytest.raises(ValueError, match="licence"):
        ingest_pdf(pdf, "Unit 3 notes", "AIML005", "CNN", "  ")
