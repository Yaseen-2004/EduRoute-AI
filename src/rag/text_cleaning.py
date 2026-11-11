"""
Section-aware cleaning for Wikipedia plain-text extracts.

V3 truncated each article at the first reference-style heading. That missed
"See also" (present in 73 of 92 articles) and was never applied to the two
articles integrated after preparation, which is why a bibliography chunk from
"Attention Is All You Need" ranked first for the Transformer query (report §19).

V4 instead parses the heading structure, drops non-content sections (and their
subsections) wherever they occur, and keeps the heading path of each surviving
section so chunks can carry section metadata.
"""

import re
import unicodedata

# Works on both raw extracts ("\n\n== Heading ==\n") and the whitespace-collapsed
# V3 text ("... end. == Heading == Start ...").
HEADING_PATTERN = re.compile(r"(?<!=)(={2,6})\s*([^=\n]{1,120}?)\s*\1(?!=)")

NON_CONTENT_SECTIONS = {
    "see also",
    "notes",
    "references",
    "external links",
    "further reading",
    "bibliography",
    "footnotes",
    "sources",
    "citations",
    "notes and references",
    "references and notes",
    "works cited",
    "explanatory notes",
    "general references",
    "literature",
    "other uses",
}

# Signals of citation/bibliography text that slipped outside a heading.
CITATION_SIGNALS = re.compile(
    r"\bretrieved\b|\bdoi\b|\bisbn\b|\barxiv\b|\bpp?\.\s*\d|"
    r"\(\d{1,2} \w+ \d{4}\)|\bvol\.|\bproceedings\b|\bjournal\b|"
    r"\bcs\.[a-z]{2}\b",
    re.IGNORECASE,
)


def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sections(text):
    """
    Split an article into sections.

    Returns a list of {"heading_path": [...], "level": int, "text": str}.
    The lead section has an empty heading path.
    """
    sections = []
    path = []  # list of (level, heading)
    cursor = 0
    current_path = []
    current_level = 1

    for match in HEADING_PATTERN.finditer(text):
        body = text[cursor:match.start()]
        sections.append({
            "heading_path": [h for _, h in current_path],
            "level": current_level,
            "text": body,
        })

        level = len(match.group(1))
        heading = match.group(2).strip()

        while path and path[-1][0] >= level:
            path.pop()
        path.append((level, heading))

        current_path = list(path)
        current_level = level
        cursor = match.end()

    sections.append({
        "heading_path": [h for _, h in current_path],
        "level": current_level,
        "text": text[cursor:],
    })

    for section in sections:
        section["text"] = normalize_text(section["text"])

    return sections


def is_non_content(heading_path):
    return any(h.strip().lower() in NON_CONTENT_SECTIONS for h in heading_path)


def citation_density(text):
    words = max(1, len(text.split()))
    return len(CITATION_SIGNALS.findall(text)) / words * 100


def clean_article(text, max_citation_density=4.0):
    """
    Return (kept_sections, dropped_report).

    max_citation_density: citation signals per 100 words above which a
    paragraph is treated as bibliography noise. Normal technical prose
    scores well below 1; reference lists score 8+.
    """
    kept = []
    dropped = {"sections": [], "citation_paragraphs": 0}

    for section in split_sections(text):
        if is_non_content(section["heading_path"]):
            if section["text"]:
                dropped["sections"].append(" > ".join(section["heading_path"]))
            continue

        paragraphs = [p for p in re.split(r"\n\s*\n|\n", section["text"]) if p.strip()]
        clean_paragraphs = []
        for paragraph in paragraphs:
            # "et al." is deliberately not a signal: history sections cite
            # authors in normal prose ("Graves et al. applied LSTM ...").
            if (
                len(CITATION_SIGNALS.findall(paragraph)) >= 3
                and citation_density(paragraph) > max_citation_density
            ):
                dropped["citation_paragraphs"] += 1
                continue
            clean_paragraphs.append(paragraph.strip())

        body = "\n".join(clean_paragraphs).strip()
        if body:
            kept.append({**section, "text": body})

    return kept, dropped
