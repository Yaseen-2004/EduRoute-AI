"""
Compare the subject catalogues with the AICTE model curriculum.

Sources (official PDFs from aicte.gov.in, downloaded to data/raw/aicte/):
  ug-vol1.pdf         Vol. I (2018): first year, Civil, Electrical, Mechanical, ECE, CSE
  ug-vol2.pdf         Vol. II (2018): Chemical, Metallurgical
  ug-cse-2022.pdf     revised CSE (2022)
  ug-civil-2024.pdf   revised Civil (2024)
  ug-mechanical.pdf   revised Mechanical
  ug-robotics-ai.pdf  Robotics and Artificial Intelligence Engineering
  ug-emerging.pdf     course outlines for emerging areas (AI, IoT, data science, cyber security, ...)

AICTE has no model curriculum for the other branch groups in these documents
(biotechnology, biomedical, aerospace, automobile, mechatronics, instrumentation,
agriculture, mining, petroleum, marine, ocean, textile); they are reported as
"no AICTE reference" and must be checked against a university syllabus.

Outputs (data/metadata/):
  aicte_courses.csv      every extracted AICTE course/unit with document and page
  aicte_comparison.csv   each catalogue subject -> closest AICTE course (+ similarity, status)
  aicte_gaps.csv         AICTE professional-core courses with no close catalogue subject

Matching uses the same MiniLM embeddings as the app (cosine similarity of
"subject: topics" vs the course title). Thresholds: >= 0.60 match, >= 0.45
partial. It is a screening aid for the human review in the evaluation studio,
not a verdict.

    venv\\Scripts\\python -m src.data_collection.compare_aicte
"""

import csv
import re
from collections import defaultdict

import numpy as np

from config import settings
from src.utils.catalogue_loader import load_catalogue

AICTE_DIR = settings.RAW_DATA_DIR / "aicte"
OUT_DIR = settings.METADATA_DIR
MATCH, PARTIAL = 0.60, 0.45

# Course code -> AICTE branch letters used in Vol. I/II
CODE = re.compile(r"\b(PCC|PEC|ESC|BSC|HSMC|OEC)[\s-]*([A-Z]{0,3})[\s-]*(\d{2,3})[A-Z]?\b\s*:?\s*(.*)")
LETTERS_TO_BRANCH = {"CS": "CSE", "CE": "CE", "ME": "ME", "EE": "EE", "EC": "ECE",
                     "CH": "CHE", "MT": "MET", "MM": "MET"}
DOCUMENT_BRANCH = {"ug-cse-2022": "CSE", "ug-civil-2024": "CE", "ug-mechanical": "ME",
                   "ug-robotics-ai": "ROBO"}
# Our branch groups -> AICTE reference branches (in order of preference)
REFERENCE = {
    "COMMON": ["FIRST_YEAR"], "CSE": ["CSE"], "IT": ["CSE"], "AIML": ["AI", "CSE"],
    "DS": ["DATA_SCIENCE", "CSE"], "CYBER": ["CYBER_SECURITY", "CSE"], "ECE": ["ECE"],
    "EEE": ["EE"], "EE": ["EE"], "ME": ["ME"], "CE": ["CE"], "CHE": ["CHE"], "MET": ["MET"],
    "ROBO": ["ROBO", "ROBOTICS"],
}
EMERGING = {"ARTIFICIAL INTELLIGENCE": "AI", "INTERNET OF THINGS": "IOT", "ROBOTICS": "ROBOTICS",
            "QUANTUM COMPUTING": "QUANTUM", "DATA SCIENCES": "DATA_SCIENCE", "DATA SCIENCE": "DATA_SCIENCE",
            "CYBER SECURITY": "CYBER_SECURITY", "3D PRINTING AND DESIGN": "3D_PRINTING",
            "VIRTUAL REALITY": "VR"}
ECE_CODE = re.compile(r"^EC\d{2}\s*:\s*(.+)")   # Vol. I ECE: "EC01: Electronic Devices"
NOT_A_COURSE = re.compile(
    r"\b(lab|laboratory|workshop|project|seminar|internship|elective course|introduction|"
    r"pre-?requisites?|contents|course outcomes?)\b|^design$|\b(PCC|PEC|ESC|BSC)\b", re.IGNORECASE)
UNIT = re.compile(r"^\s*\d{1,2}\.\s+([A-Z][A-Za-z0-9 ,&/()'-]{3,80}?)\s*$")


BRANCH_NAME_ONLY = re.compile(
    r"^(mechanical|civil|electrical|chemical|metallurgical|computer science( and| &)?|electronics( and| &)? "
    r"communication)( engineering)?$", re.IGNORECASE)
TYPOS = {"Complier": "Compiler"}                          # as printed in the AICTE PDFs
CONNECTOR_END = ("&", "and", "of", "-", "its", "for", "in", "to", "with", "Machine")


def normalise_title(title):
    """Key for duplicates: 'Measurements & Metrology' == 'Measurements and Metrology'."""
    key = title.lower().replace("&", " and ")
    key = re.sub(r"\b\d+l\s*:\s*\d+t\s*:\s*\d+p\b", "", key)
    return " ".join(re.findall(r"[a-z0-9]+", key))


def clean(title):
    for wrong, right in TYPOS.items():
        title = title.replace(wrong, right)
    title = re.sub(r"\b\d+L\s*:\s*\d+T\s*:\s*\d+P\b", "", title)   # "3L:0T:4P"
    title = re.sub(r"\[[^\]]*\]", "", title)            # "[0L:0T:2P 1 credit]"
    title = re.sub(r"\((New|Updated|Revised)\)", "", title)
    if title.count("(") > title.count(")"):               # wrapped "(Sci" fragment
        title = title[:title.rfind("(")]
    title = re.sub(r"\s+", " ", title).strip(" :-–")
    return title


def extract_coded(doc_name, pages):
    """Courses written as CODE[: title] with the title possibly wrapped onto following lines."""
    courses = []
    for page_no, text in enumerate(pages, start=1):
        lines = [l.strip() for l in text.splitlines()]
        for i, line in enumerate(lines):
            ece = ECE_CODE.match(line) if doc_name == "ug-vol1" else None
            if ece:
                title = clean(ece.group(1))
                if not NOT_A_COURSE.search(title) and len(title) >= 5:
                    courses.append({"branch": "ECE", "kind": "PCC", "title": title,
                                    "document": f"{doc_name}.pdf", "page": page_no})
                continue
            m = CODE.search(line)
            if not m:
                continue
            kind, letters, title = m.group(1), m.group(2), m.group(4)
            if kind == "OEC":
                continue
            j = i + 1
            while (len(title) < 6 or title.endswith(CONNECTOR_END)) and j < min(i + 4, len(lines)):
                nxt = lines[j]
                if not nxt or CODE.search(nxt) or re.fullmatch(r"[\d:=.\s-]+", nxt):
                    break
                title = f"{title} {nxt}".strip()
                j += 1
            title = clean(title)
            if len(re.findall(r"[A-Za-z]{3,}", title)) < 1 or len(title) < 5 or len(title) > 90:
                continue
            if NOT_A_COURSE.search(title) or BRANCH_NAME_ONLY.match(title):
                continue
            if re.search(r"from basket|credits|hours|semester|total", title, re.IGNORECASE):
                continue
            if doc_name in DOCUMENT_BRANCH:
                branch = DOCUMENT_BRANCH[doc_name] if kind in ("PCC", "PEC") else "FIRST_YEAR"
            elif kind in ("BSC", "ESC", "HSMC"):
                branch = "FIRST_YEAR"
            elif doc_name == "ug-vol2" and letters == "CS":
                # Vol. II codes its Chemical courses "PCC-CS1xx" (sic); its PCC-CS3xx+ are CSE
                # courses repeated from Vol. I, already covered there.
                if int(m.group(3)) >= 200:
                    continue
                branch = "CHE"
            else:
                branch = LETTERS_TO_BRANCH.get(letters)
            if branch:
                courses.append({"branch": branch, "kind": kind, "title": title,
                                "document": f"{doc_name}.pdf", "page": page_no})
    return courses


def extract_emerging(pages):
    courses, area = [], None
    for page_no, text in enumerate(pages, start=1):
        for line in text.splitlines():
            heading = line.strip().upper()
            if heading in EMERGING:
                area = EMERGING[heading]
                continue
            m = UNIT.match(line)
            if area and m and not re.search(r"programme|program to|write|implement", m.group(1), re.I):
                courses.append({"branch": area, "kind": "UNIT", "title": clean(m.group(1)),
                                "document": "ug-emerging.pdf", "page": page_no})
    return courses


def load_pages(path):
    import pymupdf
    with pymupdf.open(path) as doc:
        return [page.get_text() for page in doc]


def main():
    from sentence_transformers import SentenceTransformer

    courses = []
    for pdf in sorted(AICTE_DIR.glob("*.pdf")):
        pages = load_pages(pdf)
        courses += extract_emerging(pages) if pdf.stem == "ug-emerging" else extract_coded(pdf.stem, pages)
    # De-duplicate (the same course is listed in the scheme and again in the syllabus)
    seen, unique = set(), []
    for c in courses:
        key = (c["branch"], normalise_title(c["title"]))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    # Drop truncated copies: "Formal Language & Automata" when "... Automata Theory"
    # exists, "Phase Transformation" when "Phase Transformations" exists.
    keys = defaultdict(set)
    for c in unique:
        keys[c["branch"]].add(normalise_title(c["title"]))
    courses = [c for c in unique if not any(
        other != (k := normalise_title(c["title"])) and (other.startswith(k + " ") or other == k + "s")
        for other in keys[c["branch"]])]
    by_branch = defaultdict(list)
    for c in courses:
        by_branch[c["branch"]].append(c)

    model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    catalogue = load_catalogue()
    comparison, gaps = [], []
    for code, subjects in catalogue["groups"].items():
        refs = [c for b in REFERENCE.get(code, []) for c in by_branch.get(b, [])]
        if not refs:
            for s in subjects:
                comparison.append({"branch": code, "subject_id": s["id"], "subject": s["name"],
                                   "aicte_course": "", "aicte_branch": "", "similarity": "",
                                   "status": "no AICTE reference", "document": "", "page": ""})
            continue
        ref_vecs = model.encode([r["title"] for r in refs], normalize_embeddings=True, show_progress_bar=False)
        sub_vecs = model.encode([f"{s['name']}: {', '.join(s['topics'])}" for s in subjects],
                                normalize_embeddings=True, show_progress_bar=False)
        # Name + topics is the main signal. The bare name counts only when it is
        # near-identical (>= 0.80): otherwise "Industrial Engineering" matched
        # "Aerospace Engineering" (0.64) through the shared word "Engineering".
        name_vecs = model.encode([s["name"] for s in subjects], normalize_embeddings=True, show_progress_bar=False)
        name_sims = name_vecs @ ref_vecs.T
        sims = np.maximum(sub_vecs @ ref_vecs.T, np.where(name_sims >= 0.80, name_sims, 0.0))
        for s, row in zip(subjects, sims):
            best = int(np.argmax(row))
            score = float(row[best])
            status = "match" if score >= MATCH else "partial" if score >= PARTIAL else "no close AICTE course"
            comparison.append({"branch": code, "subject_id": s["id"], "subject": s["name"],
                               "aicte_course": refs[best]["title"], "aicte_branch": refs[best]["branch"],
                               "similarity": round(score, 3), "status": status,
                               "document": refs[best]["document"], "page": refs[best]["page"]})
        # AICTE professional-core courses that no subject of this group covers
        primary = [r for r in refs if r["branch"] == REFERENCE[code][0] and r["kind"] in ("PCC", "UNIT")]
        if primary:
            idx = [refs.index(r) for r in primary]
            best_cover = sims[:, idx].max(axis=0)
            for r, cover in zip(primary, best_cover):
                if cover < PARTIAL:
                    gaps.append({"branch": code, "aicte_course": r["title"], "aicte_branch": r["branch"],
                                 "best_coverage": round(float(cover), 3), "document": r["document"],
                                 "page": r["page"]})

    def write(name, rows):
        with (OUT_DIR / name).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write("aicte_courses.csv", courses)
    write("aicte_comparison.csv", comparison)
    write("aicte_gaps.csv", gaps)

    print(f"AICTE courses extracted: {len(courses)} "
          f"({', '.join(f'{b}:{len(v)}' for b, v in sorted(by_branch.items()))})")
    counts = defaultdict(lambda: defaultdict(int))
    for row in comparison:
        counts[row["branch"]][row["status"]] += 1
    print(f"\n{'group':<8}{'match':>7}{'partial':>9}{'none':>6}{'no ref':>8}{'AICTE gaps':>12}")
    gap_counts = defaultdict(int)
    for g in gaps:
        gap_counts[g["branch"]] += 1
    for code in catalogue["groups"]:
        c = counts[code]
        print(f"{code:<8}{c['match']:>7}{c['partial']:>9}{c['no close AICTE course']:>6}"
              f"{c['no AICTE reference']:>8}{gap_counts[code]:>12}")
    print(f"\nSaved: {OUT_DIR / 'aicte_comparison.csv'}, aicte_gaps.csv, aicte_courses.csv")


if __name__ == "__main__":
    main()
