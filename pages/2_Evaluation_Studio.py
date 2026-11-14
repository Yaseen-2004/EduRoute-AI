"""
Evaluation studio: the team's remaining evaluation tasks, one click at a time.

  Grade answers   blind 0-3 grading of the system benchmark (one file per rater)
  Label claims    supported / unsupported judgements for verifier calibration
  Record voice    consented recordings + reference text for the WER evaluation
  Add images      real images + question + expected facts for the image benchmark
  Review syllabus confirm or correct each catalogue subject (AICTE comparison shown)

Everything is saved under evaluation/ (and config/branches_review/) in the
formats the analysis scripts read. Run locally: the studio writes files, so it
is disabled on a hosted deployment (EDUROUTE_ENABLE_STUDIO=false).
"""

import csv
import glob
import json
import re
from datetime import datetime
from pathlib import Path

import sys

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitAPIException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root, however started
from config import settings  # noqa: E402
from src.utils.access import require_access  # noqa: E402
from src.utils.catalogue_loader import load_catalogue

ROOT = settings.BASE_DIR
RESULTS = ROOT / "evaluation" / "results"
GRADING_DIR = RESULTS / "grading"
LABEL_DIR = RESULTS / "claim_labelling"
VOICE_DIR = ROOT / "evaluation" / "voice_eval"
IMAGE_DIR = ROOT / "evaluation" / "image_eval"
REVIEW_DIR = ROOT / "config" / "branches_review"
AICTE_FILE = ROOT / "data" / "metadata" / "aicte_comparison.csv"
CRITERIA = ["correctness_0_3", "completeness_0_3", "relevance_0_3", "clarity_0_3"]

st.set_page_config(page_title="Evaluation studio", page_icon=":material/fact_check:", layout="wide")

if not settings.ENABLE_STUDIO:
    st.info("The evaluation studio is disabled on this deployment. Run the app locally to use it.")
    st.stop()
# Grading keys and ratings live here: the team's own code, not the students' code.
require_access("studio", settings.STUDIO_CODE, title="Evaluation studio")

with st.sidebar:
    try:  # the links need the multipage app; skipped when this page runs on its own (tests)
        st.page_link("app.py", label="Ask EduRoute", icon=":material/forum:")
        st.page_link("pages/4_Past_Papers.py", label="Past papers", icon=":material/history_edu:")
        st.page_link("pages/5_Practice.py", label="Practice", icon=":material/quiz:")
        st.page_link("pages/2_Evaluation_Studio.py", label="Evaluation studio", icon=":material/fact_check:")
        st.page_link("pages/3_Operations_Dashboard.py", label="Operations", icon=":material/monitoring:")
    except StreamlitAPIException:
        pass

st.title("Evaluation studio")
st.caption("Each person works under their own name; files are saved after every click.")

rater = st.text_input("Your name or initials", key="rater",
                      placeholder="e.g. Yaseen").strip()
safe_rater = re.sub(r"[^A-Za-z0-9_-]", "", rater)[:30]


def latest(pattern):
    files = sorted(glob.glob(str(RESULTS / pattern)))
    return Path(files[-1]) if files else None


def rater_copy(source, folder, name):
    """Each rater works on their own copy of the sheet, so scores are independent."""
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{source.stem}__{name}.csv"
    if not target.exists():
        target.write_bytes(source.read_bytes())
    return target


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


grade_tab, claim_tab, voice_tab, image_tab, syllabus_tab = st.tabs(
    ["Grade answers", "Label claims", "Record voice", "Add images", "Review syllabus"])

# ---------------------------------------------------------------- grading
with grade_tab:
    sheet = latest("grading_sheet_*.csv")
    if not sheet:
        st.warning("No grading sheet found in evaluation/results/.")
    elif not safe_rater:
        st.info("Enter your name above to start grading.")
    else:
        path = rater_copy(sheet, GRADING_DIR, safe_rater)
        df = read_csv(path)
        done = df[CRITERIA].apply(lambda r: all(v != "" for v in r), axis=1)
        st.progress(done.mean(), text=f"{int(done.sum())} of {len(df)} answers graded · {path.name}")
        with st.expander("Rubric", icon=":material/rule:"):
            st.markdown((ROOT / "evaluation" / "ANSWER_RUBRIC.md").read_text(encoding="utf-8"))
        remaining = [i for i in range(len(df)) if not done.iloc[i]]
        if not remaining:
            st.success("All answers graded. Thank you!")
            st.code(f"venv\\Scripts\\python -m evaluation.analyse_grading --key "
                    f"evaluation\\results\\{sheet.name.replace('grading_sheet', 'grading_key').replace('.csv', '.json')} "
                    f"--benchmark evaluation\\results\\{sheet.name.replace('grading_sheet', 'system_benchmark').replace('.csv', '.json')} "
                    f"evaluation\\results\\grading\\*{sheet.stem}__*.csv", language="bat")
        else:
            i = st.session_state.get("grade_index", remaining[0])
            if i not in remaining:
                i = remaining[0]
            row = df.iloc[i]
            st.subheader(f"Question {row['id']} · {row['answer_mode']} · system {row['system_code']}")
            st.markdown(f"**{row['question']}**")
            with st.container(border=True):
                st.markdown(row["answer"] or "_(empty answer)_")
            if row["sources"]:
                st.caption("Sources: " + row["sources"])
            with st.form(f"grade_{i}", border=False):
                cols = st.columns(4)
                scores = {}
                for col, crit in zip(cols, CRITERIA):
                    with col:
                        scores[crit] = st.segmented_control(
                            crit.replace("_0_3", "").title(), options=[0, 1, 2, 3], key=f"{crit}_{i}")
                notes = st.text_input("Notes (optional)", key=f"notes_{i}")
                if st.form_submit_button("Save and next", type="primary"):
                    if any(v is None for v in scores.values()):
                        st.error("Give all four scores before saving.")
                    else:
                        for crit, value in scores.items():
                            df.at[i, crit] = str(value)
                        df.at[i, "notes"] = notes
                        df.to_csv(path, index=False)
                        later = [j for j in remaining if j != i]
                        st.session_state.grade_index = later[0] if later else -1
                        st.rerun()

# ---------------------------------------------------------- claim labels
with claim_tab:
    sheet = latest("claim_labels_*.csv")
    if not sheet:
        st.warning("No claim-labelling sheet found in evaluation/results/.")
    elif not safe_rater:
        st.info("Enter your name above to start labelling.")
    else:
        path = rater_copy(sheet, LABEL_DIR, safe_rater)
        df = read_csv(path)
        done = df["human_label"].str.strip() != ""
        st.progress(done.mean(), text=f"{int(done.sum())} of {len(df)} claims labelled · {path.name}")
        st.caption("Judge only whether the **evidence shown** backs the sentence, not your own knowledge.")
        remaining = [i for i in range(len(df)) if not done.iloc[i]]
        if not remaining:
            st.success("All claims labelled.")
            st.code(f"venv\\Scripts\\python -m evaluation.claim_labels score {path}", language="bat")
        else:
            i = remaining[0]
            row = df.iloc[i]
            st.markdown(f"**Question:** {row['question']}")
            st.markdown(f"**Claim:** {row['claim']}")
            with st.container(border=True):
                st.markdown(f"**Evidence:** {row['evidence'] or '_(none)_'}")
            buttons = st.columns(3)
            for col, label, kind in zip(buttons, ("supported", "unsupported", "unsure"),
                                        ("primary", "secondary", "tertiary")):
                if col.button(label.title(), key=f"claim_{label}_{i}", type=kind, width="stretch"):
                    df.at[i, "human_label"] = label
                    df.to_csv(path, index=False)
                    st.rerun()

# ------------------------------------------------------------ voice
with voice_tab:
    st.markdown("Record students reading engineering questions, for the word-error-rate evaluation. "
                "Recordings stay on this computer in `evaluation/voice_eval/`.")
    consent = st.checkbox("The speaker agrees to this recording being stored and used to evaluate "
                          "the speech recognition of this project.")
    speaker = re.sub(r"[^A-Za-z0-9]", "", st.text_input("Speaker code (no real names)", placeholder="S1"))
    accent = st.text_input("Accent / first language (optional)", placeholder="Telugu-English")
    questions = []
    for name in ("retrieval_engineering_v1.jsonl", "retrieval_aiml_v1.jsonl"):
        for line in open(ROOT / "evaluation" / "benchmarks" / name, encoding="utf-8"):
            questions.append(json.loads(line))
    refs_path = VOICE_DIR / "references.csv"
    recorded = set()
    if refs_path.exists():
        recorded = {(r["speaker"], r["file"].split("_")[1].split(".")[0])
                    for r in csv.DictReader(open(refs_path, encoding="utf-8"))}
    if consent and speaker:
        todo = [q for q in questions if (speaker, q["id"]) not in recorded]
        st.progress(1 - len(todo) / len(questions), text=f"{len(questions) - len(todo)} recorded by {speaker}")
        if todo:
            q = todo[0]
            st.subheader("Please read aloud:")
            st.markdown(f"### {q['question']}")
            audio = st.audio_input("Record", key=f"audio_{speaker}_{q['id']}")
            if audio is not None and st.button("Save recording", type="primary"):
                VOICE_DIR.mkdir(parents=True, exist_ok=True)
                fname = f"{speaker}_{q['id']}.wav"
                (VOICE_DIR / fname).write_bytes(audio.getvalue())
                new = not refs_path.exists()
                with refs_path.open("a", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=["file", "reference", "speaker", "accent"])
                    if new:
                        writer.writeheader()
                    writer.writerow({"file": fname, "reference": q["question"], "speaker": speaker,
                                     "accent": accent})
                st.rerun()
        st.code("venv\\Scripts\\python -m evaluation.evaluate_asr evaluation\\voice_eval", language="bat")
    elif not consent:
        st.info("Tick the consent box to start recording.")

# ------------------------------------------------------------ images
with image_tab:
    st.markdown("Add real images (circuit photos, handwritten notes, textbook figures) to the image benchmark.")
    with st.form("add_image", clear_on_submit=True):
        upload = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"])
        question = st.text_input("Question a student would ask about it")
        mode = st.segmented_control("Answer mode", ["2-mark", "5-mark", "10-mark"], default="2-mark")
        expected = st.text_input(
            "Facts a correct answer must contain",
            help="Separate required facts with ';' and accepted spellings with '|'. "
                 "Example: 30 Ω|30 ohm;0.4 A|0.4 amp")
        kind = st.selectbox("Kind", ["circuit", "graph", "chart", "table", "diagram", "equation",
                                     "code", "flowchart", "handwritten", "photo"])
        if st.form_submit_button("Add to benchmark", type="primary"):
            if not (upload and question and expected):
                st.error("Image, question and expected facts are all required.")
            else:
                IMAGE_DIR.mkdir(parents=True, exist_ok=True)
                name = f"team_{datetime.now():%Y%m%d_%H%M%S}{Path(upload.name).suffix.lower()}"
                (IMAGE_DIR / name).write_bytes(upload.getvalue())
                with (IMAGE_DIR / "questions.csv").open("a", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerow([name, question, mode or "2-mark", expected, kind,
                                             f"team ({safe_rater or 'unknown'})"])
                st.success(f"Added {name}. Rebuild the Colab bundle to include it.")
    team_rows = [r for r in csv.DictReader(open(IMAGE_DIR / "questions.csv", encoding="utf-8"))
                 if r.get("source", "").startswith("team")] if (IMAGE_DIR / "questions.csv").exists() else []
    st.caption(f"{len(team_rows)} team images in the benchmark (plus 8 synthetic starter images).")

# --------------------------------------------------------- syllabus review
with syllabus_tab:
    catalogue = load_catalogue()
    groups = {g["code"]: g["name"] for g in catalogue["metadata"]["groups"]}
    code = st.selectbox("Branch", list(groups), format_func=lambda c: f"{c} · {groups[c]}")
    aicte = read_csv(AICTE_FILE) if AICTE_FILE.exists() else None
    review_path = REVIEW_DIR / f"{code}.csv"
    rows = []
    for s in catalogue["groups"].get(code, []):
        row = {"subject_id": s["id"], "subject": s["name"], "year": s["year"],
               "topics": "; ".join(s["topics"]), "keep": True, "change_to": "", "notes": ""}
        if aicte is not None:
            match = aicte[aicte["subject_id"] == s["id"]]
            row["aicte_match"] = match["aicte_course"].iloc[0] if len(match) else ""
        rows.append(row)
    table = pd.DataFrame(rows)
    if review_path.exists():
        saved = read_csv(review_path).set_index("subject_id")
        for col in ("change_to", "notes"):
            table[col] = table["subject_id"].map(saved[col]).fillna("")
        table["keep"] = table["subject_id"].map(saved["keep"].str.lower() == "true").fillna(True)
    edited = st.data_editor(
        table, hide_index=True, width="stretch", key=f"review_{code}",
        disabled=[c for c in table.columns if c not in ("keep", "change_to", "notes")],
        column_config={"keep": st.column_config.CheckboxColumn("Keep as is"),
                       "topics": st.column_config.TextColumn(width="large")})
    if st.button("Save review", type="primary"):
        REVIEW_DIR.mkdir(parents=True, exist_ok=True)
        edited.to_csv(review_path, index=False)
        st.success(f"Saved {review_path.relative_to(ROOT)}")
    reviewed = len(list(REVIEW_DIR.glob("*.csv"))) if REVIEW_DIR.exists() else 0
    st.caption(f"{reviewed} of {len(groups)} branches reviewed.")
