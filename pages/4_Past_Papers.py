"""
Past papers: upload a previous-year question paper, see its questions with
their marks and matched subject, and answer any of them. The team can add a
whole paper to the question bank, so every question in it becomes instant.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitAPIException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root, however started

from config import settings  # noqa: E402
from src.data_collection.past_papers import SubjectMatcher, split_questions  # noqa: E402
from src.rag.session_document import DOCUMENT_TYPES, DocumentError, read_document  # noqa: E402
from src.ui.resources import get_pipeline, get_retriever, load_limits  # noqa: E402
from src.utils.access import code_matches, require_access  # noqa: E402
from src.utils.catalogue_loader import load_catalogue  # noqa: E402

st.set_page_config(page_title="Past papers", page_icon=":material/history_edu:", layout="wide")
require_access("app", settings.ACCESS_CODE, title="EduRoute")

QUEUE_DIR = settings.BASE_DIR / "outputs" / "past_papers"

with st.sidebar:
    try:
        st.page_link("app.py", label="Ask EduRoute", icon=":material/forum:")
        st.page_link("pages/4_Past_Papers.py", label="Past papers", icon=":material/history_edu:")
        st.page_link("pages/5_Practice.py", label="Practice", icon=":material/quiz:")
        if settings.ENABLE_STUDIO:
            st.page_link("pages/2_Evaluation_Studio.py", label="Evaluation studio", icon=":material/fact_check:")
            st.page_link("pages/3_Operations_Dashboard.py", label="Operations", icon=":material/monitoring:")
    except StreamlitAPIException:
        pass


@st.cache_resource(show_spinner="Preparing subject matching...")
def get_matcher():
    return SubjectMatcher(load_catalogue(), get_retriever().model)


@st.cache_data(show_spinner=False)
def subject_options():
    catalogue = load_catalogue()
    return {s["id"]: f"{s['name']} (Year {s['year']})" for s in catalogue["subjects"]}


def home_branch(subject_id):
    for code, subjects in load_catalogue()["groups"].items():
        if subject_id.startswith(code) and any(s["id"] == subject_id for s in subjects):
            return code
    return None


st.title("Past papers")
st.markdown("Upload a previous-year question paper. Each question is listed with its marks and the "
            "subject it belongs to, and you can get an exam-style answer for any of them.")

subjects = subject_options()
left, right = st.columns([3, 2], vertical_alignment="bottom")
paper = left.file_uploader("Question paper", type=DOCUMENT_TYPES,
                           help="PDF, Word or text. A scanned PDF needs OCR first.")
hint = right.selectbox("Subject of this paper (optional)", [None, *subjects],
                       format_func=lambda s: "Work it out from each question" if s is None else subjects[s])

if paper is not None:
    stamp = (paper.name, paper.size, hint)
    if st.session_state.get("paper_stamp") != stamp:
        try:
            with st.spinner("Reading the paper..."):
                questions = split_questions(read_document(paper.name, paper.getvalue()))
                get_matcher().match(questions, subject_hint=hint)
        except DocumentError as error:
            st.error(str(error))
            st.stop()
        st.session_state.paper_questions = [q.to_dict() for q in questions]
        st.session_state.paper_stamp = stamp
        st.session_state.pop("paper_answer", None)

questions = st.session_state.get("paper_questions")
if not questions:
    st.caption("Tip: the paper's own question numbers and marks, like *Q2 (b) ... (5 marks)*, "
               "are read automatically.")
    st.stop()

st.subheader(f"{len(questions)} questions found")
table = pd.DataFrame(questions)
st.dataframe(
    table[["number", "text", "marks", "answer_mode", "subject_name", "topic", "match_score"]],
    hide_index=True, width="stretch",
    column_config={"number": "No.", "text": st.column_config.TextColumn("Question", width="large"),
                   "marks": "Marks", "answer_mode": "Answer format", "subject_name": "Subject",
                   "topic": "Closest topic",
                   "match_score": st.column_config.ProgressColumn("Match", min_value=0, max_value=1,
                                                                  format="%.2f")},
)

st.subheader("Answer a question")
pick = st.selectbox("Question", range(len(questions)),
                    format_func=lambda i: f"{questions[i]['number']}. {questions[i]['text'][:110]}")
q = questions[pick]
mode = st.segmented_control("Answer format", ["2-mark", "5-mark", "10-mark"], default=q["answer_mode"],
                            key=f"mode_{pick}") or q["answer_mode"]
if st.button("Answer this question", type="primary", icon=":material/menu_book:"):
    rate_limiter, answer_slots = load_limits()
    if not rate_limiter.allow(st.session_state.setdefault("session_key", datetime.now().isoformat())):
        st.warning("You have reached the hourly question limit. Please try again later.")
        st.stop()
    branch = home_branch(q["subject_id"]) if q.get("subject_id") else None
    with st.status("Looking it up…", type="compact") as status:
        with answer_slots.acquire(settings.QUEUE_TIMEOUT) as got_slot:
            if not got_slot:
                status.update(label="Server busy", state="error")
                st.stop()
            response = get_pipeline().ask(
                q["text"], answer_mode=mode,
                filters={"subject_id": q["subject_id"]} if q.get("subject_id") else None,
                fallback_filters=[{"discipline": branch}, None] if branch else [None])
        seconds = response.telemetry["total_ms"] / 1000
        status.update(label="Answered instantly from the question bank" if response.telemetry.get("cache_hit")
                      else f"Answered in {seconds:.0f} s", state="complete")
    st.session_state.paper_answer = (pick, mode, response.to_dict())

if st.session_state.get("paper_answer") and st.session_state.paper_answer[0] == pick:
    _, shown_mode, r = st.session_state.paper_answer
    with st.container(border=True):
        st.caption(f"{q['number']} · {shown_mode} · {q.get('subject_name') or 'all subjects'}")
        for notice in r["notices"]:
            st.warning(notice, icon=":material/info:")
        st.markdown(r["answer"])
        if r["sources"]:
            with st.expander(f"Sources ({len(r['cited_sources'])} cited of {len(r['sources'])})",
                             icon=":material/menu_book:"):
                for s in r["sources"]:
                    title = f"[{s['article_title']}]({s['source_url']})" if s.get("source_url") else s["article_title"]
                    st.markdown(f"**[{s['number']}] {title}**")
                    st.caption(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))

with st.expander("For the team: add this paper to the question bank", icon=":material/library_add:"):
    st.caption("Every question in the paper is answered in the background (about 1–2 minutes each on "
               "this laptop) and saved, so students get it instantly. Needs the team code.")
    code = st.text_input("Team code", type="password", key="bank_code")
    if st.button("Answer the whole paper in the background", icon=":material/schedule:"):
        if not code_matches(code, settings.STUDIO_CODE):
            st.error("That team code is not right.")
        else:
            QUEUE_DIR.mkdir(parents=True, exist_ok=True)
            queue = QUEUE_DIR / f"paper_{datetime.now():%Y%m%d_%H%M%S}.json"
            queue.write_text(json.dumps([{"question": x["text"], "answer_mode": x["answer_mode"],
                                          "subject_id": x["subject_id"]} for x in questions], indent=1),
                             encoding="utf-8")
            flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS) if sys.platform == "win32" else 0
            subprocess.Popen([sys.executable, "-u", "-m", "src.inference.question_bank", "--from-file", str(queue)],
                             cwd=settings.BASE_DIR, creationflags=flags,
                             stdout=open(settings.BASE_DIR / "outputs" / "question_bank.log", "a"),
                             stderr=subprocess.STDOUT)
            st.success(f"Answering {len(questions)} questions in the background. Each becomes instant as soon "
                       "as it is done; progress is in outputs/question_bank.log.")
