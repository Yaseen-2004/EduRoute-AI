"""
Practice: self-test quizzes written from the reference sources. Every question's
correct answer is checked against the sources before it is shown.
"""

import sys
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitAPIException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root, however started

from config import settings  # noqa: E402
from src.learning.quiz import QuizMaker  # noqa: E402
from src.ui.resources import get_pipeline, load_limits  # noqa: E402
from src.utils.access import require_access  # noqa: E402
from src.utils.catalogue_loader import load_catalogue  # noqa: E402

st.set_page_config(page_title="Practice", page_icon=":material/quiz:", layout="centered")
require_access("app", settings.ACCESS_CODE, title="EduRoute")

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


@st.cache_data(show_spinner=False)
def catalogue_choices():
    catalogue = load_catalogue()
    names = {g["code"]: g["name"] for g in catalogue["metadata"]["groups"]}
    subjects = {code: {s["id"]: s for s in group} for code, group in catalogue["groups"].items() if group}
    return names, subjects


st.title("Practice")
st.markdown("Test yourself on any topic. The questions are written from the same reference sources "
            "as the answers, and each correct answer is checked against them first.")

names, subjects_by_branch = catalogue_choices()
branch = st.selectbox("Branch", list(subjects_by_branch), format_func=lambda c: names.get(c, c))
subjects = subjects_by_branch[branch]
subject_id = st.selectbox("Subject", list(subjects),
                          format_func=lambda s: f"{subjects[s]['name']} (Year {subjects[s]['year']})")
subject = subjects[subject_id]
cols = st.columns([3, 1], vertical_alignment="bottom")
topic = cols[0].selectbox("Topic", subject["topics"])
n = cols[1].segmented_control("Questions", [3, 5], default=5) or 5

key = (subject_id, topic)
make = st.button("Start a quiz", type="primary", icon=":material/quiz:")
fresh = st.session_state.get("quiz_key") == key and st.button("New questions on this topic",
                                                                  icon=":material/refresh:")
if make or fresh:
    rate_limiter, answer_slots = load_limits()
    with st.status("Writing the questions from the sources…", type="compact") as status:
        with answer_slots.acquire(settings.QUEUE_TIMEOUT) as got_slot:
            if not got_slot:
                status.update(label="Server busy - please try again in a minute", state="error")
                st.stop()
            quiz = QuizMaker(get_pipeline()).make(
                topic, subject_id=subject_id, filters={"subject_id": subject_id},
                fallback_filters=[{"discipline": branch}, None], n=n, fresh=bool(fresh))
        quiz = quiz.to_dict()
        if not quiz["questions"]:
            status.update(label="No reliable questions could be written for this topic", state="error")
        else:
            status.update(label=f"{len(quiz['questions'])} questions ready" +
                          (f" ({quiz['dropped']} rejected by the source check)" if quiz["dropped"] else ""),
                          state="complete")
    st.session_state.quiz, st.session_state.quiz_key = quiz, key
    st.session_state.pop("quiz_marked", None)

quiz = st.session_state.get("quiz")
if not quiz or st.session_state.get("quiz_key") != key:
    st.stop()
if not quiz["questions"]:
    st.info("The knowledge base had too little on this topic for a checked quiz. Try another topic.")
    st.stop()

with st.form("quiz_form"):
    for i, q in enumerate(quiz["questions"]):
        st.radio(f"**{i + 1}.** {q['question']}", range(4), index=None, key=f"q_{i}",
                 format_func=lambda j, q=q: q["options"][j])
    marked = st.form_submit_button("Check my answers", type="primary")
if marked:
    st.session_state.quiz_marked = True

if st.session_state.get("quiz_marked"):
    sources = {s["number"]: s for s in quiz["sources"]}
    answers = [st.session_state.get(f"q_{i}") for i in range(len(quiz["questions"]))]
    score = sum(a == q["answer"] for a, q in zip(answers, quiz["questions"]))
    st.subheader(f"Score: {score} / {len(quiz['questions'])}")
    for i, (q, given) in enumerate(zip(quiz["questions"], answers)):
        right = given == q["answer"]
        with st.container(border=True):
            st.markdown(f"{':green[:material/check_circle:]' if right else ':red[:material/cancel:]'} "
                        f"**{i + 1}. {q['question']}**")
            if not right:
                yours = q["options"][given] if given is not None else "no answer"
                st.markdown(f"Your answer: {yours}")
            st.markdown(f"Correct answer: **{q['options'][q['answer']]}**")
            if q.get("explanation"):
                st.caption(q["explanation"])
            s = sources.get(q["source"])
            if s:
                title = f"[{s['article_title']}]({s['source_url']})" if s.get("source_url") else s["article_title"]
                st.caption(f"Source: {title}")
