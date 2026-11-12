"""
EduRoute AI - student-facing Streamlit app (report §20).

Run from the project root:
    venv\\Scripts\\streamlit run app.py
"""

import hashlib
import logging
import threading
import time
import uuid
from datetime import datetime, timezone

import streamlit as st

from config import settings
from src.llm.prompts import ANSWER_MODES, strip_prompt_echo
from src.rag.session_document import (DOCUMENT_TYPES, MAX_DOCUMENT_BYTES, DocumentError,
                                      build_session_document)
from src.utils.access import AnswerSlots, SlidingWindowLimiter, require_access
from src.utils.catalogue_loader import load_catalogue
from src.utils.io import append_jsonl
from src.utils.retention import purge_logs
from src.ui.galaxy import knowledge_galaxy

AVATARS = {"user": ":material/person:", "assistant": ":material/auto_awesome:"}
logger = logging.getLogger("eduroute.app")

st.set_page_config(page_title="EduRoute AI", page_icon=":material/school:", layout="centered")
require_access("app", settings.ACCESS_CODE)

IMAGE_TYPES = ["png", "jpg", "jpeg", "webp", "bmp"]
WORKFLOW_LABELS = {
    "direct_llm": ":material/bolt: Direct answer",
    "rag_llm": ":material/menu_book: Answer from knowledge base",
    "vision_llm": ":material/image: Image analysis",
    "hybrid_multimodal_rag": ":material/image_search: Image + knowledge base",
}


PRIVACY_NOTICE = """
**What is collected.** For each question the system logs a timestamp, a
one-way hash of the question (not its text), the route taken, timings, token
counts and verification results. Thumbs-up/down feedback is logged the same
way. Question text is logged only if the operator turns that on for a
consented evaluation.

**What is not kept.** Uploaded images, documents and voice recordings are processed in
memory or in a temporary file that is deleted at once. Spoken answers are not
stored.

**Retention.** Logs are deleted automatically after {days} days.

**Where it goes.** Questions are sent only to the language and vision models
this deployment is configured to use. If those run on a cloud service (for
example Google Colab), that service processes the question text.

**Your choices.** Do not include personal details in questions. Ask the
project team to delete logs early if you need that.
"""


# ----------------------------------------------------------------- resources

@st.cache_resource(ttl=24 * 3600, show_spinner=False)
def enforce_log_retention():
    """Delete logs older than the retention period, at most once a day (report §21)."""
    return purge_logs()


enforce_log_retention()


@st.cache_resource(show_spinner="Loading the knowledge base...")
def get_retriever():
    from src.rag.retrieval import EduRouteRetriever
    return EduRouteRetriever(version=settings.RAG_VERSION, mode=settings.RETRIEVAL_MODE)


@st.cache_resource(show_spinner=False, max_entries=10)
def get_pipeline(strategy, cost_preference):
    from src.pipeline import EduRoutePipeline
    return EduRoutePipeline(strategy=strategy, cost_preference=cost_preference,
                            retriever=get_retriever())


@st.cache_resource(show_spinner=False)
def load_limits():
    """Shared by every session on this server."""
    return (SlidingWindowLimiter(settings.MAX_QUESTIONS_PER_HOUR, 3600),
            AnswerSlots(settings.MAX_CONCURRENT_ANSWERS))


@st.cache_resource(show_spinner="Loading speech recognition...")
def get_recognizer():
    from src.voice.speech import SpeechRecognizer
    return SpeechRecognizer()


@st.cache_data(ttl=30, show_spinner=False)
def system_status():
    from src.llm.backends import build_backend
    from src.voice.speech import availability

    llm = build_backend(settings, "text")
    llm_ok, llm_detail = llm.health_check() if llm.name != "extractive" else (False, "extractive mode")
    vision = build_backend(settings, "vision")
    vision_ok, vision_detail = vision.health_check() if vision else (False, "not configured")
    voice = availability()
    return {
        "llm": (llm_ok, llm_detail),
        "vision": (vision_ok, vision_detail),
        "asr": voice["asr"],
        "tts": voice["tts_edge"] or voice["tts_gtts"],
    }


@st.cache_data(show_spinner=False)
def catalogue_options():
    """Branches and subjects that actually have indexed content (report §20:
    do not offer coverage that does not exist)."""
    catalogue = load_catalogue()
    retriever = get_retriever()
    indexed_subjects = {s for m in retriever.metadata for s in (m.get("subject_ids") or [m.get("subject_id")])}
    branches, subjects = {}, {}
    for group in catalogue["metadata"]["groups"]:
        code = group["code"]
        available = [s for s in catalogue["groups"].get(code, []) if s["id"] in indexed_subjects]
        if available:
            branches[code] = group["name"]
            subjects[code] = {s["id"]: f"{s['name']} (Year {s['year']})" for s in available}
    return branches, subjects


@st.cache_data(show_spinner=False)
def subject_record(subject_id):
    return next((s for s in load_catalogue()["subjects"] if s["id"] == subject_id), None)


@st.cache_data(show_spinner=False)
def coverage_table():
    """Topic -> source article mapping written by the corpus builder (audit file)."""
    import pandas as pd

    path = settings.METADATA_DIR / "engineering_topic_resolution_v1.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    table["source"] = "https://en.wikipedia.org/wiki/" + table["resolved_title"].str.replace(" ", "_")
    return table


@st.dialog("Knowledge base coverage", width="large")
def show_coverage(branch_code):
    table = coverage_table()
    if table is None:
        st.info("Coverage data is available once the engineering knowledge base has been built.")
        return
    if branch_code:
        table = table[table["discipline"] == branch_code]
    st.caption(
        f"{len(table)} catalogue topics and the Wikipedia article each one is answered from. "
        "Match score is semantic similarity to the topic in its subject context (a ranking "
        "signal, not a probability). The catalogue is illustrative and should be checked "
        "against your university's official syllabus."
    )
    st.dataframe(
        table[["subject_name", "topic", "resolved_title", "score", "source"]],
        hide_index=True,
        column_config={
            "subject_name": "Subject",
            "topic": "Topic",
            "resolved_title": "Source article",
            "score": st.column_config.NumberColumn("Match score", format="%.2f"),
            "source": st.column_config.LinkColumn("Link", display_text="open"),
        },
        alt="Catalogue topics and their source articles",
    )


# ------------------------------------------------------------------- state

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_key" not in st.session_state:
    st.session_state.session_key = uuid.uuid4().hex
if "voice_pending" not in st.session_state:
    st.session_state.voice_pending = False
if "voice_corrections" not in st.session_state:
    st.session_state.voice_corrections = []

status = system_status()


@st.cache_resource(show_spinner=False)
def warm_up_models():
    """Load the models into memory in the background when the app starts, so the
    first student question does not wait for it (a cold load took ~48 s on the
    laptop CPU and several seconds on a GPU)."""
    def run():
        from src.llm.backends import build_backend
        for role in ("text", "vision"):
            try:
                backend = build_backend(settings, role)
                if backend is not None and backend.name != "extractive" and backend.is_available():
                    backend.generate([{"role": "user", "content": "OK"}], max_new_tokens=1)
            except Exception:
                pass  # warm-up is best effort; a real question reports real errors
    threading.Thread(target=run, daemon=True).start()
    return True


@st.cache_resource(show_spinner=False)
def warm_up_speech():
    """Load Whisper in the background so the first voice question is not slowed down."""
    threading.Thread(target=get_recognizer()._load, daemon=True).start()
    return True


if status["llm"][0]:
    warm_up_models()
if status["asr"]:
    warm_up_speech()


# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.header("EduRoute AI")
    st.caption("Adaptive academic assistant for B.Tech students (prototype)")
    st.page_link("app.py", label="Ask EduRoute", icon=":material/forum:")
    if settings.ENABLE_STUDIO:
        st.page_link("pages/2_Evaluation_Studio.py", label="Evaluation studio", icon=":material/fact_check:")

    branches, subjects_by_branch = catalogue_options()
    branch = st.selectbox(
        "Branch", options=[None, *branches], key="branch_select",
        format_func=lambda b: f"All branches ({len(branches)})" if b is None else branches[b],
    )
    subjects = subjects_by_branch.get(branch, {}) if branch else {}
    subject_id = st.selectbox(
        "Subject", options=[None, *subjects], index=0, disabled=branch is None,
        format_func=lambda s: "All subjects in this branch" if s is None else subjects[s],
    )
    answer_mode = st.segmented_control(
        "Answer format", options=list(ANSWER_MODES), default="5-mark",
        format_func=lambda m: ANSWER_MODES[m]["label"].split(" (")[0].capitalize(),
    ) or "5-mark"
    speak = st.toggle("Read answers aloud", value=False, disabled=not status["tts"],
                      help=None if status["tts"] else "Install edge-tts or gTTS to enable.")
    developer_view = settings.ENABLE_DEVELOPER_VIEW and st.toggle(
        "Developer view", value=False,
        help="Show routing decisions, retrieval scores, verification and telemetry.")

    strategy, cost_preference = "adaptive", "balanced"
    if developer_view:
        strategy = st.selectbox(
            "Routing strategy", ["adaptive", "direct", "fixed_rag", "always_verify", "modality"],
            help="Baselines from report §22.1, for side-by-side comparison.",
        )
        cost_preference = st.segmented_control(
            "Cost preference", ["economy", "balanced", "quality"], default="balanced",
        ) or "balanced"

    with st.container(border=True):
        st.markdown("**System status**")
        llm_ok, llm_detail = status["llm"]
        st.markdown(f"{':green[:material/check_circle:]' if llm_ok else ':orange[:material/warning:]'} "
                    f"Language model - {'ready' if llm_ok else 'unavailable'}")
        if not llm_ok:
            st.caption(f"{llm_detail}. Answers will show source excerpts only. See README to start Qwen2.5-3B.")
        vision_ok = status["vision"][0]
        st.markdown(f"{':green[:material/check_circle:]' if vision_ok else ':gray[:material/block:]'} "
                    f"Image questions - {'ready' if vision_ok else 'not configured'}")
        st.markdown(f"{':green[:material/check_circle:]' if status['asr'] else ':gray[:material/block:]'} "
                    f"Voice input - {'ready' if status['asr'] else 'not installed'}")

    if st.button("Knowledge base coverage", icon=":material/library_books:", width="stretch"):
        show_coverage(branch)

    if st.session_state.get("document") is not None:
        doc = st.session_state.document
        with st.container(border=True):
            st.markdown(f":material/description: **{doc.name}**")
            st.caption(f"{doc.pages} pages · used for every question in this conversation")
            if st.button("Remove document", icon=":material/close:", width="stretch"):
                st.session_state.document = None
                st.rerun()

    if st.button("New conversation", icon=":material/refresh:", width="stretch"):
        st.session_state.messages = []
        st.session_state.document = None
        st.rerun()

    st.caption(
        "Privacy: questions are processed on this server and logged only as an anonymous "
        "hash with timing data. Uploaded images, documents and recordings are not stored. Answers are "
        "study aids, not official university guidance."
    )
    with st.expander("Privacy notice", icon=":material/shield:"):
        st.markdown(PRIVACY_NOTICE.format(days=settings.LOG_RETENTION_DAYS or "an unlimited number of"))


# --------------------------------------------------------------- rendering

def render_sources(sources):
    for s in sources:
        badge = ":green-badge[cited]" if s["cited"] else ":gray-badge[not cited]"
        section = f" - {s['section']}" if s.get("section") and s["section"] != "Introduction" else ""
        title = f"[{s['article_title']}]({s['source_url']})" if s.get("source_url") else s["article_title"]
        st.markdown(f"**[{s['number']}] {title}{section}** {badge}")
        st.caption(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))


def render_verification(v):
    if v["total_claims"] == 0:
        st.caption("No checkable claims found.")
        return
    cols = st.columns(3)
    cols[0].metric("Supported", v["supported"])
    cols[1].metric("Partly supported", v["partial"])
    cols[2].metric("Unsupported", v["unsupported"])
    if v.get("revised"):
        st.caption(f"The answer was revised after verification. {v.get('note', '')}")
    for claim in v["claims"]:
        icon = {"supported": ":green[:material/check:]", "partial": ":orange[:material/remove:]",
                "unsupported": ":red[:material/close:]"}.get(claim["label"], "")
        source = f" (source [{claim['best_source']}])" if claim.get("best_source") else ""
        st.markdown(f"{icon} {claim['text']}{source}")
        if claim["issues"]:
            st.caption("; ".join(claim["issues"]))


def render_developer(r):
    d = r["decision"]
    t = r["telemetry"]
    cols = st.columns(4)
    cols[0].metric("Total time", f"{t['total_ms'] / 1000:.1f} s")
    cols[1].metric("Model calls", t["model_calls"])
    cols[2].metric("Input tokens", t["input_tokens"])
    cols[3].metric("Output tokens", t["output_tokens"])
    st.markdown(f"**Strategy:** `{d['strategy']}` - **workflow:** `{r['workflow']}` - "
                f"**verify:** `{d['verify']}`")
    st.markdown("**Routing reasons**\n" + "\n".join(f"- {x}" for x in d["reasons"]))
    if d.get("verify_triggers"):
        st.markdown("**Verification triggers**\n" + "\n".join(f"- {x}" for x in d["verify_triggers"]))
    st.json({"stages_ms": t["stages_ms"], "retrieval_signal": d.get("retrieval_signal"),
             "analysis": d.get("analysis"), "backends": t["backends"]}, expanded=False)


def record_feedback(index):
    message = st.session_state.messages[index]
    value = st.session_state.get(f"feedback_{index}")
    if value is None:
        return
    append_jsonl(settings.LOG_DIR / "feedback.jsonl", {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question_sha256": hashlib.sha256(message["response"]["question"].encode()).hexdigest()[:16],
        "rating": "helpful" if value == 1 else "not_helpful",
        "workflow": message["response"]["workflow"],
        "answer_mode": message["response"]["answer_mode"],
    })


def render_assistant(message, index):
    r = message["response"]
    st.caption(WORKFLOW_LABELS.get(r["workflow"], r["workflow"]))
    for notice in r["notices"]:
        st.warning(notice, icon=":material/info:")
    st.markdown(r["answer"])
    if r.get("diagram_dot"):
        with st.container(border=True):
            st.caption(":material/account_tree: Workflow diagram (drawn from the answer above)")
            try:
                st.graphviz_chart(r["diagram_dot"], width="stretch")
            except Exception:
                st.caption("The diagram could not be drawn.")
            with st.expander("Diagram source (Graphviz DOT)"):
                st.code(r["diagram_dot"], language="dot")
    if r.get("visual_description"):
        with st.expander("What the vision model read from the image", icon=":material/visibility:"):
            st.caption("Machine reading of the image - may contain mistakes.")
            st.markdown(r["visual_description"])
    if message.get("audio"):
        st.audio(message["audio"], format="audio/mp3")
    if r["sources"]:
        with st.expander(f"Sources ({len(r['cited_sources'])} cited of {len(r['sources'])})",
                         icon=":material/menu_book:"):
            render_sources(r["sources"])
    if r.get("verification"):
        v = r["verification"]
        label = ("Verified" if not v["unsupported"] else f"Verification: {v['unsupported']} unsupported claim(s)")
        with st.expander(label, icon=":material/fact_check:"):
            render_verification(v)
    if developer_view:
        with st.expander("Developer view", icon=":material/developer_mode:"):
            render_developer(r)
    st.feedback("thumbs", key=f"feedback_{index}", on_change=record_feedback, args=(index,))


def _focus_branch_from_galaxy():
    """A branch clicked in the galaxy becomes the sidebar's branch ('' = all branches)."""
    code = getattr(st.session_state.get("galaxy"), "branch", None)
    if code is not None:
        st.session_state.branch_select = code or None


knowledge_galaxy(
    [{"code": code, "name": name, "subjects": len(subjects_by_branch.get(code, {}))}
     for code, name in branches.items()],
    selected=branch,
    compact=bool(st.session_state.messages),
    tagline="Ask anything across 26 engineering branches, by text, photo, voice or your own notes. "
            "Every answer is routed, grounded and checked.",
    chips=[f"{len(branches)} branches", f"{sum(len(v) for v in subjects_by_branch.values())} subjects",
           f"{len(get_retriever().metadata):,} passages", "Text · photo · voice · PDF"],
    on_branch_change=_focus_branch_from_galaxy,
)

for index, message in enumerate(st.session_state.messages):
    role = "assistant" if message["role"] == "note" else message["role"]
    with st.chat_message(role, avatar=AVATARS.get(role)):
        if message["role"] == "user":
            st.markdown(message["content"] or "_(image only)_")
            if message.get("image"):
                st.image(message["image"], width=280, alt="Image attached by the student")
        elif message["role"] == "note":
            st.markdown(message["content"])
        else:
            render_assistant(message, index)


# ------------------------------------------------------------------- input

if st.session_state.voice_pending:
    fixes = st.session_state.voice_corrections
    st.info("Check the transcript in the box below, correct it if needed, then send."
            + ("  \nTechnical terms corrected: " + ", ".join(f"*{a}* → **{b}**" for a, b in fixes)
               if fixes else ""), icon=":material/mic:")

# Photo from the laptop camera (handwritten notes, a textbook page), attached to
# the next question. On phones the upload button in the chat box offers the camera.
with st.popover("Take a photo", icon=":material/photo_camera:"):
    photo = st.camera_input("Photo of your question, notes or diagram",
                            key=f"camera_{st.session_state.get('camera_round', 0)}")
    if photo is not None:
        st.session_state.camera_image = photo.getvalue()
        st.caption("The photo will be attached to your next question.")
if st.session_state.get("camera_image"):
    st.caption(":material/photo_camera: A camera photo is attached to your next question.")

prompt = st.chat_input(
    "Ask an academic question...",
    key="chat",
    accept_file=True,
    file_type=IMAGE_TYPES + DOCUMENT_TYPES,
    accept_audio=status["asr"],
    max_upload_size=MAX_DOCUMENT_BYTES // (1024 * 1024),
    max_chars=settings.MAX_QUESTION_CHARS,
    submit_mode="disable",
)

if not st.session_state.messages and prompt is None:
    examples = {
        ":material/local_fire_department: Rankine cycle": "Explain the Rankine cycle used in steam power plants.",
        ":material/electric_bolt: Induction motor": "How does a three-phase induction motor work?",
        ":material/foundation: Foundations": "What are the different types of foundations in building construction?",
        ":material/school: Backpropagation": "Explain backpropagation in neural networks.",
    }
    choice = st.pills("Try asking", list(examples), label_visibility="collapsed")
    if choice:
        prompt = examples[choice]

if prompt:
    text = prompt if isinstance(prompt, str) else (prompt.text or "")
    text = text[:settings.MAX_QUESTION_CHARS]
    files = [] if isinstance(prompt, str) else list(prompt.files or [])
    audio = None if isinstance(prompt, str) else getattr(prompt, "audio", None)

    # Voice: transcribe, then let the student review the transcript (report §13.1).
    if audio is not None and not text.strip():
        with st.spinner("Transcribing..."):
            try:
                subject = subject_record(subject_id) if subject_id else None
                transcript = get_recognizer().transcribe(
                    audio.getvalue(), ".wav",
                    subject_name=subject["name"] if subject else (branches.get(branch) if branch else None),
                    topics=subject["topics"] if subject else None,
                )
            except Exception as error:
                st.error(f"Speech recognition failed: {error}")
                st.stop()
        if not transcript.text:
            st.warning("No speech was detected. Please try again.")
            st.stop()
        st.session_state.chat = transcript.text
        st.session_state.voice_corrections = transcript.corrections
        st.session_state.voice_pending = True
        st.rerun()

    # Counted once the question is sent (a voice transcript is not a question yet).
    rate_limiter, answer_slots = load_limits()
    if not rate_limiter.allow(st.session_state.session_key):
        wait = rate_limiter.retry_after(st.session_state.session_key)
        st.warning(f"You have reached the limit of {settings.MAX_QUESTIONS_PER_HOUR} questions an hour. "
                   f"Please try again in about {max(1, wait // 60)} minutes.", icon=":material/hourglass:")
        st.stop()

    from_voice = st.session_state.voice_pending
    st.session_state.voice_pending = False
    images = [f for f in files if f.name.rsplit(".", 1)[-1].lower() in IMAGE_TYPES]
    documents = [f for f in files if f.name.rsplit(".", 1)[-1].lower() in DOCUMENT_TYPES]
    image_bytes = images[0].getvalue() if images else None
    if image_bytes is None and st.session_state.get("camera_image"):
        image_bytes = st.session_state.camera_image
    if st.session_state.get("camera_image"):
        # A new widget key empties the camera, so the photo is used once only.
        st.session_state.camera_image = None
        st.session_state.camera_round = st.session_state.get("camera_round", 0) + 1
    if images and images[0].size > settings.MAX_IMAGE_BYTES:
        st.error(f"Images are limited to {settings.MAX_IMAGE_BYTES // (1024 * 1024)} MB.")
        st.stop()

    # A document is read once and kept for the rest of the conversation.
    if documents:
        doc_file = documents[0]
        with st.spinner(f"Reading {doc_file.name}..."):
            try:
                st.session_state.document = build_session_document(
                    doc_file.name, doc_file.getvalue(), get_retriever().model)
            except DocumentError as error:
                st.error(str(error))
                st.stop()
        doc = st.session_state.document
        if not text.strip() and image_bytes is None:
            st.session_state.messages.append({"role": "user", "content": f"Uploaded {doc.name}", "image": None})
            st.session_state.messages.append({"role": "note", "content":
                f"Read **{doc.name}**: {doc.pages} pages, {doc.words:,} words. Ask anything about it, "
                "for example *Summarise this document* or a question on a topic it covers. "
                "Answers will cite it by page."})
            st.rerun()

    st.session_state.messages.append({"role": "user", "content": text, "image": image_bytes})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(text or "_(image only)_")
        if image_bytes:
            st.image(image_bytes, width=280, alt="Image attached by the student")

    history = [
        {"question": m["response"]["question"], "answer": m["response"]["answer"]}
        for m in st.session_state.messages if m["role"] == "assistant"
    ]
    filters = {"subject_id": subject_id} if subject_id else ({"discipline": branch} if branch else None)
    # Subjects overlap, so widen subject -> branch -> everything when the
    # selected scope has no close match (otherwise the answer would be ungrounded).
    fallback_filters = [{"discipline": branch}, None] if subject_id else [None]
    subject_context = {
        "Branch": branches.get(branch) if branch else None,
        "Subject": subjects[subject_id] if subject_id else None,
    } if branch else None

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        thinking = st.status(":shimmer[Thinking]", type="compact")
        # The answer appears word by word while it is generated; sources and
        # verification are added when it is complete.
        live = st.empty()
        last_draw = [0.0]

        def show_partial(text_so_far):
            now = time.monotonic()
            if now - last_draw[0] >= 0.08:          # ~12 refreshes a second
                live.markdown(strip_prompt_echo(text_so_far) + " ▌")
                last_draw[0] = now

        pipeline = get_pipeline(strategy, cost_preference)
        if answer_slots.limit and answer_slots.busy >= answer_slots.limit:
            thinking.update(label=":shimmer[Many students are asking right now - you are in the queue]")
        try:
            with answer_slots.acquire(settings.QUEUE_TIMEOUT) as got_slot:
                if not got_slot:
                    thinking.update(label="Server busy", state="error")
                    st.warning("The server is busy answering other students. Please send your question "
                               "again in a minute.", icon=":material/hourglass:")
                    st.stop()
                thinking.update(label=":shimmer[Thinking]")
                response = pipeline.ask(
                    text, image_bytes=image_bytes, answer_mode=answer_mode, history=history,
                    filters=filters, subject_context=subject_context, from_voice=from_voice,
                    fallback_filters=fallback_filters, on_token=show_partial,
                    document=st.session_state.get("document"),
                ).to_dict()
        except ValueError as error:
            thinking.update(label="Could not answer", state="error")
            st.error(str(error))
            st.stop()
        except Exception:
            reference = uuid.uuid4().hex[:8]
            logger.exception("Answer failed (reference %s)", reference)
            thinking.update(label="Could not answer", state="error")
            live.empty()
            st.error(f"Something went wrong while answering. Please try again. "
                     f"If it keeps happening, give this reference to the team: **{reference}**.")
            st.stop()
        live.empty()
        first = response["telemetry"]["stages_ms"].get("first_token")
        thinking.update(
            label=f"{WORKFLOW_LABELS.get(response['workflow'], response['workflow'])} - "
                  f"{response['telemetry']['total_ms'] / 1000:.1f} s"
                  + (f" (first words after {first / 1000:.1f} s)" if first else ""),
            state="complete",
        )

        audio_bytes = None
        if speak:
            try:
                from src.voice.speech import synthesize
                audio_bytes, _ = synthesize(response["answer"])
            except Exception as error:
                response["notices"].append(f"Spoken answer unavailable: {error}")

        message = {"role": "assistant", "response": response, "audio": audio_bytes}
        st.session_state.messages.append(message)
        render_assistant(message, len(st.session_state.messages) - 1)
