"""
Operations dashboard: how the deployed assistant is being used and how well it
is doing. Reads the privacy-preserving logs (hashes, timings, routes; no
question text unless consented logging is on). Team only: same code as the studio.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitAPIException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root, however started

from config import settings  # noqa: E402
from src.utils.access import require_access  # noqa: E402

st.set_page_config(page_title="Operations", page_icon=":material/monitoring:", layout="wide")
require_access("studio", settings.STUDIO_CODE, title="Operations dashboard")

WORKFLOWS = {"direct_llm": "Direct answer", "rag_llm": "Knowledge base", "vision_llm": "Image",
             "hybrid_multimodal_rag": "Image + knowledge base"}

with st.sidebar:
    try:
        st.page_link("app.py", label="Ask EduRoute", icon=":material/forum:")
        st.page_link("pages/4_Past_Papers.py", label="Past papers", icon=":material/history_edu:")
        if settings.ENABLE_STUDIO:
            st.page_link("pages/2_Evaluation_Studio.py", label="Evaluation studio", icon=":material/fact_check:")
        st.page_link("pages/3_Operations_Dashboard.py", label="Operations", icon=":material/monitoring:")
    except StreamlitAPIException:
        pass


@st.cache_data(ttl=60, show_spinner=False)
def load(name):
    path = settings.LOG_DIR / name
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    if "timestamp" in frame:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    return frame


st.title("Operations")
telemetry, feedback, errors = load("telemetry.jsonl"), load("feedback.jsonl"), load("errors.jsonl")
if telemetry.empty:
    st.info("No questions have been answered on this server yet.")
    st.stop()

days = st.segmented_control("Period", [1, 7, 30], default=7, format_func=lambda d: f"Last {d} day{'s' * (d > 1)}") or 7
since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
t = telemetry[telemetry["timestamp"] >= since]
f = feedback[feedback["timestamp"] >= since] if not feedback.empty else feedback
e = errors[errors["timestamp"] >= since] if not errors.empty else errors
st.caption(f"Logs keep {settings.LOG_RETENTION_DAYS or 'all'} days · question text is "
           f"{'logged (consented evaluation)' if settings.LOG_QUERY_TEXT else 'never logged, only a hash'} · "
           "refreshed every minute")
if t.empty:
    st.info("No questions in this period.")
    st.stop()

seconds = t["total_ms"] / 1000
helpful = (f["rating"] == "helpful").mean() if not f.empty else None
grounded = t["workflow"].isin(["rag_llm", "hybrid_multimodal_rag"])
cols = st.columns(5)
cols[0].metric("Questions", f"{len(t):,}")
cols[1].metric("Median answer time", f"{seconds.median():.1f} s", help="95th percentile: "
               f"{seconds.quantile(0.95):.1f} s")
cols[2].metric("Answered from sources", f"{grounded.mean():.0%}")
cols[3].metric("Rated helpful", "–" if helpful is None else f"{helpful:.0%}",
               help=f"{len(f)} ratings" if not f.empty else "No ratings yet")
cols[4].metric("Failed answers", f"{len(e)}", help="Each failure has a reference code in run/app.log")

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.markdown("**Questions per day**")
        per_day = t.set_index("timestamp").resample("D").size().rename("Questions")
        per_day.index = per_day.index.date
        st.bar_chart(per_day, height=240, x_label="", y_label="Questions")
with right:
    with st.container(border=True):
        st.markdown("**How questions were answered**")
        mix = t["workflow"].map(lambda w: WORKFLOWS.get(w, w)).value_counts().rename("Questions")
        st.bar_chart(mix, horizontal=True, height=240, x_label="Questions", y_label="")

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.markdown("**Answer time by route** (seconds)")
        speed = (t.assign(route=t["workflow"].map(lambda w: WORKFLOWS.get(w, w)), s=seconds)
                 .groupby("route")["s"].agg(Questions="size", Median="median",
                                            P95=lambda x: x.quantile(0.95)).round(1)
                 .sort_values("Questions", ascending=False))
        st.dataframe(speed, width="stretch", column_config={
            "P95": st.column_config.NumberColumn("95th percentile")})
with right:
    with st.container(border=True):
        st.markdown("**Answer quality signals**")
        verified = t[t["verified"] == True]  # noqa: E712 (pandas column)
        rows = {
            "Answers verified against sources": f"{len(verified) / len(t):.0%}",
            "Mean unsupported-claim rate (verified)": "–" if verified.empty else
            f"{verified['unsupported_claim_rate'].dropna().mean():.0%}",
            "Answers revised after checking": f"{t['revised'].fillna(False).mean():.0%}",
            "Grounded answers citing a source": "–" if not grounded.any() else
            f"{(t.loc[grounded, 'cited'] > 0).mean():.0%}",
            "Mean output tokens": f"{t['output_tokens'].mean():.0f}",
        }
        if "cache_hit" in t:
            rows["Answered instantly from the answer cache"] = f"{t['cache_hit'].fillna(False).astype(bool).mean():.0%}"
        if "context" in t:
            saved = [c.get("saved_pct") for c in t["context"] if isinstance(c, dict) and c]
            if saved:
                rows["Prompt tokens saved by context engineering (median)"] = f"{pd.Series(saved).median():.0f}%"
        if "agent_trace" in t:
            refined = [any(s.get("action") == "refine" for s in tr) for tr in t["agent_trace"]
                       if isinstance(tr, list) and tr]
            if refined:
                rows["Agentic retrieval needed a second search"] = f"{pd.Series(refined).mean():.0%}"
        st.dataframe(pd.DataFrame(rows.items(), columns=["Signal", "Value"]), hide_index=True, width="stretch")

if not e.empty:
    with st.expander(f"Failed answers ({len(e)})", icon=":material/error:"):
        st.dataframe(e.sort_values("timestamp", ascending=False), hide_index=True, width="stretch")
