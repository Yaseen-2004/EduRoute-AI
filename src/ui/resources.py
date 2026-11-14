"""
Resources shared by every page of the app (one copy per server process):
the knowledge base, the answer cache, the pipelines and the load limits.

Defined here, not in a page script, so that st.cache_resource gives the chat
page and the past-papers page the same index, models and answer cache instead
of loading a second copy.
"""

import streamlit as st

from config import settings
from src.utils.access import AnswerSlots, SlidingWindowLimiter


@st.cache_resource(show_spinner="Loading the knowledge base...")
def get_retriever():
    from src.rag.retrieval import EduRouteRetriever
    return EduRouteRetriever(version=settings.RAG_VERSION, mode=settings.RETRIEVAL_MODE)


@st.cache_resource(show_spinner=False)
def get_answer_cache():
    """One answer cache for every session on this server: a question one student
    asked is answered instantly for the next. Saved to disk and pre-filled by the
    question bank."""
    if not settings.ANSWER_CACHE:
        return None
    from src.inference.answer_cache import SemanticAnswerCache
    return SemanticAnswerCache(get_retriever().model, path=settings.ANSWER_CACHE_PATH)


@st.cache_resource(show_spinner=False, max_entries=10)
def get_pipeline(strategy="adaptive", cost_preference="balanced"):
    from src.pipeline import EduRoutePipeline
    return EduRoutePipeline(strategy=strategy, cost_preference=cost_preference,
                            retriever=get_retriever(), answer_cache=get_answer_cache())


@st.cache_resource(show_spinner=False)
def load_limits():
    """Hourly question limit per session and the server-wide answer queue."""
    return (SlidingWindowLimiter(settings.MAX_QUESTIONS_PER_HOUR, 3600),
            AnswerSlots(settings.MAX_CONCURRENT_ANSWERS))
