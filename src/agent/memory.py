"""
Conversation memory: what the model remembers of the conversation, within a budget.

Before this, the prompt carried the last 3 turns as raw text (up to 600
characters of each answer), whatever their relevance, and retrieval saw only
the new question. A follow-up such as "What are its advantages?" was searched
on its own words and found unrelated passages.

Memory has three tiers:
- working memory: the most recent turns, each answer reduced to its key
  sentences, newest first until the token budget is spent;
- episodic summary: one line naming the topics of older turns, so the model
  knows what was covered without paying for the full text;
- topic state: the subject of the latest turn, used to rewrite follow-up
  questions into self-contained search queries.

Everything is bounded: at most MAX_TURNS turns are considered, and the
rendered memory never exceeds the token budget.
"""

import re

from config import settings
from src.utils.text import content_words, estimate_tokens, split_sentences

MAX_TURNS = 20
ANSWER_SENTENCES = 2

# "it", "its", "this", "they"... at the start, or short questions that only
# make sense with the previous turn ("why?", "give an example", "explain more").
FOLLOW_UP = re.compile(
    r"^\s*(and\s+)?(what|how|why|where|when|which|can|could|is|are|does|do|explain|give|list|tell)?\b.{0,25}"
    r"\b(it|its|this|that|they|them|their|these|those|the same|the above|above)\b"
    r"|^\s*(why|how so|and|also|then|more|explain more|elaborate|example|examples|give (an|one|some)? ?examples?"
    r"|what about|how about|continue|go on|in detail|tell me more)\b",
    re.IGNORECASE,
)
CITATION = re.compile(r"\s*\[\d{1,2}\](\[\d{1,2}\])*")


def _strip_markup(text):
    text = CITATION.sub("", text or "")
    return re.sub(r"[#*_`>|]+", " ", text)


def topic_of(turn):
    """The subject of a turn: the content words of its question (first 6)."""
    words = content_words(turn.get("question", ""))
    return " ".join(dict.fromkeys(words))[:80] if words else ""


def is_follow_up(question, history):
    if not history or not question:
        return False
    words = content_words(question)
    return bool(FOLLOW_UP.search(question)) or len(words) <= 1


def contextualize(question, history):
    """A self-contained search query for a follow-up question; otherwise the question.
    Returns (query, topic_or_None)."""
    if not settings.CONVERSATION_MEMORY or not is_follow_up(question, history):
        return question, None
    topic = ""
    for turn in reversed(history[-MAX_TURNS:]):
        topic = topic_of(turn)
        if topic:
            break
    if not topic:
        return question, None
    return f"{question} (about: {topic})", topic


def _key_sentences(answer, n=ANSWER_SENTENCES):
    sentences = [s for s in split_sentences(_strip_markup(answer)) if len(s.split()) >= 4]
    return " ".join(sentences[:n])


def render(history, budget_tokens=None):
    """The memory block for the prompt, newest turns first within the budget,
    older turns folded into a one-line topic summary."""
    if not history:
        return ""
    budget = budget_tokens or settings.MEMORY_BUDGET_TOKENS
    turns = history[-MAX_TURNS:]
    kept, used = [], 0
    for turn in reversed(turns):
        block = f"Student: {turn['question']}\nTutor (summary): {_key_sentences(turn['answer'])}"
        cost = estimate_tokens(block)
        if kept and used + cost > budget:
            break
        kept.append(block)
        used += cost
    older = turns[: len(turns) - len(kept)]
    summary = ""
    if older:
        topics = [t for t in dict.fromkeys(topic_of(t) for t in older) if t]
        summary = "Earlier topics in this conversation: " + "; ".join(topics[-8:]) + "\n"
    return "Earlier in this conversation:\n" + summary + "\n".join(reversed(kept)) + "\n\n"


def memory_stats(history, rendered):
    raw = sum(estimate_tokens(t["question"]) + estimate_tokens(t["answer"][:600]) for t in (history or [])[-3:])
    return {"turns": len(history or []), "tokens": estimate_tokens(rendered) if rendered else 0,
            "tokens_previous_method": raw}
