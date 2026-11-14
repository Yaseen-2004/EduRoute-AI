"""Small text helpers shared by context engineering, memory, caching and the agent."""

import re

STOPWORDS = frozenset("""
a an the and or but if then else of to in on at by for with from into onto over under about as is are was were be
been being do does did done has have had having it its this that these those they them their there here what which
who whom whose when where why how can could should would may might must shall will yes than so such very
also just only more most less least much many some any each every all both either neither other another same own
please explain describe define discuss write give list state briefly detail details short answer question mark marks
between difference differences compare comparison versus vs using use used work works working me my we our you your
i about tell know need want help
""".split())

WORD = re.compile(r"[a-z0-9][a-z0-9\-]*", re.IGNORECASE)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


def estimate_tokens(text):
    """Qwen tokens for English text, ~4 characters each (no tokenizer load needed)."""
    return max(1, len(text or "") // 4)


def _stem(word):
    for suffix in ("ies", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def content_words(text, min_len=3):
    """Lower-cased, lightly stemmed words that carry meaning (no stopwords)."""
    return [_stem(w) for w in (m.lower() for m in WORD.findall(text or ""))
            if len(w) >= min_len and w not in STOPWORDS]


def split_sentences(text):
    return [s.strip() for s in SENTENCE_END.split(text or "") if s.strip()]


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a and b else 0.0
