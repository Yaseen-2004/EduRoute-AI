"""
Correct technical terms in speech transcripts (report §13.3: technical terms
are the known weakness of speech recognition).

Whisper writes what a term sounds like: "Homan transfer orbit", "back
propagation". The knowledge base knows the right spellings (its article titles
and the catalogue topics), so a transcript phrase that closely resembles a
known multi-word term, or has the same letters with different spacing, is
replaced by it.

Conservative on purpose (a wrong "correction" changes what the student said):
- only phrases of two or more words, replaced by a term with the same number of
  words (or the same letters joined, "back propagation" -> "backpropagation");
- small words must be identical ("application in AI" is never turned into
  "applications of AI"), and singular/plural differences are not errors;
- the differing words must actually look alike (Homan ~ Hohmann);
- the student's capitalisation is kept.
"""

import re
from difflib import SequenceMatcher
from functools import lru_cache

MIN_SIMILARITY = 0.85        # whole phrase
MIN_WORD_SIMILARITY = 0.6    # each differing word
SMALL_WORDS = set("a an the of in on for to and or with by at from as is are its it this that "
                  "into over under via per".split())


def _norm(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _same_word(a, b):
    """Equal up to singular/plural."""
    return a == b or a.rstrip("s") == b.rstrip("s") or a.rstrip("es") == b.rstrip("es")


@lru_cache(maxsize=1)
def vocabulary():
    """Known technical terms: knowledge-base article titles + catalogue topics."""
    terms = set()
    try:
        import json
        from config import settings
        from src.utils.catalogue_loader import load_catalogue
        for subject in load_catalogue()["subjects"]:
            terms.update(subject["topics"])
        meta = settings.VECTOR_STORE_DIR / "engineering_faiss_v1_metadata.jsonl"
        if meta.exists():
            with meta.open(encoding="utf-8") as fh:
                for line in fh:
                    title = json.loads(line).get("article_title")
                    if title:
                        terms.add(title)
    except Exception:
        pass
    cleaned = {}
    for term in terms:
        term = re.sub(r"\s*\([^)]*\)", "", term).strip()          # "Cracking (chemistry)" -> "Cracking"
        n = _norm(term)
        if len(n) >= 6:
            cleaned[n] = term
    return cleaned


def _is_correction(heard, key):
    """`heard` and `key` have the same number of words: is `key` a fix of a mis-hearing?"""
    differing = [(h, k) for h, k in zip(heard.split(), key.split()) if not _same_word(h, k)]
    if not differing:
        return False                                              # only case/plural differences
    for h, k in differing:
        if h in SMALL_WORDS or k in SMALL_WORDS:
            return False
        if SequenceMatcher(None, h, k).ratio() < MIN_WORD_SIMILARITY:
            return False
    return SequenceMatcher(None, heard, key).ratio() >= MIN_SIMILARITY


def _match_case(original, target):
    if original[:1].islower() and target[:1].isupper() and not target.split()[0].isupper() \
            and target.split()[0][1:].islower():
        return target[0].lower() + target[1:]
    return target


def correct_terms(text, vocab=None):
    """Return (corrected_text, [(heard, corrected), ...])."""
    vocab = vocabulary() if vocab is None else vocab
    if not text or not vocab:
        return text, []
    by_nospace = {k.replace(" ", ""): v for k, v in vocab.items()}
    by_length = {}
    for key in vocab:
        by_length.setdefault(len(key.split()), []).append(key)

    tokens = re.findall(r"\S+", text)
    out, changes, i = [], [], 0
    while i < len(tokens):
        replaced = False
        for size in (4, 3, 2):
            if i + size > len(tokens):
                continue
            span = tokens[i:i + size]
            lead = re.match(r"^[^\w]*", span[0]).group(0)
            trail = re.search(r"[^\w]*$", span[-1]).group(0)
            heard = _norm(" ".join(span))
            if len(heard.split()) != size or heard in vocab:
                continue
            target = None
            joined = heard.replace(" ", "")
            if len(joined) >= 8 and joined in by_nospace and " " not in _norm(by_nospace[joined]):
                target = by_nospace[joined]                          # "back propagation"
            else:
                best, best_ratio = None, 0.0
                for key in by_length.get(size, []):
                    if key[0] != heard[0] or abs(len(key) - len(heard)) > 4:
                        continue
                    if _is_correction(heard, key):
                        ratio = SequenceMatcher(None, heard, key).ratio()
                        if ratio > best_ratio:
                            best, best_ratio = key, ratio
                if best:
                    target = vocab[best]
            if target:
                fixed = _match_case(span[0].lstrip(lead), target)
                out.append(lead + fixed + trail)
                changes.append((" ".join(span).strip(lead + trail) if (lead or trail) else " ".join(span), fixed))
                i += size
                replaced = True
                break
        if not replaced:
            out.append(tokens[i])
            i += 1
    return " ".join(out), changes
