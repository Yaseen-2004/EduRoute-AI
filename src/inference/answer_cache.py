"""
Semantic answer cache: a question already answered is answered again at once.

In a class, many students ask the same question within hours ("Explain the
Rankine cycle"); on this CPU each answer costs 1-3 minutes of generation. The
cache returns a stored answer when a new question is the same question:

- exact tier: identical after normalising case, spacing and punctuation;
- semantic tier: MiniLM cosine >= ANSWER_CACHE_SIMILARITY AND the same set of
  content words, short ones included. The word condition keeps "advantages of
  AC motors" and "advantages of DC motors" (cosine ~0.97) apart, and "4-pole"
  from "6-pole": a rewording may change the filler words and their order, but
  never what is asked about. (A test caught that a 3-letter minimum let AC/DC
  through.)

Only the same answer mode, scope (branch/subject filter) and routing strategy
match. Questions with an image, an uploaded document or follow-up context are
never cached, nor are failed, extractive or warning-carrying answers.

Persistence (14 Nov): entries are appended to a JSON-lines file and reloaded
at start-up, so a restart no longer empties the cache. Every record carries the
index, model and prompt version it was made with; a record from another
version is dropped, so a changed knowledge base or model never serves stale
answers. Question-bank answers (src/inference/question_bank.py) are PINNED:
never expired, never evicted.

Thread-safe; shared by every session on a server.
"""

import copy
import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from config import settings
from src.utils.text import content_words

# Bump when prompts or answer formats change in a way that should retire cached answers.
CACHE_FORMAT = 3


def normalise(question):
    return re.sub(r"[^a-z0-9 ]+", "", re.sub(r"\s+", " ", (question or "").lower())).strip()


def _h(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def question_key(question):
    """One-way hash of the normalised question: the exact tier never needs the wording."""
    return _h(normalise(question))


# Words the general stoplist drops but that change what answer is wanted:
# "Compare AC and DC motors" must not match "Explain AC and DC motors".
INTENT_MARKERS = {
    "compare": "compare", "comparison": "compare", "difference": "compare", "differences": "compare",
    "differentiate": "compare", "distinguish": "compare", "contrast": "compare", "versus": "compare",
    "vs": "compare", "between": "compare", "why": "why", "derive": "derive", "derivation": "derive",
    "prove": "derive", "proof": "derive", "calculate": "calculate", "compute": "calculate",
    "find": "calculate", "example": "example", "examples": "example",
}


def word_set(question):
    """Hashed content words (short ones included) plus intent markers, for the same-words guard."""
    words = set(content_words(question, min_len=1))
    words |= {f"<{INTENT_MARKERS[w]}>" for w in re.findall(r"[a-z]+", (question or "").lower())
              if w in INTENT_MARKERS}
    return frozenset(_h(w) for w in words)


def _scrub(response):
    """A response without the student's wording (§21): what is saved to disk."""
    clean = copy.deepcopy(response)
    clean["question"] = ""
    analysis = (clean.get("decision") or {}).get("analysis")
    if isinstance(analysis, dict):
        analysis["question"] = None
    if isinstance(clean.get("telemetry"), dict):
        clean["telemetry"].pop("memory", None)
        clean["telemetry"]["agent_trace"] = [   # its queries are the student's words
            {k: s.get(k) for k in ("action", "found", "top_score", "coverage")}
            for s in clean["telemetry"].get("agent_trace") or []]
    return clean


def cache_version():
    return f"{settings.RAG_VERSION}|{settings.LLM_SERVED_MODEL}|{CACHE_FORMAT}"


def _tuple(value):
    return tuple(_tuple(v) for v in value) if isinstance(value, list) else value


class SemanticAnswerCache:

    def __init__(self, embedder=None, max_entries=None, ttl_hours=None, similarity=None, path=None):
        self.embedder = embedder
        self.max_entries = max_entries or settings.ANSWER_CACHE_SIZE
        self.ttl_s = 3600 * (ttl_hours if ttl_hours is not None else settings.ANSWER_CACHE_TTL_HOURS)
        self.similarity = similarity or settings.ANSWER_CACHE_SIMILARITY
        self.path = Path(path) if path else None
        # key -> {"stored", "vector", "words", "response", "pinned", "question"}
        self._entries = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0
        if self.path is not None:
            self._load()

    # ------------------------------------------------------------ helpers
    @staticmethod
    def signature(answer_mode, filters, strategy):
        scope = tuple(sorted((filters or {}).items()))
        return (answer_mode, scope, strategy)

    def _vector(self, question):
        if self.embedder is None:
            return None
        v = self.embedder.encode([question], normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)[0]

    def _expire(self, now):
        for key in [k for k, e in self._entries.items() if not e["pinned"] and now - e["stored"] > self.ttl_s]:
            del self._entries[key]

    def _evict(self):
        unpinned = [k for k, e in self._entries.items() if not e["pinned"]]
        while len(unpinned) > self.max_entries:
            del self._entries[unpinned.pop(0)]            # least recently used first

    # ------------------------------------------------------------ disk
    def _load(self, compact=True):
        """Read the saved cache. `compact=False`: only merge in lines other processes
        added (the question-bank builder runs beside the app), never rewrite."""
        if not self.path.exists():
            return
        self._seen = self.path.stat().st_mtime_ns
        now, version, records = time.time(), cache_version(), {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                                   # a half-written last line after a crash
            if r.get("version") != version:
                continue
            if not r.get("pinned") and now - r.get("stored", 0) > self.ttl_s:
                continue
            records[(_tuple(r["signature"]), r["qkey"])] = r   # later lines win
        for key, r in records.items():
            if not compact and key in self._entries and self._entries[key]["stored"] >= r["stored"]:
                continue                                   # ours is as new: keep its place in the LRU
            vector = np.asarray(r["vector"], dtype=np.float32) if r.get("vector") else None
            self._entries[key] = {"stored": r["stored"], "vector": vector, "pinned": bool(r.get("pinned")),
                                  "words": frozenset(r["words"]), "response": r["response"],
                                  "question": r.get("question")}
        self._evict()
        if compact:
            self._rewrite()                                # compact: one line per live entry
            self._seen = self.path.stat().st_mtime_ns

    def _sync(self):
        """Pick up answers another process saved since we last looked (cheap stat check)."""
        if self.path is None or not self.path.exists():
            return
        if self.path.stat().st_mtime_ns != getattr(self, "_seen", None):
            self._load(compact=False)

    def _record(self, key, entry):
        """A saved line. Student questions are kept only as hashes and a meaning vector;
        question-bank entries (pinned, written by us) keep their text for readability."""
        vector = entry["vector"]
        return {"version": cache_version(), "signature": key[0], "qkey": key[1],
                "question": entry["question"] if entry["pinned"] else None,
                "words": sorted(entry["words"]), "stored": entry["stored"], "pinned": entry["pinned"],
                "vector": [round(float(x), 5) for x in vector] if vector is not None else None,
                "response": entry["response"] if entry["pinned"] else _scrub(entry["response"])}

    def _rewrite(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for key, entry in self._entries.items():
                f.write(json.dumps(self._record(key, entry), ensure_ascii=False) + "\n")
        tmp.replace(self.path)                             # atomic swap: never a truncated cache

    def _append(self, key, entry):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stale = self.path.exists() and self.path.stat().st_mtime_ns != getattr(self, "_seen", None)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self._record(key, entry), ensure_ascii=False) + "\n")
        if not stale:
            self._seen = self.path.stat().st_mtime_ns     # our own write is not news

    # ------------------------------------------------------------ API
    def get(self, question, signature, vector=None):
        """The cached response dict, or None."""
        now = time.time()
        key = (signature, question_key(question))
        with self._lock:
            self._sync()
            self._expire(now)
            if key in self._entries:
                self._entries.move_to_end(key)
                self.hits += 1
                return self._entries[key]["response"]
        vector = vector if vector is not None else self._vector(question)
        if vector is not None:
            words = word_set(question)
            with self._lock:
                best, best_key = 0.0, None
                for k, e in self._entries.items():
                    if k[0] != signature or e["vector"] is None or e["words"] != words:
                        continue
                    score = float(e["vector"] @ vector)
                    if score > best:
                        best, best_key = score, k
                if best_key is not None and best >= self.similarity:
                    self._entries.move_to_end(best_key)
                    self.hits += 1
                    return self._entries[best_key]["response"]
        with self._lock:
            self.misses += 1
        return None

    def put(self, question, signature, response, vector=None, pinned=False):
        vector = vector if vector is not None else self._vector(question)
        key = (signature, question_key(question))
        entry = {"stored": time.time(), "vector": vector, "pinned": pinned, "question": question,
                 "words": word_set(question), "response": response}
        with self._lock:
            if key in self._entries and self._entries[key]["pinned"] and not pinned:
                return                                     # a question-bank answer is never overwritten
            self._entries[key] = entry
            self._entries.move_to_end(key)
            self._evict()
            self._append(key, entry)

    def contains(self, question, signature):
        with self._lock:
            return (signature, question_key(question)) in self._entries

    def stats(self):
        total = self.hits + self.misses
        pinned = sum(e["pinned"] for e in self._entries.values())
        return {"entries": len(self._entries), "pinned": pinned, "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0}

    @staticmethod
    def cacheable(response):
        """Only clean, complete, generated answers are worth repeating."""
        return (response.is_model_output and not response.error and not response.notices
                and response.answer and response.workflow in ("direct_llm", "rag_llm"))
