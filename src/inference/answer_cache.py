"""
Semantic answer cache: a question already answered is answered again at once.

In a class, many students ask the same question within hours ("Explain the
Rankine cycle"); on this CPU each answer costs 1-3 minutes of generation. The
cache returns a stored answer when a new question is the same question:

- exact tier: identical after normalising case, spacing and punctuation;
- semantic tier: MiniLM cosine >= ANSWER_CACHE_SIMILARITY (0.97) AND the same
  set of content words, short ones included. The second condition keeps
  "advantages of AC motors" and "advantages of DC motors" (cosine ~0.97) apart,
  and "4-pole" from "6-pole": a paraphrase may reorder words, but it must not
  change what is asked about. (A test caught that a 3-letter minimum let AC/DC
  through.)

Only the same answer mode, scope (branch/subject filter) and routing strategy
match. Questions with an image, an uploaded document or follow-up context are
never cached, nor are failed, extractive or warning-carrying answers.
Bounded LRU with a time-to-live, safe to share between sessions (threads).
"""

import re
import threading
import time
from collections import OrderedDict

import numpy as np

from config import settings
from src.utils.text import content_words


def normalise(question):
    return re.sub(r"[^a-z0-9 ]+", "", re.sub(r"\s+", " ", (question or "").lower())).strip()


class SemanticAnswerCache:

    def __init__(self, embedder=None, max_entries=None, ttl_hours=None, similarity=None):
        self.embedder = embedder
        self.max_entries = max_entries or settings.ANSWER_CACHE_SIZE
        self.ttl_s = 3600 * (ttl_hours if ttl_hours is not None else settings.ANSWER_CACHE_TTL_HOURS)
        self.similarity = similarity or settings.ANSWER_CACHE_SIMILARITY
        self._entries = OrderedDict()        # key -> (stored_at, vector, words, response dict)
        self._lock = threading.Lock()
        self.hits = self.misses = 0

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
        for key in [k for k, (stored, *_rest) in self._entries.items() if now - stored > self.ttl_s]:
            del self._entries[key]

    def get(self, question, signature, vector=None):
        """The cached response dict, or None."""
        now = time.time()
        key = (signature, normalise(question))
        with self._lock:
            self._expire(now)
            if key in self._entries:
                self._entries.move_to_end(key)
                self.hits += 1
                return self._entries[key][3]
        vector = vector if vector is not None else self._vector(question)
        if vector is not None:
            words = frozenset(content_words(question, min_len=1))
            with self._lock:
                best, best_key = 0.0, None
                for k, (_, v, w, _resp) in self._entries.items():
                    if k[0] != signature or v is None or w != words:
                        continue
                    score = float(v @ vector)
                    if score > best:
                        best, best_key = score, k
                if best_key is not None and best >= self.similarity:
                    self._entries.move_to_end(best_key)
                    self.hits += 1
                    return self._entries[best_key][3]
        with self._lock:
            self.misses += 1
        return None

    def put(self, question, signature, response, vector=None):
        vector = vector if vector is not None else self._vector(question)
        key = (signature, normalise(question))
        with self._lock:
            self._entries[key] = (time.time(), vector, frozenset(content_words(question, min_len=1)), response)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)        # least recently used

    def stats(self):
        total = self.hits + self.misses
        return {"entries": len(self._entries), "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0}

    @staticmethod
    def cacheable(response):
        """Only clean, complete, generated answers are worth repeating."""
        return (response.is_model_output and not response.error and not response.notices
                and response.answer and response.workflow in ("direct_llm", "rag_llm"))
