"""
Context engineering: what goes into the prompt, and how much of it.

On this project's CPU the model reads prompts at ~20 tokens/s, so every source
token costs ~50 ms before the first word appears, and a small model also
answers worse when relevant sentences are buried in long passages. The builder:

1. removes near-duplicate passages (same chunk, or >= 80% shared content words);
2. compresses each passage to the sentences most relevant to the question,
   always keeping its first sentence (what the passage is about), in the
   original order so the text still reads naturally;
3. fits the result to a token budget per answer mode, dropping the
   lowest-ranked passages first (never the best one).

Only the PROMPT copy is compressed (`prompt_text`); verification and the
sources shown to the student keep the full passages, so a claim can still be
checked against everything that was retrieved.
"""

from dataclasses import dataclass, field

import numpy as np

from config import settings
from src.utils.text import content_words, estimate_tokens, jaccard, split_sentences

DUPLICATE_OVERLAP = 0.8
MIN_SENTENCE_SCORE = 0.25      # MiniLM cosine below which a sentence is off-topic


@dataclass
class ContextStats:
    passages_in: int = 0
    passages_out: int = 0
    duplicates_removed: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    budget: int = 0
    compressed: list = field(default_factory=list)

    def to_dict(self):
        saved = self.tokens_before - self.tokens_after
        return {**self.__dict__, "tokens_saved": saved,
                "saved_pct": round(100 * saved / self.tokens_before, 1) if self.tokens_before else 0.0}


class ContextBuilder:

    def __init__(self, embedder=None):
        self.embedder = embedder          # a SentenceTransformer, or None for lexical scoring

    # ------------------------------------------------------------ scoring
    def _sentence_scores(self, question, sentences):
        if self.embedder is not None:
            vectors = self.embedder.encode([question] + sentences, normalize_embeddings=True,
                                           show_progress_bar=False)
            vectors = np.asarray(vectors)
            return [float(v) for v in vectors[1:] @ vectors[0]]
        q = set(content_words(question))
        return [len(q & set(content_words(s))) / (len(q) or 1) for s in sentences]

    # ------------------------------------------------------------ steps
    @staticmethod
    def deduplicate(passages):
        kept, words, removed = [], [], 0
        seen_chunks = set()
        for p in passages:
            chunk = p.get("chunk_id")
            w = content_words(p["text"])
            if (chunk and chunk in seen_chunks) or any(jaccard(w, k) >= DUPLICATE_OVERLAP for k in words):
                removed += 1
                continue
            seen_chunks.add(chunk)
            kept.append(p)
            words.append(w)
        return kept, removed

    def compress(self, question, passage, max_tokens):
        """The passage's first sentence plus its most relevant sentences, in order."""
        text = passage["text"]
        if estimate_tokens(text) <= max_tokens:
            return text
        sentences = split_sentences(text)
        if len(sentences) <= 2:
            return text[: max_tokens * 4].rsplit(" ", 1)[0] + " ..."
        scores = self._sentence_scores(question, sentences)
        chosen, used = {0}, estimate_tokens(sentences[0])
        for i in sorted(range(1, len(sentences)), key=lambda j: scores[j], reverse=True):
            cost = estimate_tokens(sentences[i])
            if scores[i] < MIN_SENTENCE_SCORE and len(chosen) > 1:
                break
            if used + cost > max_tokens:
                continue
            chosen.add(i)
            used += cost
        parts, last = [], -1
        for i in sorted(chosen):
            if last >= 0 and i != last + 1:
                parts.append("...")
            parts.append(sentences[i])
            last = i
        return " ".join(parts)

    # ------------------------------------------------------------ main
    def build(self, question, passages, answer_mode="5-mark", budget=None):
        """Returns (passages for the prompt, each with `prompt_text`; ContextStats)."""
        budget = budget or settings.CONTEXT_BUDGET.get(answer_mode, 750)
        stats = ContextStats(passages_in=len(passages), budget=budget,
                             tokens_before=sum(estimate_tokens(p["text"]) for p in passages))
        unique, stats.duplicates_removed = self.deduplicate(passages)
        if not unique:
            return [], stats
        # Each passage gets a fair share; the best passage may use a little more.
        share = max(80, budget // len(unique))
        selected, used = [], 0
        for rank, passage in enumerate(unique):
            allowance = int(share * 1.5) if rank == 0 else share
            text = self.compress(question, passage, allowance)
            cost = estimate_tokens(text)
            if selected and used + cost > budget:
                break                                    # lower-ranked passages are dropped first
            if text != passage["text"]:
                stats.compressed.append(passage.get("chunk_id") or passage.get("article_title"))
            selected.append({**passage, "prompt_text": text})
            used += cost
        stats.passages_out, stats.tokens_after = len(selected), used
        return selected, stats
