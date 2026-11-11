"""
EduRoute AI - versioned RAG retrieval engine.

Modes
  dense   FAISS cosine similarity only (the V3 behaviour in the report).
  hybrid  Dense + BM25 lexical scores combined with Reciprocal Rank Fusion.
          Helps exact-term queries (acronyms like SHAP, LIME, YOLO) that a
          small embedding model may rank below generic passages.

Optional
  filters         {"subject_id": ..., "topic": ..., "discipline": ...,
                   "article_title": ...}; each value a string or list.
                  Applied inside FAISS with an IDSelector (pre-filtering).
  rerank          Cross-encoder re-scoring of the fused top candidates.
  max_per_article Cap results per article for source diversity.

Scores: `similarity_score` is always the dense cosine similarity. It is a
ranking signal, not a calibrated probability (report §8.2, §19).
"""

import json
import logging
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

TOP_K = 5
EXPECTED_DIMENSION = 384
CANDIDATE_POOL = 50
RRF_K = 60

# version key -> (file stem, file version). "engineering_v1" covers all 26
# catalogue groups (AIML V4 articles included); v2-v4 are AIML-only.
_INDEXES = {
    "v2": ("aiml", "v2"),
    "v3": ("aiml", "v3"),
    "v4": ("aiml", "v4"),
    "engineering_v1": ("engineering", "v1"),
}
SUPPORTED_VERSIONS = tuple(_INDEXES)

INDEX_FILES = {
    k: BASE_DIR / "vector_store" / f"{stem}_faiss_{v}.index" for k, (stem, v) in _INDEXES.items()
}
METADATA_FILES = {
    k: BASE_DIR / "vector_store" / f"{stem}_faiss_{v}_metadata.jsonl" for k, (stem, v) in _INDEXES.items()
}

# A filter on a single-valued field also matches the record's multi-valued
# list, because one article can serve several branches/subjects.
LIST_FIELD = {"discipline": "disciplines", "subject_id": "subject_ids"}

FILTER_FIELDS = ("subject_id", "topic", "discipline", "article_title")

_STOPWORDS = set("""
a an the and or of to in on for with by is are was were be been being as at from that this
these those it its into how what why when which who whom explain describe define give
does do did can could would should may might will shall about between using use used
""".split())

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokenize(text):
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25Index:
    """Okapi BM25 over the chunk texts, precomputed as a sparse matrix."""

    def __init__(self, texts, k1=1.5, b=0.75):
        from scipy import sparse

        vocabulary = {}
        rows, cols, values = [], [], []
        lengths = np.zeros(len(texts), dtype=np.float32)

        for doc_id, text in enumerate(texts):
            counts = {}
            tokens = tokenize(text)
            lengths[doc_id] = len(tokens)
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                term_id = vocabulary.setdefault(token, len(vocabulary))
                rows.append(doc_id)
                cols.append(term_id)
                values.append(count)

        tf = sparse.csr_matrix(
            (np.asarray(values, dtype=np.float32), (rows, cols)),
            shape=(len(texts), len(vocabulary)),
        )
        n_docs = len(texts)
        doc_freq = np.bincount(np.asarray(cols), minlength=len(vocabulary))
        idf = np.log(1 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5)).astype(np.float32)
        avgdl = float(lengths.mean()) if n_docs else 1.0

        tf = tf.tocoo()
        norm = k1 * (1 - b + b * lengths[tf.row] / avgdl)
        weights = tf.data * (k1 + 1) / (tf.data + norm) * idf[tf.col]
        self.matrix = sparse.csc_matrix((weights, (tf.row, tf.col)), shape=tf.shape)
        self.vocabulary = vocabulary

    def scores(self, query):
        term_ids = [self.vocabulary[t] for t in tokenize(query) if t in self.vocabulary]
        if not term_ids:
            return np.zeros(self.matrix.shape[0], dtype=np.float32)
        return np.asarray(self.matrix[:, term_ids].sum(axis=1)).ravel()


class EduRouteRetriever:

    def __init__(self, top_k=TOP_K, version="v3", mode="dense", rerank=False,
                 embedding_model=None, verbose=False):
        self.top_k = top_k
        self.version = version.lower()
        self.mode = mode
        self.rerank_enabled = rerank
        self.verbose = verbose

        if self.version not in SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported version. Choose one of {SUPPORTED_VERSIONS}.")
        if mode not in ("dense", "hybrid"):
            raise ValueError("mode must be 'dense' or 'hybrid'.")

        self.index_file = INDEX_FILES[self.version]
        self.metadata_file = METADATA_FILES[self.version]

        for path in (self.index_file, self.metadata_file):
            if not path.exists():
                raise FileNotFoundError(f"Missing retrieval artifact: {path}")

        self.index = faiss.read_index(str(self.index_file))
        self.metadata = self._load_metadata()
        self._validate()

        # Reuse an already-loaded embedding model when several retrievers
        # are created (e.g. in evaluation) to avoid reloading weights.
        self.model = embedding_model or SentenceTransformer(MODEL_NAME, device="cpu")

        self._bm25 = None
        self._reranker = None

        self._log(
            f"Retriever {self.version.upper()} ready: {self.index.ntotal} vectors, "
            f"mode={self.mode}, rerank={self.rerank_enabled}"
        )

    # ------------------------------------------------------------------ setup

    def _log(self, message):
        logger.info(message)
        if self.verbose:
            print(message)

    def _load_metadata(self):
        records = []
        with self.metadata_file.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid metadata at line {line_number}: {error}") from error
        return records

    def _validate(self):
        if self.index.d != EXPECTED_DIMENSION:
            raise ValueError(f"Expected dimension {EXPECTED_DIMENSION}, received {self.index.d}")
        if self.index.ntotal == 0:
            raise ValueError("FAISS index is empty.")
        if self.index.ntotal != len(self.metadata):
            raise ValueError("FAISS index and metadata counts do not match.")
        if [m.get("vector_id") for m in self.metadata] != list(range(len(self.metadata))):
            raise ValueError("Metadata vector ordering validation failed.")
        chunk_ids = [m.get("chunk_id") for m in self.metadata]
        if any(not c for c in chunk_ids) or len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Duplicate or missing chunk IDs found.")

    @property
    def bm25(self):
        if self._bm25 is None:
            self._bm25 = BM25Index([self._lexical_text(m) for m in self.metadata])
        return self._bm25

    @staticmethod
    def _lexical_text(record):
        # Title and section are indexed lexically as well, mirroring the
        # contextual prefix used for V4 dense embeddings.
        return " ".join(filter(None, (
            record.get("article_title"), record.get("section"), record.get("text")
        )))

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANKER_NAME, device="cpu")
        return self._reranker

    # -------------------------------------------------------------- utilities

    def embed_query(self, question):
        embedding = self.model.encode(
            [question.strip()], convert_to_numpy=True, normalize_embeddings=True
        )
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.shape != (1, EXPECTED_DIMENSION) or not np.isfinite(embedding).all():
            raise ValueError("Invalid query embedding.")
        return embedding

    def allowed_ids(self, filters):
        """Vector IDs that satisfy the metadata filters, or None for no filtering."""
        if not filters:
            return None
        normalized = {}
        for field, value in filters.items():
            if field not in FILTER_FIELDS:
                raise ValueError(f"Unsupported filter '{field}'. Use one of {FILTER_FIELDS}.")
            if value in (None, "", []):
                continue
            values = value if isinstance(value, (list, tuple, set)) else [value]
            normalized[field] = {str(v).lower() for v in values}
        if not normalized:
            return None
        def matches(record, field, values):
            candidates = record.get(LIST_FIELD.get(field, ""), None) or [record.get(field, "")]
            return any(str(c).lower() in values for c in candidates)

        return np.asarray([
            i for i, record in enumerate(self.metadata)
            if all(matches(record, f, vals) for f, vals in normalized.items())
        ], dtype=np.int64)

    def _dense_search(self, query_embedding, k, allowed):
        if allowed is not None:
            if len(allowed) == 0:
                return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
            params = faiss.SearchParameters()
            params.sel = faiss.IDSelectorBatch(allowed)
            scores, ids = self.index.search(query_embedding, min(k, len(allowed)), params=params)
        else:
            scores, ids = self.index.search(query_embedding, min(k, self.index.ntotal))
        keep = ids[0] >= 0
        return scores[0][keep], ids[0][keep]

    # --------------------------------------------------------------- retrieve

    def retrieve(self, question, top_k=None, filters=None, mode=None,
                 rerank=None, max_per_article=None):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Please provide a valid question.")

        k = top_k if top_k is not None else self.top_k
        k = min(max(1, int(k)), self.index.ntotal)
        mode = mode or self.mode
        rerank = self.rerank_enabled if rerank is None else rerank

        allowed = self.allowed_ids(filters)
        query_embedding = self.embed_query(question)

        pool = max(CANDIDATE_POOL, k * 4)
        dense_scores, dense_ids = self._dense_search(query_embedding, pool, allowed)
        dense_score_of = {int(i): float(s) for s, i in zip(dense_scores, dense_ids)}

        bm25_score_of = {}
        if mode == "hybrid":
            lexical = self.bm25.scores(question)
            if allowed is not None:
                mask = np.full(lexical.shape, -np.inf, dtype=np.float32)
                mask[allowed] = 0
                lexical = lexical + mask
            lexical_ids = np.argsort(-lexical)[:pool]
            lexical_ids = [int(i) for i in lexical_ids if np.isfinite(lexical[i]) and lexical[i] > 0]
            bm25_score_of = {i: float(lexical[i]) for i in lexical_ids}

            fused = {}
            for rank, vid in enumerate(dense_ids, start=1):
                fused[int(vid)] = fused.get(int(vid), 0.0) + 1.0 / (RRF_K + rank)
            for rank, vid in enumerate(lexical_ids, start=1):
                fused[vid] = fused.get(vid, 0.0) + 1.0 / (RRF_K + rank)
            ranked = sorted(fused, key=fused.get, reverse=True)
            fused_score_of = fused
        else:
            ranked = [int(i) for i in dense_ids]
            fused_score_of = {}

        rerank_score_of = {}
        if rerank and ranked:
            candidates = ranked[:max(20, k * 3)]
            pairs = [(question, self.metadata[i]["text"]) for i in candidates]
            scores = self.reranker.predict(pairs, show_progress_bar=False)
            rerank_score_of = {i: float(s) for i, s in zip(candidates, scores)}
            ranked = sorted(candidates, key=rerank_score_of.get, reverse=True)

        # Some lexical-only hits are not in the dense pool; compute their cosine.
        missing = [i for i in ranked[: k * 4] if i not in dense_score_of]
        if missing:
            vectors = np.vstack([self.index.reconstruct(int(i)) for i in missing])
            for i, s in zip(missing, vectors @ query_embedding[0]):
                dense_score_of[i] = float(s)

        results = []
        per_article = {}
        for vid in ranked:
            record = self.metadata[vid]
            title = record.get("article_title")
            if max_per_article and per_article.get(title, 0) >= max_per_article:
                continue
            per_article[title] = per_article.get(title, 0) + 1
            results.append(self._format(record, len(results) + 1, mode, dense_score_of.get(vid),
                                        bm25_score_of.get(vid), fused_score_of.get(vid),
                                        rerank_score_of.get(vid)))
            if len(results) >= k:
                break
        return results

    def _format(self, metadata, rank, mode, dense, bm25, fused, rerank):
        return {
            "dataset_version": self.version.upper(),
            "retrieval_mode": mode,
            "rank": rank,
            "similarity_score": round(dense, 4) if dense is not None else None,
            "bm25_score": round(bm25, 4) if bm25 is not None else None,
            "fused_score": round(fused, 6) if fused is not None else None,
            "rerank_score": round(rerank, 4) if rerank is not None else None,
            "chunk_id": metadata.get("chunk_id"),
            "vector_id": metadata.get("vector_id"),
            "article_title": metadata.get("article_title"),
            "section": metadata.get("section"),
            "subject_id": metadata.get("subject_id"),
            "subject_name": metadata.get("subject_name"),
            "topic": metadata.get("topic"),
            "discipline": metadata.get("discipline"),
            "disciplines": metadata.get("disciplines"),
            "review_status": metadata.get("review_status"),
            "text": metadata.get("text"),
            "source": metadata.get("source"),
            "source_url": metadata.get("source_url"),
            "page_id": metadata.get("page_id"),
            "revision_id": metadata.get("revision_id"),
            "revision_timestamp": metadata.get("revision_timestamp"),
        }


# ============================================================
# DISPLAY RETRIEVAL RESULTS
# ============================================================

def print_results(question, results):
    print("\n" + "=" * 75)
    print("QUESTION:", question)
    print("=" * 75)
    if not results:
        print("No relevant chunks retrieved.")
        return
    for result in results:
        print(f"\nRank: {result['rank']}")
        print("Cosine similarity:", result["similarity_score"])
        if result.get("bm25_score") is not None:
            print("BM25 score:", result["bm25_score"])
        if result.get("rerank_score") is not None:
            print("Rerank score:", result["rerank_score"])
        print("Article:", result["article_title"])
        if result.get("section"):
            print("Section:", result["section"])
        print("Subject:", result["subject_name"], f"({result['subject_id']})")
        print("Topic:", result["topic"])
        print("Chunk ID:", result["chunk_id"])
        print("Source URL:", result["source_url"])
        print("\nRetrieved text:")
        print(result["text"][:700])
        print("\n" + "-" * 75)


# ============================================================
# RETRIEVAL SMOKE TEST (report §19)
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EduRoute AI retrieval smoke test")
    parser.add_argument("--version", default="v4", choices=SUPPORTED_VERSIONS)
    parser.add_argument("--mode", default="dense", choices=("dense", "hybrid"))
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--question", help="Run a single question instead of the smoke test")
    args = parser.parse_args()

    retriever = EduRouteRetriever(top_k=args.top_k, version=args.version, mode=args.mode,
                                  rerank=args.rerank, verbose=True)

    questions = [args.question] if args.question else [
        "Explain backpropagation in artificial neural networks.",
        "What is supervised machine learning?",
        "Explain the transformer architecture in NLP.",
        "What is Q-learning in reinforcement learning?",
        "How does SHAP explain machine learning predictions?",
        "Explain Bayes theorem and its application in artificial intelligence.",
        "What is a convolutional neural network and how does it work?",
        "Explain generative adversarial networks.",
        "What is MLOps and why is it important?",
        "What is the purpose of explainable artificial intelligence?",
    ]
    for question in questions:
        print_results(question, retriever.retrieve(question))
