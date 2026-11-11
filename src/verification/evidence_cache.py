"""
Precomputed evidence-window embeddings for fast verification.

Evidence windows (1-2 sentence spans of each chunk) depend only on chunk text,
so their embeddings are computed once per index instead of on every answer.
On this project's CPU that turns ~1.5-3 s of verification embedding into a
lookup; only the answer's claims are embedded at query time.

Build (project root, ~15 min on CPU for V4):
    venv\\Scripts\\python -m src.verification.evidence_cache --version v4
"""

import argparse
import time

import numpy as np

from config import settings
from src.utils.io import load_jsonl
from src.verification.claims import evidence_windows


def cache_path(version):
    # Keep the original AIML V4 file name; other indexes use their version key.
    name = "aiml_evidence_windows_v4" if version == "v4" else f"evidence_windows_{version}"
    return settings.EMBEDDINGS_DIR / f"{name}.npz"


class EvidenceCache:

    def __init__(self, version):
        path = cache_path(version)
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path, allow_pickle=False)
        self.vectors = data["vectors"]
        self.offsets = data["offsets"]
        self.chunk_ids = [str(c) for c in data["chunk_ids"]]
        self.position = {c: i for i, c in enumerate(self.chunk_ids)}

    @classmethod
    def load_if_available(cls, version):
        try:
            return cls(version)
        except FileNotFoundError:
            return None

    def vectors_for(self, chunk_id):
        i = self.position.get(chunk_id)
        if i is None:
            return None
        return self.vectors[self.offsets[i]:self.offsets[i + 1]].astype(np.float32)


def build(version):
    from sentence_transformers import SentenceTransformer

    from src.rag.retrieval import METADATA_FILES

    metadata = load_jsonl(METADATA_FILES[version])
    model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device="cpu")

    texts, offsets = [], [0]
    for record in metadata:
        windows = [w for _, w in evidence_windows([record])]
        texts.extend(windows)
        offsets.append(len(texts))
    print(f"{len(metadata)} chunks -> {len(texts)} evidence windows")

    started = time.time()
    vectors = model.encode(texts, batch_size=16, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=True)
    path = cache_path(version)
    np.savez_compressed(
        path,
        vectors=vectors.astype(np.float16),
        offsets=np.asarray(offsets, dtype=np.int64),
        chunk_ids=np.asarray([m["chunk_id"] for m in metadata]),
    )
    print(f"Saved {path} in {time.time() - started:.0f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=settings.RAG_VERSION)
    build(parser.parse_args().version)
