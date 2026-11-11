
import os
from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Secrets and deployment choices come from the environment or a local .env
# file (never committed; see .env.example). Report §21.
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass


def _env(name, default):
    return os.environ.get(name, default)

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
WIKIPEDIA_RAW_DIR = RAW_DATA_DIR / "wikipedia"
WIKIPEDIA_PROCESSED_DIR = PROCESSED_DATA_DIR / "wikipedia"
METADATA_DIR = DATA_DIR / "metadata"

# Model directories
MODELS_DIR = BASE_DIR / "models"
LLM_DIR = MODELS_DIR / "llm"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"
VISION_DIR = MODELS_DIR / "vision"

# Vector database (versioned AIML indexes: vector_store/aiml_faiss_<version>.index)
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

# Knowledge collection settings
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 0.5

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# RAG configuration (matches the built indexes; see data/metadata reports)
CHUNK_SIZE = 220
CHUNK_OVERLAP = 30
TOP_K = 5
# Default: the engineering-wide index (all 26 catalogue groups) once built,
# otherwise the AIML-only V4 index.
RAG_VERSION = _env(
    "EDUROUTE_RAG_VERSION",
    "engineering_v1" if (BASE_DIR / "vector_store" / "engineering_faiss_v1.index").exists() else "v4",
)
RETRIEVAL_MODE = _env("EDUROUTE_RETRIEVAL_MODE", "dense")  # dense | hybrid (hybrid did not help on V4; see docs)

# LLM configuration (report §9.3)
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = int(_env("EDUROUTE_MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(_env("EDUROUTE_TEMPERATURE", "0.3"))

# Where the LLM runs: openai_compatible | transformers | extractive
LLM_BACKEND = _env("EDUROUTE_LLM_BACKEND", "openai_compatible")
LLM_BASE_URL = _env("EDUROUTE_LLM_BASE_URL", "http://localhost:11434/v1")  # Ollama default
LLM_SERVED_MODEL = _env("EDUROUTE_LLM_MODEL", "qwen2.5:3b")
LLM_API_KEY = _env("EDUROUTE_LLM_API_KEY", "")
# Seconds to wait for one generation. 180 suits a GPU; a laptop CPU at ~1-2
# tokens/s needs ~1800 for 10-mark answers.
LLM_TIMEOUT = int(_env("EDUROUTE_LLM_TIMEOUT", "180"))

# Vision model (OpenAI-compatible endpoint serving e.g. Qwen2.5-VL). Empty = disabled.
VISION_BASE_URL = _env("EDUROUTE_VISION_BASE_URL", "")
VISION_MODEL = _env("EDUROUTE_VISION_MODEL", "")
VISION_API_KEY = _env("EDUROUTE_VISION_API_KEY", "")
# Image analysis is slower than text; by default it gets the same budget as the LLM.
VISION_TIMEOUT = int(_env("EDUROUTE_VISION_TIMEOUT", str(LLM_TIMEOUT)))
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_SIDE = 1536

# Verification: optional NLI model that detects claims contradicted by the
# sources (e.g. cross-encoder/nli-deberta-v3-xsmall). Empty = heuristics only.
NLI_MODEL = _env("EDUROUTE_NLI_MODEL", "")

# Voice
WHISPER_MODEL_SIZE = _env("EDUROUTE_WHISPER_MODEL", "base.en")
TTS_VOICE = _env("EDUROUTE_TTS_VOICE", "en-IN-NeerjaNeural")
MAX_AUDIO_BYTES = 25 * 1024 * 1024

# Logs and telemetry
OUTPUTS_DIR = BASE_DIR / "outputs"
LOG_DIR = OUTPUTS_DIR / "logs"
def router_calibration_path(version):
    """Thresholds are only valid for the index they were fitted on."""
    return BASE_DIR / "config" / f"router_calibration_{version}.json"
# Student questions are personal data (report §21): logs keep a hash only
# unless this is enabled, e.g. for a consented evaluation run.
LOG_QUERY_TEXT = _env("EDUROUTE_LOG_QUERY_TEXT", "false").lower() == "true"
# Logs older than this are deleted (src/utils/retention.py). 0 = keep forever.
LOG_RETENTION_DAYS = int(_env("EDUROUTE_LOG_RETENTION_DAYS", "30"))

# Application settings
APP_NAME = "EduRoute AI"
APP_VERSION = "1.1.0"

# Create required directories
REQUIRED_DIRECTORIES = [
    WIKIPEDIA_RAW_DIR,
    WIKIPEDIA_PROCESSED_DIR,
    METADATA_DIR,
    LLM_DIR,
    EMBEDDINGS_DIR,
    VISION_DIR,
    VECTOR_STORE_DIR,
    LOG_DIR,
]

for directory in REQUIRED_DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)
