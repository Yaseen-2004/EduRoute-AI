
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


def _flag(name, default):
    return _env(name, default).lower() == "true"


# Agentic RAG (src/agent/agentic_rag.py): plan sub-queries, judge evidence
# coverage, reformulate, and fetch evidence for unsupported claims before revising.
AGENTIC_RAG = _flag("EDUROUTE_AGENTIC_RAG", "true")
AGENTIC_MAX_STEPS = int(_env("EDUROUTE_AGENTIC_MAX_STEPS", "3"))
AGENTIC_MIN_COVERAGE = float(_env("EDUROUTE_AGENTIC_MIN_COVERAGE", "0.6"))
# Ask the LLM for a better search query when rules could not find coverage (one short call).
AGENTIC_LLM_REWRITE = _flag("EDUROUTE_AGENTIC_LLM_REWRITE", "false")

# Context engineering (src/rag/context.py): token-budgeted, de-duplicated,
# sentence-compressed sources and a cache-friendly prompt layout.
CONTEXT_ENGINEERING = _flag("EDUROUTE_CONTEXT_ENGINEERING", "true")
CONTEXT_BUDGET = {"2-mark": 450, "5-mark": 750, "10-mark": 1100, "conversational": 600}

# Memory (src/agent/memory.py): budgeted conversation memory and follow-up resolution.
CONVERSATION_MEMORY = _flag("EDUROUTE_CONVERSATION_MEMORY", "true")
MEMORY_BUDGET_TOKENS = int(_env("EDUROUTE_MEMORY_BUDGET_TOKENS", "500"))

# Inference optimisation (src/inference/answer_cache.py): repeated questions answered from cache.
ANSWER_CACHE = _flag("EDUROUTE_ANSWER_CACHE", "true")
ANSWER_CACHE_SIZE = int(_env("EDUROUTE_ANSWER_CACHE_SIZE", "500"))
# Saved answers stay valid until the index, model or prompt format changes
# (each record carries that version); a week bounds how long an answer can age.
ANSWER_CACHE_TTL_HOURS = float(_env("EDUROUTE_ANSWER_CACHE_TTL_HOURS", "168"))
ANSWER_CACHE_PATH = BASE_DIR / "outputs" / "cache" / "answer_cache.jsonl"
# Matching needs the SAME content words (the safety rule: 0 of 4,950 different benchmark
# questions share them) AND cosine >= this. Phrasings of one question ("What is X?",
# "Tell me about X", "X") scored as low as 0.69 against "Explain X." (14 Nov), so 0.65.
ANSWER_CACHE_SIMILARITY = float(_env("EDUROUTE_ANSWER_CACHE_SIMILARITY", "0.65"))

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
# Image-only questions: the vision model transcribes the image, the text model
# answers from that ("read, then reason"). One-step answers by the 3B vision
# model misjudged a truth table it had transcribed correctly (12 Nov).
VISION_TWO_STEP = _env("EDUROUTE_VISION_TWO_STEP", "true").lower() == "true"
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_SIDE = 1536

# Verification: optional NLI model that detects claims contradicted by the
# sources (e.g. cross-encoder/nli-deberta-v3-xsmall). Empty = heuristics only.
NLI_MODEL = _env("EDUROUTE_NLI_MODEL", "")

# Voice
WHISPER_MODEL_SIZE = _env("EDUROUTE_WHISPER_MODEL", "base.en")
# Beam 1 was as accurate as beam 5 and faster on the synthetic test set (Nov 2026).
WHISPER_BEAM_SIZE = int(_env("EDUROUTE_WHISPER_BEAM", "1"))
# cpu (int8) on the laptop; on an NVIDIA server set cuda and float16.
WHISPER_DEVICE = _env("EDUROUTE_WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = _env("EDUROUTE_WHISPER_COMPUTE_TYPE", "int8" if WHISPER_DEVICE == "cpu" else "float16")
WHISPER_TERM_CORRECTION =_env("EDUROUTE_WHISPER_TERM_CORRECTION", "true").lower() == "true"
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

# Access and load limits (production; deploy/gpu/server.env). Empty access
# code = open app (laptop / lab use). Codes are compared in constant time.
ACCESS_CODE = _env("EDUROUTE_ACCESS_CODE", "")
# The evaluation studio holds grading data: its own code, else the access code.
STUDIO_CODE = _env("EDUROUTE_STUDIO_CODE", "") or ACCESS_CODE
ENABLE_STUDIO = _env("EDUROUTE_ENABLE_STUDIO", "true").lower() == "true"
ENABLE_DEVELOPER_VIEW = _env("EDUROUTE_ENABLE_DEVELOPER_VIEW", "true").lower() == "true"
MAX_QUESTIONS_PER_HOUR = int(_env("EDUROUTE_MAX_QUESTIONS_PER_HOUR", "0"))     # per session; 0 = no limit
MAX_CONCURRENT_ANSWERS = int(_env("EDUROUTE_MAX_CONCURRENT_ANSWERS", "0"))     # whole server; 0 = no limit
QUEUE_TIMEOUT = int(_env("EDUROUTE_QUEUE_TIMEOUT", "120"))                     # seconds to wait for a slot
MAX_QUESTION_CHARS = int(_env("EDUROUTE_MAX_QUESTION_CHARS", "4000"))

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
