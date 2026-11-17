"""
Translation between English and Indian languages, offline on the CPU.

Model: Meta NLLB-200 distilled 600M, converted to CTranslate2 int8 (~600 MB,
models/translation/nllb-200-600M-ct2-int8; rebuild steps in docs/MULTILINGUAL.md).
Licence CC-BY-NC 4.0: research and non-commercial use. No downloaded model code
runs: the model is plain weights run by CTranslate2, the tokenizer is the standard
NLLB tokenizer in transformers.

EduRoute keeps thinking in English: a question in an Indian language is
translated to English for retrieval, routing and verification, and only the
final, already-checked answer is translated back. Answer translation keeps the
Markdown structure (headings, bullets, numbered lists, tables, code blocks) and
protects what must not change: citations [n], inline code, formulas and URLs.
"""

import re
import threading

from config import settings
from src.language.languages import LANGUAGES

MODEL_DIR = settings.BASE_DIR / "models" / "translation" / "nllb-200-600M-ct2-int8"
_lock = threading.Lock()
_model = {}

# Citations at the end of a sentence, before or after its final punctuation:
# "four parts [1]." and "four parts. [1]" both keep [1] out of the translation.
CITATIONS = re.compile(r"\s*((?:\[\d{1,2}\])+)\s*(?=[.!?।]?\s*$)")
PROTECT = re.compile(r"`[^`]+`|\$[^$]+\$|\\\(.+?\\\)|https?://\S+|\b[A-Za-z]\s*=\s*[^,;.]+(?=[,;.]|$)")
SENTENCE = re.compile(r"(?<=[.!?।])\s+")
LINE = re.compile(r"^(\s*(?:#{1,6}\s+|[-*•]\s+|\d+[.)]\s+|>\s+)?)(.*)$")


# Keep technical terms in English, per language. The quality gate of 17 Nov
# (evaluate_translation.py, both settings) found term survival higher with terms
# kept for all eight languages (e.g. hi 0.88 vs 0.85, te 0.77 vs 0.70, ml 0.84 vs
# 0.72), so the default (True) applies everywhere; set False here to override one.
TERMS_IN_ENGLISH = {}


class TranslationUnavailable(RuntimeError):
    pass


def available():
    return (MODEL_DIR / "model.bin").exists()


def _load():
    with _lock:
        if not _model:
            if not available():
                raise TranslationUnavailable("The translation model is not installed (see docs/MULTILINGUAL.md).")
            import ctranslate2
            from transformers import AutoTokenizer
            _model["translator"] = ctranslate2.Translator(str(MODEL_DIR), device="cpu", compute_type="int8",
                                                          inter_threads=1, intra_threads=4)
            # transformers warns about a tokenizer regex aimed at Mistral models; for NLLB it is a
            # false alarm (English, Hindi and Tamil text round-trip exactly, checked 17 Nov).
            from transformers.utils import logging as hf_logging
            level = hf_logging.get_verbosity()
            hf_logging.set_verbosity_error()
            try:
                _model["tokenizer"] = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
            finally:
                hf_logging.set_verbosity(level)
        return _model["translator"], _model["tokenizer"]


def translate_sentences(sentences, src="en", tgt="hi", beam_size=2, max_length=256):
    """Translate plain sentences (no Markdown). Language codes as in languages.py."""
    sentences = [s for s in sentences]
    if not sentences or src == tgt:
        return list(sentences)
    translator, tokenizer = _load()
    tokenizer.src_lang = LANGUAGES[src].nllb
    batch = [tokenizer.convert_ids_to_tokens(tokenizer.encode(s)) for s in sentences]
    results = translator.translate_batch(batch, target_prefix=[[LANGUAGES[tgt].nllb]] * len(batch),
                                         beam_size=beam_size, max_decoding_length=max_length,
                                         repetition_penalty=1.1)
    out = []
    for r in results:
        tokens = r.hypotheses[0][1:]                              # drop the language tag
        out.append(tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True).strip())
    return out


_glossary = {}


def glossary():
    """Technical terms that stay in English inside a translated answer, as in Indian
    engineering classrooms ("deadlock", "turbine"). Taken from the knowledge base itself:
    its article titles and catalogue topics. Without this, NLLB turned "steam" into
    "fog" (Tamil) and "deadlock" into "passivity" (Kannada) on 17 Nov."""
    with _lock:
        if "pattern" in _glossary:
            return _glossary["pattern"]
    import json
    terms = set()
    meta = settings.VECTOR_STORE_DIR / f"{settings.RAG_VERSION}_faiss_v1_metadata.jsonl"
    meta = meta if meta.exists() else settings.VECTOR_STORE_DIR / "engineering_faiss_v1_metadata.jsonl"
    if meta.exists():
        with meta.open(encoding="utf-8") as f:
            for line in f:
                title = json.loads(line).get("article_title") or ""
                terms.add(re.sub(r"\s*\(.*?\)\s*", "", title).strip())
    try:
        from src.utils.catalogue_loader import load_catalogue
        for s in load_catalogue()["subjects"]:
            terms.update(s["topics"])
    except Exception:
        pass
    words = sorted({t.lower() for t in terms
                    if t and len(t) >= 5 and len(t.split()) <= 4 and re.fullmatch(r"[A-Za-z0-9 \-']+", t)},
                   key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE) if words else None
    with _lock:
        _glossary["pattern"] = pattern
    return pattern


def _protect(sentence, terms=False):
    """Swap spans that must not be translated for numbered markers NLLB copies through:
    citations' neighbours, code, formulas, URLs and (when `terms`) technical terms."""
    kept = []

    def keep(m):
        kept.append(m.group(0))
        return f"X{len(kept) - 1}X"
    sentence = PROTECT.sub(keep, sentence)
    if terms and glossary() is not None:
        sentence = glossary().sub(keep, sentence)
    return sentence, kept


# NLLB sometimes transliterates the marker "X0X" into the target script
# ("एक्स0एक्स" in Hindi, 17 Nov); accept the letter X in every supported script.
_X = r"(?:X|x|एक्स|ಎಕ್ಸ್|ఎక్స్|எக்ஸ்|എക്സ്|এক্স|એક્સ)"


def _restore(text, kept):
    for i, span in enumerate(kept):
        text, n = re.subn(rf"{_X}\s*-?\s*{i}\s*-?\s*{_X}", lambda _m, s=span: s, text)
        if n == 0:
            text += f" {span}"                                   # never lose a formula or code
    return text


def translate_answer(markdown, tgt, src="en", keep_terms=None):
    """Translate a Markdown answer, keeping its structure, citations and formulas.
    `keep_terms`: keep technical terms in English (default: per language, TERMS_IN_ENGLISH)."""
    keep_terms = TERMS_IN_ENGLISH.get(tgt, True) if keep_terms is None else keep_terms
    if tgt == src or not markdown:
        return markdown
    lines = markdown.split("\n")
    jobs, plan, in_code = [], [], False
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            plan.append(("keep", line))
            continue
        if in_code or not line.strip() or re.fullmatch(r"\s*\|?[\s:\-|]+\|?\s*", line):
            plan.append(("keep", line))
            continue
        if line.strip().startswith("|"):                       # table row: translate each cell
            cells = line.strip().strip("|").split("|")
            refs = []
            for cell in cells:
                refs.append(len(jobs)); jobs.append(_protect(cell.strip(), terms=keep_terms))
            plan.append(("table", refs))
            continue
        prefix, body = LINE.match(line).groups()
        body = body.replace("**", "").replace("__", "")
        refs = []
        for sentence in SENTENCE.split(body):
            if not sentence.strip():
                continue
            cite = CITATIONS.search(sentence)
            citation = cite.group(1) if cite else ""
            plain = (sentence[: cite.start()] + sentence[cite.end():]) if cite else sentence
            text, kept = _protect(plain.strip(), terms=keep_terms)
            refs.append((len(jobs), citation)); jobs.append((text, kept))
        plan.append(("line", (prefix, refs)))
    translated = translate_sentences([t for t, _ in jobs], src, tgt) if jobs else []
    restored = [_restore(t, kept) for t, (_, kept) in zip(translated, jobs)]
    out = []
    for kind, item in plan:
        if kind == "keep":
            out.append(item)
        elif kind == "table":
            out.append("| " + " | ".join(restored[i] for i in item) + " |")
        else:
            prefix, refs = item
            out.append(prefix + " ".join(restored[i] + (f" {c}" if c else "") for i, c in refs))
    return "\n".join(out)


_cache = {}
_CACHE_SIZE = 512


def translate_cached(markdown, tgt, src="en"):  # noqa: D401
    """translate_answer with a small memory cache: the same answer in the same language
    (a repeated question served from the answer cache) is translated once."""
    key = (hash(markdown), src, tgt)
    with _lock:
        if key in _cache:
            return _cache[key]
    result = translate_answer(markdown, tgt, src)
    with _lock:
        _cache[key] = result
        while len(_cache) > _CACHE_SIZE:
            _cache.pop(next(iter(_cache)))
    return result


def to_english(text, src):
    """A question in an Indian language, in English for retrieval and the model."""
    if src == "en" or not text.strip():
        return text
    parts = [p for p in SENTENCE.split(text.strip()) if p.strip()]
    return " ".join(translate_sentences(parts, src, "en"))
