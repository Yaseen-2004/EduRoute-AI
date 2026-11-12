"""
LLM backends for EduRoute AI.

All backends share one interface: generate(messages, ...) -> GenerationResult,
where `messages` is an OpenAI-style chat list. The proposed model is
Qwen2.5-3B-Instruct (report §9); where it runs is a deployment choice:

  openai_compatible  Any server exposing POST /v1/chat/completions:
                     - Ollama on this laptop:  ollama pull qwen2.5:3b
                       base_url http://localhost:11434/v1, model qwen2.5:3b
                     - vLLM on Colab T4:  vllm serve Qwen/Qwen2.5-3B-Instruct --dtype half
                     Also carries images for a vision model (Qwen2.5-VL).
  transformers       Loads the HF model in-process. 4-bit (bitsandbytes) when
                     CUDA is available; float32 on CPU works but is slow.
  extractive         No model. Returns the most relevant evidence sentences.
                     For offline demos/tests only and always labelled as such.
"""

import base64
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """Raised when a configured model or service cannot be reached."""


@dataclass
class GenerationResult:
    text: str
    model: str
    backend: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    is_model_output: bool = True  # False for the extractive fallback
    extra: dict = field(default_factory=dict)


class LLMBackend:
    name = "base"
    supports_images = False

    def generate(self, messages, max_new_tokens=512, temperature=0.3, images=None, evidence=None,
                 on_token=None):
        """`evidence` is only used by the extractive fallback; model backends ignore it."""
        raise NotImplementedError

    def health_check(self):
        """Return (ok: bool, detail: str) without raising."""
        return True, "ok"

    def is_available(self):
        """Cheap check used for routing fallbacks; may be cached."""
        return True


# ---------------------------------------------------------------------------


class OpenAICompatibleBackend(LLMBackend):
    name = "openai_compatible"

    def __init__(self, base_url, model, api_key=None, timeout=180, supports_images=False):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.supports_images = supports_images
        self._session = None
        # After a connection failure, skip calls for a short period instead of
        # paying the connect timeout on every request.
        self._down_until = 0.0
        self._down_reason = ""
        self.retry_after_s = 30

    @property
    def session(self):
        if self._session is None:
            try:
                import truststore
                truststore.inject_into_ssl()
            except ImportError:
                pass
            import requests
            self._session = requests.Session()
            if self.api_key:
                self._session.headers["Authorization"] = f"Bearer {self.api_key}"
        return self._session

    @staticmethod
    def _attach_images(messages, images):
        """Attach images (bytes) to the last user message, OpenAI vision format."""
        if not images:
            return messages
        messages = [dict(m) for m in messages]
        last = messages[-1]
        content = [{"type": "text", "text": last["content"]}]
        for image_bytes, mime in images:
            encoded = base64.b64encode(image_bytes).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{encoded}"},
            })
        last["content"] = content
        return messages

    def generate(self, messages, max_new_tokens=512, temperature=0.3, images=None, evidence=None,
                 on_token=None):
        """`on_token(text_so_far)` is called while the answer streams in, so the
        interface can show it immediately instead of after the whole answer."""
        if images and not self.supports_images:
            raise LLMUnavailableError(f"Model '{self.model}' is not configured for images.")
        payload = {
            "model": self.model,
            "messages": self._attach_images(messages, images),
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if on_token is not None:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        if time.monotonic() < self._down_until:
            raise LLMUnavailableError(self._down_reason)
        started = time.perf_counter()
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions", json=payload,
                timeout=(3.05, self.timeout),  # (connect, read)
                stream=on_token is not None,
            )
        except Exception as error:  # connection refused, DNS, TLS, timeout
            self._down_reason = f"Cannot reach {self.base_url} ({type(error).__name__})"
            self._down_until = time.monotonic() + self.retry_after_s
            raise LLMUnavailableError(self._down_reason) from error
        if response.status_code >= 400:
            raise LLMUnavailableError(
                f"{self.base_url} returned HTTP {response.status_code}: {response.text[:300]}"
            )
        if on_token is not None:
            return self._read_stream(response, on_token, started)
        data = response.json()
        usage = data.get("usage") or {}
        return GenerationResult(
            text=data["choices"][0]["message"]["content"].strip(),
            model=data.get("model", self.model),
            backend=self.name,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            latency_s=time.perf_counter() - started,
        )

    def _read_stream(self, response, on_token, started):
        """Server-sent events from an OpenAI-compatible /chat/completions stream."""
        import json as _json
        parts, usage, model, chunks = [], {}, self.model, 0
        first_token_s = None
        try:
            for raw in response.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                event = _json.loads(data)
                model = event.get("model", model)
                usage = event.get("usage") or usage
                for choice in event.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        if first_token_s is None:
                            first_token_s = time.perf_counter() - started
                        parts.append(delta)
                        chunks += 1
                        on_token("".join(parts))
        except Exception as error:  # dropped connection mid-answer
            if not parts:
                raise LLMUnavailableError(f"Stream from {self.base_url} failed: {error}") from error
        result = GenerationResult(
            text="".join(parts).strip(),
            model=model,
            backend=self.name,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            # Servers without usage in streams: one chunk is about one token.
            output_tokens=int(usage.get("completion_tokens", chunks)),
            latency_s=time.perf_counter() - started,
        )
        result.extra["first_token_s"] = first_token_s
        return result

    def is_available(self):
        if time.monotonic() < self._down_until:
            return False
        ok, detail = self.health_check()
        if not ok:
            self._down_reason = detail
            self._down_until = time.monotonic() + self.retry_after_s
        return ok

    def health_check(self):
        try:
            response = self.session.get(f"{self.base_url}/models", timeout=(3.05, 5))
            if response.status_code >= 400:
                return False, f"HTTP {response.status_code} from {self.base_url}/models"
            served = [m.get("id") for m in response.json().get("data", [])]
            if served and self.model not in served:
                return False, f"Model '{self.model}' not served. Available: {served}"
            return True, f"{self.model} @ {self.base_url}"
        except Exception as error:
            return False, f"Cannot reach {self.base_url}: {error}"


# ---------------------------------------------------------------------------


class TransformersBackend(LLMBackend):
    name = "transformers"

    def __init__(self, model_name, cache_dir=None, load_in_4bit=True):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise LLMUnavailableError(f"transformers/torch not installed: {error}") from error

        kwargs = {"cache_dir": self.cache_dir}
        if torch.cuda.is_available():
            if self.load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                    )
                except ImportError:
                    logger.warning("bitsandbytes not installed; loading in float16.")
            kwargs.setdefault("dtype", torch.float16)
            kwargs["device_map"] = "auto"
        else:
            logger.warning("No CUDA GPU: loading %s on CPU in float32 (slow).", self.model_name)
            kwargs["dtype"] = torch.float32

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        except Exception as error:
            raise LLMUnavailableError(f"Could not load {self.model_name}: {error}") from error
        self._model.eval()

    def generate(self, messages, max_new_tokens=512, temperature=0.3, images=None, evidence=None,
                 on_token=None):
        if images:
            raise LLMUnavailableError("The text LLM backend cannot process images.")
        import torch

        self._load()
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.9 if temperature > 0 else None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return GenerationResult(
            text=self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
            model=self.model_name,
            backend=self.name,
            input_tokens=int(inputs["input_ids"].shape[1]),
            output_tokens=int(len(new_tokens)),
            latency_s=time.perf_counter() - started,
        )

    def health_check(self):
        try:
            import importlib
            torch = importlib.import_module("torch")
            importlib.import_module("transformers")
        except ImportError as error:
            return False, str(error)
        device = "CUDA" if torch.cuda.is_available() else "CPU (slow)"
        return True, f"{self.model_name} in-process on {device} (loads on first use)"


# ---------------------------------------------------------------------------


class ExtractiveBackend(LLMBackend):
    """
    Not a language model. Selects evidence sentences from the supplied
    context. Exists so the pipeline, router and UI can be exercised offline;
    results are flagged is_model_output=False and the UI says so.
    """

    name = "extractive"

    def is_available(self):
        return False  # not a language model

    def __init__(self, sentence_scorer=None):
        self.sentence_scorer = sentence_scorer

    def generate(self, messages, max_new_tokens=512, temperature=0.3, images=None, evidence=None,
                 on_token=None):
        started = time.perf_counter()
        # `evidence` is {"question": str, "passages": [retrieved chunk dicts]}.
        evidence = evidence or {}
        context = evidence.get("passages") or []
        question = evidence.get("question", "")
        if not context:
            text = (
                "No language model is configured, and this workflow has no retrieved "
                "evidence to summarise. Configure an LLM backend to answer directly."
            )
        else:
            text = self._summarise(question, context, max_sentences=max(2, max_new_tokens // 90))
        return GenerationResult(
            text=text, model="extractive-evidence", backend=self.name,
            latency_s=time.perf_counter() - started, is_model_output=False,
        )

    def _summarise(self, question, context, max_sentences):
        from src.verification.claims import split_sentences

        candidates = []
        for source_number, passage in enumerate(context, start=1):
            for sentence in split_sentences(passage["text"]):
                if len(sentence.split()) >= 6:
                    candidates.append((source_number, sentence))
        if not candidates:
            return "The retrieved passages did not contain usable sentences."

        if self.sentence_scorer is not None:
            scores = self.sentence_scorer(question, [s for _, s in candidates])
            order = sorted(range(len(candidates)), key=lambda i: -scores[i])
        else:
            order = range(len(candidates))

        chosen = sorted(list(order)[:max_sentences])
        lines = [f"{candidates[i][1]} [{candidates[i][0]}]" for i in chosen]
        return "\n".join(lines)


# ---------------------------------------------------------------------------


def build_backend(settings, role="text"):
    """Construct the backend for a role ('text' or 'vision') from settings."""
    if role == "vision":
        if not settings.VISION_BASE_URL or not settings.VISION_MODEL:
            return None
        return OpenAICompatibleBackend(
            settings.VISION_BASE_URL, settings.VISION_MODEL,
            api_key=settings.VISION_API_KEY, supports_images=True,
            timeout=getattr(settings, "VISION_TIMEOUT", 180),
        )

    kind = settings.LLM_BACKEND
    if kind == "openai_compatible":
        return OpenAICompatibleBackend(
            settings.LLM_BASE_URL, settings.LLM_SERVED_MODEL, api_key=settings.LLM_API_KEY,
            timeout=getattr(settings, "LLM_TIMEOUT", 180),
        )
    if kind == "transformers":
        return TransformersBackend(settings.LLM_MODEL_NAME, cache_dir=str(settings.LLM_DIR))
    if kind == "extractive":
        return ExtractiveBackend()
    raise ValueError(f"Unknown LLM backend '{kind}'")
