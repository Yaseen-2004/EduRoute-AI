"""
Vision analysis (report §12.2 steps 3-4, 7).

Uses an OpenAI-compatible vision endpoint (e.g. Qwen2.5-VL served by Ollama
`qwen2.5vl:3b` or vLLM). Two uses:
  describe()  structured extraction of visible content, marked
              VISIBLE / UNCERTAIN, for the hybrid multimodal RAG workflow
  answer()    answer the question from the image alone (vision_llm workflow)
"""

from src.llm.backends import LLMUnavailableError
from src.llm.prompts import (
    ANSWER_MODES, NO_PADDING, SYSTEM_PROMPT, VISION_ANSWER_PROMPT, VISION_EXTRACTION_PROMPT,
)


class VisionAnalyzer:

    def __init__(self, backend):
        if backend is None:
            raise LLMUnavailableError("No vision model configured (set EDUROUTE_VISION_*).")
        self.backend = backend

    def describe(self, prepared_image, question):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": VISION_EXTRACTION_PROMPT.format(question=question or "(none)")},
        ]
        return self.backend.generate(
            messages, max_new_tokens=400, temperature=0.1,
            images=[(prepared_image.data, prepared_image.mime)],
        )

    def answer(self, prepared_image, question, answer_mode="5-mark", on_token=None):
        mode = ANSWER_MODES[answer_mode]
        prompt = VISION_ANSWER_PROMPT.format(
            question=question or "Explain this image.",
            instructions=mode["instructions"], no_padding=NO_PADDING,
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}]
        return self.backend.generate(
            messages, max_new_tokens=mode["max_new_tokens"], temperature=0.2,
            images=[(prepared_image.data, prepared_image.mime)], on_token=on_token,
        )
