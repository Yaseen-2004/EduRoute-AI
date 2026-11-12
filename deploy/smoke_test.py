"""
End-to-end smoke test of a deployment: every route with the real models.

Checks: GPU, language and vision models reachable, a knowledge-base answer
with citations, an image question, a voice transcription, and the latency of
each. Exits non-zero if anything fails, so it can gate a deployment.

Run:  python -m deploy.smoke_test            (on the server: after setup/start)
      python -m deploy.smoke_test --no-gpu   (laptop)
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402

RESULTS = []


def check(name, fn, limit_s=None):
    start = time.monotonic()
    try:
        detail = fn()
        seconds = time.monotonic() - start
        ok = limit_s is None or seconds <= limit_s
        if not ok:
            detail = f"{detail}; slower than {limit_s} s"
    except Exception as error:      # report every failure, keep checking the rest
        ok, detail, seconds = False, f"{type(error).__name__}: {error}", time.monotonic() - start
    RESULTS.append({"check": name, "ok": ok, "seconds": round(seconds, 2), "detail": str(detail)[:300]})
    print(f"  [{'pass' if ok else 'FAIL'}] {name} ({seconds:.1f} s) - {detail}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gpu", action="store_true", help="do not require CUDA (laptop)")
    args = parser.parse_args()
    print("EduRoute AI smoke test")

    def gpu():
        import torch
        if not torch.cuda.is_available():
            if args.no_gpu:
                return "no GPU (allowed with --no-gpu)"
            raise RuntimeError("PyTorch cannot see a GPU")
        return torch.cuda.get_device_name(0)
    check("GPU", gpu)

    from src.llm.backends import build_backend

    def model(role):
        backend = build_backend(settings, role)
        if backend is None:
            raise RuntimeError("not configured")
        ok, detail = backend.health_check()
        if not ok:
            raise RuntimeError(detail)
        return detail
    check("Language model reachable", lambda: model("text"))
    check("Vision model reachable", lambda: model("vision"))

    from src.pipeline import EduRoutePipeline
    pipeline = EduRoutePipeline(strategy="adaptive", log=False)   # keep test questions out of telemetry
    limit = None if args.no_gpu else 60

    def rag():
        r = pipeline.ask("Explain the Rankine cycle used in steam power plants.").to_dict()
        assert r["answer"].strip(), "empty answer"
        assert r["workflow"] == "rag_llm", f"routed to {r['workflow']}"
        assert r["cited_sources"], "no cited sources"
        assert "Rankine" in r["answer"] or "rankine" in r["answer"].lower(), "off-topic answer"
        return f"{len(r['cited_sources'])} cited sources, verified={r.get('verification') is not None}"
    check("Knowledge-base answer", rag, limit)

    def direct():
        r = pipeline.ask("Hello! What can you help me with?", answer_mode="2-mark").to_dict()
        assert r["answer"].strip(), "empty answer"
        return f"workflow {r['workflow']}"
    check("General answer", direct, limit)

    def image():
        data = (ROOT / "evaluation" / "image_eval" / "truth_table.png").read_bytes()
        r = pipeline.ask("Which logic gate does this truth table represent? Explain.",
                         image_bytes=data, answer_mode="2-mark").to_dict()
        assert r["workflow"] in ("vision_llm", "hybrid_multimodal_rag"), f"image not read (workflow {r['workflow']})"
        assert re.search(r"\bAND\b(?:[- ]gate)?|\band[- ]gate\b", r["answer"]), "did not identify the AND gate"
        return f"workflow {r['workflow']}"
    check("Image question", image, None if args.no_gpu else 120)

    def voice():
        from src.voice.speech import SpeechRecognizer
        audio = (ROOT / "evaluation" / "voice_eval_synthetic" / "E20_gTTS.mp3").read_bytes()
        text = SpeechRecognizer().transcribe(audio, ".mp3").text
        assert "foundation" in text.lower(), f"heard: {text!r}"
        return text
    check("Voice transcription", voice, None if args.no_gpu else 30)

    out = settings.BASE_DIR / "evaluation" / "results" / f"smoke_test_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"llm": settings.LLM_SERVED_MODEL, "vision": settings.VISION_MODEL,
                               "results": RESULTS}, indent=2), encoding="utf-8")
    failed = [r["check"] for r in RESULTS if not r["ok"]]
    print(f"\n{'All checks passed' if not failed else 'FAILED: ' + ', '.join(failed)}  ({out.name})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
