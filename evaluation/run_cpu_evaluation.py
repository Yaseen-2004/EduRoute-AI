"""
The complete evaluation on this laptop's CPU, unattended (no GPU, no Colab).

Runs each stage in order and logs it to outputs/cpu_evaluation.log. It keeps
Windows awake for the whole run and can be stopped and restarted at any time:
finished stages are skipped and the system benchmark resumes answer by answer.

Stages (about 8-10 hours in total on an i7-8665U):
  smoke    every route with the real models                       ~10 min
  system   adaptive strategy on all 140 benchmark questions          ~6-8 h
  images   image benchmark with qwen2.5vl:3b (adaptive + modality)   ~1 h
  images7b the same with qwen2.5vl:7b, only with --with-7b and >= 8 GB free RAM
  voice    word error rate on the synthetic recordings              ~5 min
  claims   120 claims exported for human labelling                  seconds

The 5-strategy comparison (250 answers) is the 11 Nov GPU run; repeating all
5 strategies on 140 questions would take ~30 hours on this CPU.

Run:  run_cpu_evaluation.cmd          (or: venv\\Scripts\\python -m evaluation.run_cpu_evaluation)
      add --stages system images     to run only some stages
      add --redo                     to run finished stages again
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "outputs" / "cpu_evaluation.log"
STATE = ROOT / "outputs" / "cpu_evaluation_state.json"
PARTIAL = "evaluation/results/system_benchmark_partial_cpu_full_adaptive.jsonl"
PY = sys.executable
NOISE = ("Batches", "Loading weights", "FutureWarning", "torch.jit", "it/s]", "warn(")


def free_ram_gb():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    try:
        status = MemoryStatus(dwLength=ctypes.sizeof(MemoryStatus))
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullAvailPhys / 1e9
    except Exception:
        return None


def keep_awake():
    try:  # for the lifetime of this process only; no permanent setting changes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        return True
    except Exception:
        return False


def log(text):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {text}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(name, args, env=None):
    log(f"===== {name}: started")
    start = time.monotonic()
    proc = subprocess.Popen([PY, "-u", *args], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace",
                            env={**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1", **(env or {})})
    with LOG.open("a", encoding="utf-8") as f:
        for line in proc.stdout:
            if not any(n in line for n in NOISE):
                f.write(line)
                f.flush()
    code = proc.wait()
    minutes = (time.monotonic() - start) / 60
    log(f"===== {name}: {'done' if code == 0 else f'FAILED (exit {code})'} in {minutes:.1f} min")
    return code == 0


def ollama_has(model):
    try:
        import requests
        names = [m["name"] for m in requests.get("http://localhost:11434/api/tags", timeout=5).json()["models"]]
        return any(n == model or n.startswith(model + ":") for n in names) or model in names
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+",
                        default=["smoke", "system", "images", "images7b", "voice", "claims"])
    parser.add_argument("--with-7b", action="store_true", help="also compare the 7B vision model")
    parser.add_argument("--redo", action="store_true", help="run stages again even if finished")
    args = parser.parse_args()

    state = json.loads(STATE.read_text()) if STATE.exists() and not args.redo else {}
    log(f"CPU evaluation: stages {' '.join(args.stages)} | keep-awake {'on' if keep_awake() else 'unavailable'}"
        f" | free RAM {free_ram_gb() or 0:.1f} GB")
    (ROOT / PARTIAL).touch()

    stages = {
        "smoke": (["-m", "deploy.smoke_test", "--no-gpu"], None),
        "system": (["-m", "evaluation.run_system_benchmark", "--strategies", "adaptive",
                    "--seed", "7", "--resume", PARTIAL], None),
        "images": (["-m", "evaluation.run_image_benchmark"], {"EDUROUTE_VISION_MODEL": "qwen2.5vl:3b"}),
        "images7b": (["-m", "evaluation.run_image_benchmark"], {"EDUROUTE_VISION_MODEL": "qwen2.5vl:7b"}),
        "voice": (["-m", "evaluation.evaluate_asr", "evaluation/voice_eval_synthetic"], None),
        "claims": (["-m", "evaluation.claim_labels", "export", PARTIAL, "--n", "120"], None),
    }
    for name in args.stages:
        if state.get(name) == "done":
            log(f"----- {name}: already finished (use --redo to repeat)")
            continue
        if name == "images7b":
            ram = free_ram_gb() or 0
            if not args.with_7b:
                log("----- images7b: skipped (add --with-7b; needs ~8 GB free RAM and a 6 GB download)")
                continue
            if ram < 8:
                log(f"----- images7b: skipped, only {ram:.1f} GB RAM free (close other programs, then re-run)")
                continue
            if not ollama_has("qwen2.5vl:7b"):
                log("----- images7b: downloading qwen2.5vl:7b (~6 GB)")
                subprocess.run([os.environ.get("OLLAMA_EXE", r"D:\Ollama\app\ollama.exe"), "pull", "qwen2.5vl:7b"])
        if name == "claims" and not (ROOT / PARTIAL).stat().st_size:
            log("----- claims: skipped, no benchmark answers yet")
            continue
        command, env = stages[name]
        if run(name, command, env):
            state[name] = "done"
            STATE.write_text(json.dumps(state, indent=1))
    log("CPU evaluation finished. Results: evaluation/results/  (tell Claude to analyse them)")


if __name__ == "__main__":
    main()
