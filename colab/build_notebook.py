"""Writes colab/EduRoute_Colab_Benchmark.ipynb. Run: venv\\Scripts\\python colab\\build_notebook.py"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "EduRoute_Colab_Benchmark.ipynb"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md("""
# EduRoute AI: evaluation on a Colab GPU

**Before you start:** *Runtime → Change runtime type → T4 GPU*, then
*Runtime → Run all*. When step 2 asks, upload `colab/eduroute_colab_bundle.zip`.
At the end a zip downloads: copy its files **into** `evaluation\\results\\`
(do not replace the folder).

Step 0 chooses what to run. The defaults run only the image questions, comparing
the 3B and 7B vision models (about 25 minutes in total, including downloads):

| Step | What | Default | Time on a T4 |
|---|---|---|---|
| 6 | Image questions: Qwen2.5-VL 3B and 7B compared | on | ~12 min |
| 7 | System benchmark: 5 strategies × 50 questions | off (done 11 Nov) | ~30 min |
| 9 | Router check on held-out questions | off (done: 0.964) | ~1 min |
| 10 | Voice word error rate | off (synthetic baseline done) | ~3 min |
| 11 | Claim-labelling export (only after a benchmark) | on | seconds |
""")

md("## 0. Choose what to run")
code("""
RUN_IMAGES = True        # image questions (re-measures the 11 Nov image-prompt fix, ~5 min)
# Vision models to compare on the image questions. 7B (~6 GB) fits the T4 next to the
# 3B text model; it is the candidate fix for the 3B model misreading tables.
VISION_MODELS = ['qwen2.5vl:3b', 'qwen2.5vl:7b']
RUN_BENCHMARK = False    # system benchmark (done 11 Nov; set True to repeat, ~30 min)
RUN_ROUTER = False       # router held-out check (already measured)
RUN_ASR = False          # voice WER; set True and upload voice_eval/ to use real recordings
SAMPLE = 20              # questions per set; None = all 130 questions (~65 min)
""")

md("## 1. Check the GPU")
code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

md("""
## 2. Upload the project bundle

If the upload is slow, put the zip in Google Drive and use the commented lines instead.
""")
code("""
import os, zipfile
from google.colab import files

uploaded = files.upload()                      # choose eduroute_colab_bundle.zip
bundle = next(iter(uploaded))
# from google.colab import drive; drive.mount('/content/drive')
# bundle = '/content/drive/MyDrive/eduroute_colab_bundle.zip'

zipfile.ZipFile(bundle).extractall('/content')
os.chdir('/content/eduroute')
os.makedirs('evaluation/results', exist_ok=True)
print(sorted(os.listdir('.')))
""")

md("## 3. Install dependencies")
code("""
!pip -q install sentence-transformers faiss-cpu python-dotenv truststore requests Pillow
if RUN_ASR:
    !pip -q install faster-whisper av edge-tts gTTS
""")

md("## 4. Start Ollama on the GPU and download the models")
code("""
!apt-get -qq install -y zstd > /dev/null
!curl -fsSL https://ollama.com/install.sh | sh > /dev/null

import subprocess, time, requests
env = {**os.environ, 'OLLAMA_CONTEXT_LENGTH': '8192', 'OLLAMA_KEEP_ALIVE': '2h',
       'OLLAMA_MAX_LOADED_MODELS': '2'}   # keep the text and vision models loaded together
server = subprocess.Popen(['ollama', 'serve'], env=env,
                          stdout=open('/content/ollama.log', 'w'), stderr=subprocess.STDOUT)
for _ in range(60):
    try:
        print('Ollama', requests.get('http://localhost:11434/api/version').json()['version']); break
    except Exception:
        time.sleep(1)

def installed():
    return [m['name'] for m in requests.get('http://localhost:11434/api/tags').json()['models']]

!ollama pull qwen2.5:3b 2>&1 | tail -1
if RUN_IMAGES:
    for vision_model in VISION_MODELS:
        for attempt in (1, 2):
            !ollama pull {vision_model} 2>&1 | tail -1
            if vision_model in installed():
                break
    VISION_MODELS = [m for m in VISION_MODELS if m in installed()]
print('Installed models:', installed())
assert any(n.startswith('qwen2.5:3b') for n in installed()), 'Text model qwen2.5:3b is missing'
if RUN_IMAGES and not VISION_MODELS:
    RUN_IMAGES = False
    print('WARNING: no vision model could be downloaded, so the image step is skipped.')
""")

md("## 5. Settings, speed check and one question end to end")
code("""
os.environ.update({
    'EDUROUTE_RAG_VERSION': 'engineering_v1',
    'EDUROUTE_LLM_BASE_URL': 'http://localhost:11434/v1',
    'EDUROUTE_LLM_MODEL': 'qwen2.5:3b',
    'EDUROUTE_LLM_TIMEOUT': '600',
    'EDUROUTE_NLI_MODEL': 'cross-encoder/nli-deberta-v3-xsmall',
    'EDUROUTE_VISION_BASE_URL': 'http://localhost:11434/v1',
    'EDUROUTE_VISION_MODEL': 'qwen2.5vl:3b',
    'EDUROUTE_VISION_TIMEOUT': '600',
})
r = requests.post('http://localhost:11434/api/generate', json={
    'model': 'qwen2.5:3b', 'prompt': 'Define slip in an induction motor in one sentence.', 'stream': False}).json()
print(f"Text model: {r['eval_count'] / r['eval_duration'] * 1e9:.0f} tokens/s (expect tens; ~1 means no GPU)")

from src.pipeline import EduRoutePipeline
pipeline = EduRoutePipeline(log=False)
r = pipeline.ask('Explain the working of a three-phase induction motor.', answer_mode='5-mark')
print('workflow:', r.workflow, '| grounded:', r.grounded, '| %.1f s' % (r.telemetry['total_ms'] / 1000))
print('\\n' + r.answer)
print('\\nNotices:', r.notices)
if r.verification:
    for c in r.verification['claims']:
        print(f"  [{c['label']:<11}] nli={c.get('nli')} {c['text'][:90]}")
""")

md("""
## 6. Image questions (Qwen2.5-VL)

8 starter images with exactly known answers (truth table, block diagram,
signal plot, bar chart, equation, circuit, code, flowchart), plus any real
images added to `evaluation/image_eval/` before the bundle was made.
""")
code("""
if RUN_IMAGES:
    for vision_model in VISION_MODELS:      # one result file per model, to compare
        print(f'\\n===== {vision_model} =====')
        try:  # warm the vision model: its first load can take a minute
            warm = requests.post('http://localhost:11434/api/generate',
                                 json={'model': vision_model, 'prompt': 'Say OK.', 'stream': False},
                                 timeout=600).json()
            print('Vision model:', warm.get('error') or 'loaded')
        except Exception as error:
            print('Vision warm-up failed:', error)
        !EDUROUTE_VISION_MODEL={vision_model} python -m evaluation.run_image_benchmark 2>&1 | grep -v -E "Batches|Loading weights|FutureWarning|torch.jit"
else:
    print('Skipped (RUN_IMAGES = False)')
""")

md("""
## 7. System benchmark

Seeded sample: 20 in-domain, 20 out-of-domain and 10 extra questions × 5
strategies = 250 answers, the same questions as every earlier run. If the
session disconnects, run this cell again: it resumes and skips finished answers.
""")
code("""
PARTIAL = 'evaluation/results/system_benchmark_partial_colab.jsonl'
open(PARTIAL, 'a').close()
if RUN_BENCHMARK:
    sample_arg = f'--sample {SAMPLE}' if SAMPLE else ''
    !python -m evaluation.run_system_benchmark {sample_arg} --seed 7 --resume {PARTIAL} 2>&1 | grep -v -E "Batches|Loading weights|FutureWarning|torch.jit"
else:
    print('Skipped (RUN_BENCHMARK = False)')
""")

md("## 8. Results")
code("""
import glob, json
import pandas as pd

runs = sorted(glob.glob('evaluation/results/system_benchmark_2*.json'))
if runs:
    report = json.load(open(runs[-1]))
    print('LLM:', report['llm'], '| NLI:', report['config'].get('nli_model'), '| questions:', report['questions'])
    table = pd.DataFrame({s: v['all'] for s, v in report['summary'].items()}).T
    cols = ['runs', 'failures', 'median_latency_ms', 'mean_input_tokens', 'mean_output_tokens',
            'grounded_rate', 'verified_rate', 'revised_rate', 'mean_unsupported_claim_rate',
            'answers_with_citations', 'answers_with_contradicted_claims', 'answers_with_unconfirmed_figures']
    display(table[[c for c in cols if c in table]])
images = sorted(glob.glob('evaluation/results/image_benchmark_*.json'))
if images:
    image_report = json.load(open(images[-1]))
    print('Image benchmark:', image_report.get('summary') or image_report.get('reason'))
""")

md("## 9. Router check on held-out questions (optional)")
code("""
if RUN_ROUTER:
    !python -m evaluation.check_router_heldout --version engineering_v1 2>&1 | grep -v -E "Batches|Loading weights|FutureWarning"
else:
    print('Skipped (RUN_ROUTER = False)')
""")

md("""
## 10. Voice word error rate (optional)

With recordings, upload a folder `voice_eval/` (audio files plus
`references.csv` with columns `file,reference,speaker,accent`) and it is used
instead of the synthetic voices.
""")
code("""
if RUN_ASR:
    if os.path.exists('voice_eval/references.csv'):
        !python -m evaluation.evaluate_asr voice_eval 2>&1 | tail -6
    else:
        !python -m evaluation.evaluate_asr --make-synthetic 20 2>&1 | tail -6
else:
    print('Skipped (RUN_ASR = False)')
""")

md("## 11. Export claims for human labelling")
code("""
if os.path.getsize(PARTIAL):
    !python -m evaluation.claim_labels export {PARTIAL} --n 120
""")

md("""
## 12. Download the results

Copy the files from this zip **into** the project's `evaluation\\results\\`
folder (do not replace the folder). Keep `grading_key_*.json` away from the raters.
""")
code("""
!cd evaluation && zip -q -r /content/eduroute_results.zip results
files.download('/content/eduroute_results.zip')
""")

notebook = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "cells": [{**c, "source": [line + "\n" for line in c["source"].split("\n")[:-1]]
               + [c["source"].split("\n")[-1]]} for c in cells],
}
OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print("Wrote", OUT)
