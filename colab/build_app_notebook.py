"""Writes colab/EduRoute_Colab_App.ipynb: the complete app running on a Colab GPU,
reachable through a public HTTPS link and protected by access codes.
Run: venv\\Scripts\\python colab\\build_app_notebook.py"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "EduRoute_Colab_App.ipynb"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md("""
# EduRoute AI: the full app on a Colab GPU

Runs the complete assistant (text, knowledge base with citations, images, voice,
uploaded documents, verification) on a T4 GPU and gives you a **public HTTPS link**
for students. The link is protected by an **access code**; the evaluation studio
and the operations dashboard need a separate **team code**.

1. *Runtime → Change runtime type → T4 GPU*.
2. *Runtime → Run all*. When step 2 asks, upload `colab/eduroute_colab_bundle.zip`.
3. Step 7 prints the link and the two codes (about 10 minutes after starting).
4. Keep this tab open while students use it. Before you close it, run step 9 to keep the
   ratings, logs and any grading done in the studio.

The free Colab tier is meant for interactive use and sessions end after some hours;
for a class running all term, use Colab Pro or a GPU server (`docs/GPU_SERVER.md`).
""")

md("## 0. Settings")
code("""
VISION_MODEL = 'qwen2.5vl:3b'   # 'qwen2.5vl:7b' reads tables better; needs ~6 GB more GPU memory
ACCESS_CODE = ''                # for students; empty = a random code is generated
TEAM_CODE = ''                  # evaluation studio + operations dashboard; empty = random
ENABLE_STUDIO = True            # team tools (grading, claim labels, recordings) on this deployment
QUESTIONS_PER_HOUR = 60         # per student session
RUN_SMOKE_TEST = True           # check every route before opening the link
""")

md("## 1. Check the GPU")
code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

md("""
## 2. Upload the project bundle

Make it on the laptop with `venv\\\\Scripts\\\\python colab\\\\make_bundle.py`. If the upload
is slow, put the zip in Google Drive and use the commented lines instead.
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
print(sorted(os.listdir('.')))
""")

md("## 3. Install the app (Colab already has PyTorch with CUDA)")
code("""
!pip -q install "streamlit>=1.66,<1.67" sentence-transformers faiss-cpu python-dotenv truststore requests Pillow pymupdf faster-whisper av edge-tts gTTS
!apt-get -qq install -y zstd graphviz > /dev/null
import streamlit, torch
print('Streamlit', streamlit.__version__, '| torch', torch.__version__, '| GPU:', torch.cuda.is_available())
""")

md("## 4. Start Ollama on the GPU and load the models")
code("""
!curl -fsSL https://ollama.com/install.sh | sh > /dev/null
import subprocess, time, requests
env = {**os.environ, 'OLLAMA_CONTEXT_LENGTH': '8192', 'OLLAMA_KEEP_ALIVE': '24h',
       'OLLAMA_MAX_LOADED_MODELS': '2', 'OLLAMA_NUM_PARALLEL': '4'}
ollama = subprocess.Popen(['ollama', 'serve'], env=env,
                          stdout=open('/content/ollama.log', 'w'), stderr=subprocess.STDOUT)
for _ in range(60):
    try:
        print('Ollama', requests.get('http://localhost:11434/api/version').json()['version']); break
    except Exception:
        time.sleep(1)
for model in ('qwen2.5:3b', VISION_MODEL):
    !ollama pull {model} 2>&1 | tail -1
    r = requests.post('http://localhost:11434/api/generate',
                      json={'model': model, 'prompt': 'Say OK.', 'stream': False}, timeout=900).json()
    print(model, '->', r.get('error') or 'loaded on the GPU')
""")

md("## 5. Production settings (written to `.env`, never shared)")
code("""
import secrets
ACCESS_CODE = ACCESS_CODE or secrets.token_urlsafe(6)
TEAM_CODE = TEAM_CODE or secrets.token_urlsafe(9)
settings_env = {
    'EDUROUTE_LLM_BACKEND': 'openai_compatible',
    'EDUROUTE_LLM_BASE_URL': 'http://localhost:11434/v1', 'EDUROUTE_LLM_MODEL': 'qwen2.5:3b',
    'EDUROUTE_LLM_TIMEOUT': '300',
    'EDUROUTE_VISION_BASE_URL': 'http://localhost:11434/v1', 'EDUROUTE_VISION_MODEL': VISION_MODEL,
    'EDUROUTE_VISION_TIMEOUT': '300',
    'EDUROUTE_RAG_VERSION': 'engineering_v1',
    'EDUROUTE_NLI_MODEL': 'cross-encoder/nli-deberta-v3-xsmall',
    'EDUROUTE_WHISPER_MODEL': 'base.en', 'EDUROUTE_WHISPER_DEVICE': 'cpu',
    'EDUROUTE_ACCESS_CODE': ACCESS_CODE, 'EDUROUTE_STUDIO_CODE': TEAM_CODE,
    'EDUROUTE_ENABLE_STUDIO': str(ENABLE_STUDIO).lower(),
    'EDUROUTE_ENABLE_DEVELOPER_VIEW': 'false',
    'EDUROUTE_MAX_QUESTIONS_PER_HOUR': str(QUESTIONS_PER_HOUR),
    'EDUROUTE_MAX_CONCURRENT_ANSWERS': '4', 'EDUROUTE_QUEUE_TIMEOUT': '120',
    'EDUROUTE_LOG_QUERY_TEXT': 'false',
}
with open('.env', 'w') as f:
    f.writelines(f'{k}={v}\\n' for k, v in settings_env.items())
os.chmod('.env', 0o600)
os.environ.update(settings_env)
print('Settings written. Codes are shown in step 7.')
""")

md("""
## 6. Smoke test: every route with the real models

GPU, both models, a cited knowledge-base answer, a general answer, an image question and
voice, each within a time limit. Fix any FAIL before sharing the link.
""")
code("""
if RUN_SMOKE_TEST:
    !python -m deploy.smoke_test 2>&1 | grep -v -E "Batches|Loading weights|FutureWarning|torch.jit"
""")

md("## 7. Start the app and open the public link")
code("""
import re
app = subprocess.Popen(
    ['streamlit', 'run', 'app.py', '--server.port', '8501', '--server.headless', 'true',
     '--server.fileWatcherType', 'none', '--client.toolbarMode', 'minimal'],
    stdout=open('/content/app.log', 'w'), stderr=subprocess.STDOUT)
for _ in range(120):
    try:
        if requests.get('http://localhost:8501/_stcore/health', timeout=2).text == 'ok':
            break
    except Exception:
        pass
    time.sleep(1)
print('App running on the GPU.')

!wget -q -O /content/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x /content/cloudflared
tunnel = subprocess.Popen(['/content/cloudflared', 'tunnel', '--url', 'http://localhost:8501'],
                          stdout=open('/content/tunnel.log', 'w'), stderr=subprocess.STDOUT)
url = None
for _ in range(60):
    found = re.findall(r'https://[a-z0-9-]+\\.trycloudflare\\.com', open('/content/tunnel.log').read())
    if found:
        url = found[0]; break
    time.sleep(1)
print(f'''
==============================================================
  EduRoute AI is live:   {url}

  Student access code:   {ACCESS_CODE}
  Team code (studio and operations dashboard):   {TEAM_CODE}
==============================================================
Share the link and the student code with your class. Keep the team code to the team.
The link changes every time this notebook is restarted.''')
""")

md("""
## 8. Keep it running

Shows usage every 5 minutes and restarts the app or the tunnel if either stops.
Stop this cell (■) when the session is over.
""")
code("""
import json
def questions_so_far():
    path = 'outputs/logs/telemetry.jsonl'
    return sum(1 for _ in open(path)) if os.path.exists(path) else 0
try:
    while True:
        if app.poll() is not None:
            print('App stopped: restarting'); app = subprocess.Popen(app.args, stdout=open('/content/app.log', 'a'),
                                                                   stderr=subprocess.STDOUT)
        if ollama.poll() is not None:
            print('Ollama stopped: restarting'); ollama = subprocess.Popen(['ollama', 'serve'], env=env,
                stdout=open('/content/ollama.log', 'a'), stderr=subprocess.STDOUT)
        if tunnel.poll() is not None:
            print('Tunnel stopped: re-run step 7 for a new link')
        print(time.strftime('%H:%M'), f'| {questions_so_far()} questions answered | {url}', flush=True)
        time.sleep(300)
except KeyboardInterrupt:
    print('Monitoring stopped; the app keeps running until the runtime is disconnected.')
""")

md("""
## 9. Before you close: download the session's data

Ratings, anonymous usage logs, and anything the team saved in the studio (grading,
claim labels, syllabus reviews, recordings). Unzip into the project folder on the laptop.
""")
code("""
!zip -q -r /content/eduroute_session_data.zip outputs/logs evaluation/results evaluation/voice_eval config/branches_review 2>/dev/null; true
files.download('/content/eduroute_session_data.zip')
""")

notebook = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                 "kernelspec": {"name": "python3", "display_name": "Python 3"},
                 "language_info": {"name": "python"}},
    "cells": [{**c, "source": [line + "\n" for line in c["source"].split("\n")[:-1]]
               + [c["source"].split("\n")[-1]]} for c in cells],
}
OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print("Wrote", OUT)
