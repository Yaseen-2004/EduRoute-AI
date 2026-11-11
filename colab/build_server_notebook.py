"""Writes colab/EduRoute_Colab_Model_Server.ipynb. Run: venv\\Scripts\\python colab\\build_server_notebook.py"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "EduRoute_Colab_Model_Server.ipynb"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md("""
# EduRoute AI: model server for the hosted app

Streamlit Community Cloud has no GPU, so during a demo the language and vision
models run here and the hosted app calls them over a temporary HTTPS tunnel.

1. *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*.
2. Copy the secrets printed by step 4 into the Streamlit app
   (*Manage app → Settings → Secrets*) and save; the app restarts.
3. Keep this tab open during the demo (step 5 keeps the session alive).
   The URL changes every session, so repeat step 2 next time.

Anyone who has the tunnel URL can use these models while the session runs, so
do not share it and stop the session after the demo.
""")

md("## 1. GPU")
code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

md("## 2. Start Ollama and download the models")
code("""
!apt-get -qq install -y zstd > /dev/null
!curl -fsSL https://ollama.com/install.sh | sh > /dev/null
import os, re, subprocess, time, requests
env = {**os.environ, 'OLLAMA_CONTEXT_LENGTH': '8192', 'OLLAMA_KEEP_ALIVE': '24h',
       'OLLAMA_MAX_LOADED_MODELS': '2'}
server = subprocess.Popen(['ollama', 'serve'], env=env,
                          stdout=open('/content/ollama.log', 'w'), stderr=subprocess.STDOUT)
for _ in range(60):
    try:
        requests.get('http://localhost:11434/api/version'); break
    except Exception:
        time.sleep(1)
!ollama pull qwen2.5:3b 2>&1 | tail -1
!ollama pull qwen2.5vl:3b 2>&1 | tail -1
for model in ('qwen2.5:3b', 'qwen2.5vl:3b'):   # load both now so the first student question is fast
    requests.post('http://localhost:11434/api/generate', json={'model': model, 'prompt': 'OK', 'stream': False})
print([m['name'] for m in requests.get('http://localhost:11434/api/tags').json()['models']])
""")

md("## 3. Open the HTTPS tunnel (Cloudflare quick tunnel, no account needed)")
code("""
!wget -q -O /content/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x /content/cloudflared
tunnel = subprocess.Popen(
    ['/content/cloudflared', 'tunnel', '--url', 'http://localhost:11434',
     '--http-host-header', 'localhost:11434'],   # Ollama only accepts its own host name
    stdout=open('/content/tunnel.log', 'w'), stderr=subprocess.STDOUT)
url = None
for _ in range(60):
    found = re.findall(r'https://[a-z0-9-]+\\.trycloudflare\\.com', open('/content/tunnel.log').read())
    if found:
        url = found[0]; break
    time.sleep(1)
print('Tunnel:', url)
""")

md("## 4. Test the endpoint and print the Streamlit secrets")
code("""
served = None
for attempt in range(30):   # the public name can take a few seconds to resolve
    try:
        served = [m['id'] for m in requests.get(f'{url}/v1/models', timeout=10).json()['data']]
        break
    except Exception:
        time.sleep(2)
assert served, 'The tunnel did not respond; re-run step 3 (see /content/tunnel.log).'
print('Served through the tunnel:', served)
print('''
Paste into Streamlit: Manage app -> Settings -> Secrets
--------------------------------------------------------''')
print(f'''EDUROUTE_LLM_BASE_URL = "{url}/v1"
EDUROUTE_LLM_MODEL = "qwen2.5:3b"
EDUROUTE_VISION_BASE_URL = "{url}/v1"
EDUROUTE_VISION_MODEL = "qwen2.5vl:3b"
EDUROUTE_ENABLE_STUDIO = "false"
EDUROUTE_NLI_MODEL = ""''')
""")

md("""
## 5. Keep the session alive during the demo

Stop this cell (■) after the demo; then *Runtime → Disconnect and delete runtime*.
""")
code("""
try:
    while True:
        ok = server.poll() is None and tunnel.poll() is None
        print(time.strftime('%H:%M'), 'serving' if ok else 'STOPPED: re-run steps 2-4', flush=True)
        time.sleep(300)
except KeyboardInterrupt:
    tunnel.terminate(); server.terminate(); print('Stopped.')
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
