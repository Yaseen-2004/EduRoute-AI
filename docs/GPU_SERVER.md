# Running EduRoute AI on an NVIDIA GPU server

Everything for the server is in `deploy/`. It needs a Linux server with an NVIDIA
driver (`nvidia-smi` works). Root access, Docker and a pre-installed CUDA toolkit
are **not** required.

| File | What it does |
|---|---|
| `deploy/make_server_package.py` | Packages the committed project into one ~85 MB archive (laptop) |
| `deploy/gpu/preflight.sh` | Read-only check: GPU, free GPU memory, Python, disk, network, ports |
| `deploy/gpu/setup.sh` | One-time install: venv with CUDA PyTorch, Ollama (user space), models, `.env`, self-test |
| `deploy/gpu/services.sh` | `start` / `stop` / `restart` / `status` / `logs` for Ollama and the app |
| `deploy/gpu/run_evaluations.sh` | Every benchmark in sequence, then one results zip |
| `deploy/gpu/server.env` | GPU settings (copied to `.env` by setup) |
| `deploy/gpu/Dockerfile`, `docker-compose.yml` | Alternative Docker route (needs the NVIDIA Container Toolkit) |
| `requirements-gpu.txt` | Python packages for the server (CUDA PyTorch chosen by setup) |

## Resources needed

- **GPU memory:** ~3 GB for qwen2.5:3b, ~4 GB for qwen2.5vl:3b, ~7 GB for qwen2.5vl:7b, ~1 GB for embeddings, NLI and Whisper. All of it together fits on a 16 GB GPU. A 24 GB GPU or larger leaves room for parallel requests. On 8 GB, use `SKIP_7B=1`.
- **Disk:** ~25 GB (models 12 GB, Python packages 8 GB).
- **Internet:** the server must reach ollama.com, huggingface.co, pypi.org and download.pytorch.org for setup.

## 1. Access (laptop)

Use an SSH key, so no password is typed into any chat or file:

```powershell
ssh-keygen -t ed25519 -f $HOME\.ssh\eduroute_gpu -C eduroute-gpu
type $HOME\.ssh\eduroute_gpu.pub     # give this public line to the server admin,
                                      # or: append it to ~/.ssh/authorized_keys on the server
ssh -i $HOME\.ssh\eduroute_gpu user@server
```

If the server is behind a jump host: `ssh -J user@gateway user@gpu-node`.

## 2. Copy the project (laptop)

```powershell
venv\Scripts\python deploy\make_server_package.py
scp -i $HOME\.ssh\eduroute_gpu deploy\eduroute_server_package.tar.gz user@server:~/
```

Only committed files are packaged, so `.env`, grading keys and voice recordings
never leave the laptop. Commit first; the script refuses a dirty tree.

## 3. Install (server)

```bash
mkdir -p ~/eduroute && tar -xzf ~/eduroute_server_package.tar.gz -C ~/eduroute && cd ~/eduroute
bash deploy/gpu/preflight.sh          # read-only; review the [warn]/[miss] lines
bash deploy/gpu/setup.sh              # ~15-30 min, mostly downloads
```

On a **shared server**, pick a free GPU and ports first. Settings in
`deploy/gpu/local.conf` apply to every script:

```bash
cat > deploy/gpu/local.conf <<'EOF'
GPU_ID=2
OLLAMA_PORT=11500
APP_PORT=8601
EOF
```

If the cluster uses **Slurm** instead of direct login, run the same scripts inside
a job, e.g. `srun --gres=gpu:1 --time=08:00:00 --pty bash`. Behind a proxy, export
`HTTPS_PROXY` before setup.

## 4. Run the app (server, viewed from the laptop)

```bash
bash deploy/gpu/services.sh start
bash deploy/gpu/services.sh status    # loaded models and GPU memory
```

The app listens on 127.0.0.1 only, so it is not exposed to the network. From the
laptop, open a tunnel and browse http://localhost:8501:

```powershell
ssh -i $HOME\.ssh\eduroute_gpu -N -L 8501:127.0.0.1:8501 user@server
```

After changing code on the server, use `services.sh restart`: a running Streamlit
keeps the old code in memory.

## 5. Run the evaluations (server)

```bash
nohup bash deploy/gpu/run_evaluations.sh > run/eval.log 2>&1 &
tail -f run/eval.log
```

Stages, in order (choose with `STAGES="system images"`):

| Stage | What runs |
|---|---|
| `tests` | the full pytest suite |
| `retrieval` | engineering and AIML retrieval benchmarks on the engineering_v1 index |
| `router` | held-out router check (the calibrated τ = 0.491) |
| `system` | the **full** system benchmark, all questions × 5 strategies, with NLI on. It resumes if interrupted. |
| `images` | image benchmark once per vision model: **qwen2.5vl:3b vs qwen2.5vl:7b** |
| `voice` | Whisper word error rate on the synthetic set (and `evaluation/voice_eval/` if present) |
| `claims` | 120 claims exported for human labelling |

At the end the script prints the path of `run/eduroute_gpu_results_<time>.zip`
(results, per-stage logs, `server_info_*.txt` with GPU, driver and versions). Any
grading keys go in a separate `..._grading_keys_...zip`. Keep that zip away from raters.

```powershell
scp -i $HOME\.ssh\eduroute_gpu user@server:~/eduroute/run/eduroute_gpu_results_*.zip .
```

Unzip it into the project folder: the files land in `evaluation\results\`.

## Docker route (optional)

If the server has Docker with the NVIDIA runtime (`preflight.sh` says so), see the
header of `deploy/gpu/docker-compose.yml`. The native route above is simpler and
needs no root.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `PyTorch cannot see the GPU` | Driver older than the wheel: `TORCH_INDEX=cu118 bash deploy/gpu/setup.sh` |
| Whisper: `libcudnn_ops.so.9` not found | Scripts set `LD_LIBRARY_PATH` via `activate_venv`. For manual runs, `source deploy/gpu/common.sh; activate_venv` |
| Ollama out of memory | Another job holds the GPU: set `GPU_ID`, or `OLLAMA_MAX_LOADED_MODELS=1` in `local.conf` |
| Port in use | `APP_PORT=` / `OLLAMA_PORT=` in `local.conf` |
| Answers time out | Check `services.sh status` shows the model on the GPU (size on GPU > 0) |
