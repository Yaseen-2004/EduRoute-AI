# Running EduRoute AI on an NVIDIA GPU server

Everything for the server is in `deploy/`. It needs a Linux server with an NVIDIA
driver (`nvidia-smi` works). Root access, Docker and a pre-installed CUDA toolkit
are **not** required.

| File | What it does |
|---|---|
| `deploy/make_server_package.py` | Packages the committed project into one ~85 MB archive (laptop) |
| `deploy/gpu/preflight.sh` | Read-only check: GPU, free GPU memory, Python, disk, network, ports |
| `deploy/gpu/setup.sh` | One-time install: venv with CUDA PyTorch, Ollama (user space), models, `.env`, self-test |
| `deploy/gpu/services.sh` | `start` / `stop` / `restart` / `status` / `logs` / `watch` for Ollama and the app |
| `deploy/gpu/pod_start.sh` | Start command for a GPU pod: install on first boot, start, auto-restart |
| `deploy/smoke_test.py` | End-to-end check of every route with the real models (pass/fail) |
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
| `smoke` | end-to-end check of every route with the real models |
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

## University DGX cluster (Kubernetes, login node 172.19.0.11)

The login node is reachable only from the campus or lab network. Run these on the laptop, connected to that network:

```powershell
venv\Scripts\python deploy\make_server_package.py
scp deploy\eduroute_server_package.tar.gz <user>@172.19.0.11:~/
ssh <user>@172.19.0.11
```

Then on the login node:

```bash
tar -xzf eduroute_server_package.tar.gz deploy/k8s     # just the cluster scripts
bash deploy/k8s/inspect_cluster.sh                     # read-only: GPUs, quota, storage, used NodePorts
bash deploy/k8s/deploy.sh up                           # storage + GPU pod + NodePort service, copy, start
bash deploy/k8s/deploy.sh logs                         # install progress; access codes appear here once
```

- **App address:** `http://172.19.0.11:30851` once the logs show "App running". It is reachable from the campus network, and students need the access code.
- **Other commands:** `deploy.sh status | eval | results | update | shell | down`. `down` keeps the volume holding the models and results.
- **Settings:** `NAME`, `NODEPORT`, `GPUS`, `STORAGE` and `IMAGE` override the defaults. For example, `NODEPORT=30852 bash deploy/k8s/deploy.sh up` if `inspect_cluster.sh` shows 30851 is taken.
- **No Jupyter needed:** the lab's usual Jupyter step isn't required for EduRoute. If you use Jupyter, set a strong token rather than a short number: a NodePort is open to the whole campus network.

## GPU pod (RunPod, NVIDIA Brev, Kubernetes and similar)

A pod is a container: it has no systemd, and only its **persistent volume**
survives a restart. `deploy/gpu/pod_start.sh` handles both.

1. Create the pod:
   - **GPU:** 16 GB or more (24 GB recommended).
   - **Image:** a recent CUDA or PyTorch base, Ubuntu 22.04/24.04.
   - **Disk:** a persistent volume of **40 GB or more**, mounted at `/workspace`.
   - **Ports:** expose HTTP port **8501**, and SSH if offered.
2. Copy the package to the volume and unpack it to `/workspace/eduroute`, through SSH/scp or the platform's file upload.
3. Run `bash /workspace/eduroute/deploy/gpu/pod_start.sh`, or set it as the pod's start command.
   - On first boot it runs `setup.sh`: Ollama, models, Hugging Face cache and the venv all go on the volume, and it prints the **access codes**.
   - It then starts the services and keeps watching them.
   - Later boots skip the install and take about a minute.
4. Open the platform's link for port 8501.
   - The app listens on 0.0.0.0 here, which is why it refuses to start without an access code.

## Production settings

All of these live in `.env` on the server (template: `deploy/gpu/server.env`). After changing one, run `services.sh restart`.

| Setting | Default on the server | Purpose |
|---|---|---|
| `EDUROUTE_ACCESS_CODE` | random, generated by setup | Students enter it once per browser session |
| `EDUROUTE_STUDIO_CODE` | random, generated by setup | Evaluation Studio (grading data): **team only** |
| `EDUROUTE_ENABLE_DEVELOPER_VIEW` | false | Hides routing internals and strategy switches from students |
| `EDUROUTE_MAX_QUESTIONS_PER_HOUR` | 60 | Per browser session |
| `EDUROUTE_MAX_CONCURRENT_ANSWERS` | 4 | Answers generated at once. Students beyond that queue, for up to `EDUROUTE_QUEUE_TIMEOUT` = 120 s |
| `EDUROUTE_MAX_QUESTION_CHARS` | 4000 | Question length limit |
| `EDUROUTE_LOG_QUERY_TEXT` | false | Logs keep only a hash of each question (report §21) |

**Operation**
- **Smoke test:** after any change, run `python -m deploy.smoke_test`. It checks the GPU, both models, a cited knowledge-base answer, an image question and voice, each against a time limit. A non-zero exit means do not release.
- **Errors:** students see a short reference code. The full traceback is in `run/app.log`; search it for that code.
- **Recovery:** `services.sh watch` restarts Ollama or the app if either stops. `pod_start.sh` runs it automatically.
- **Logs and backups:** logs rotate at 50 MB. `services.sh watch` also writes a daily backup at 03:00 of the team's work saved on the server (grading, claim labels, syllabus reviews, results and logs) to `run/eduroute_backup_*.tar.gz`, keeping the last 14. Run `services.sh backup` for one now, and copy backups off the server: a pod's volume can be deleted with the pod.
- **Updates:** build a new package on the laptop and unpack it over the old folder. `.env`, `.venv` and `run/` are not in the package, so they survive. Then run `services.sh restart` and the smoke test.

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
