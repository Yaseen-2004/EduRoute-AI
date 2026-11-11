#!/usr/bin/env bash
# Read-only check of an NVIDIA server before installing EduRoute AI.
# Changes nothing. Prints what the server has, then which install path fits.
#   bash deploy/gpu/preflight.sh

ok()   { printf '  [ok]   %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*"; }
bad()  { printf '  [miss] %s\n' "$*"; }

echo "== System"
. /etc/os-release 2>/dev/null && ok "OS: ${PRETTY_NAME:-unknown}" || warn "OS: unknown"
ok "Kernel: $(uname -r), arch $(uname -m)"
ok "CPU cores: $(nproc), RAM: $(free -g 2>/dev/null | awk '/Mem:/{print $2" GB total, "$7" GB available"}')"
ok "User: $(id -un)"
if sudo -n true 2>/dev/null; then ok "sudo: yes (no password)"; SUDO=1
elif id -nG | grep -qw -E 'sudo|wheel|admin'; then warn "sudo: member of sudo group, password needed"; SUDO=1
else warn "sudo: no (native user-space install will be used)"; SUDO=0; fi

echo "== GPU"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader \
    | while read -r line; do ok "GPU $line"; done
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  CUDA=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+' | head -1)
  ok "Driver $DRIVER, supports CUDA up to $CUDA"
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | tail -1)
  if [ "${FREE:-0}" -lt 12000 ]; then
    warn "Most free GPU memory: ${FREE} MiB. 3B models need ~6 GB, the 7B vision model ~9 GB more."
  else ok "Most free GPU memory: ${FREE} MiB (enough for 3B text + 7B vision)"; fi
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null \
    | grep -q . && warn "Other jobs are using the GPU (shared server: pick a free GPU with CUDA_VISIBLE_DEVICES)"
else bad "nvidia-smi not found: no NVIDIA driver visible to this user"; fi

echo "== Software"
for py in python3.12 python3.11 python3.10 python3; do
  if command -v $py >/dev/null; then ok "Python: $($py --version 2>&1) ($py)"; break; fi
done
python3 -c 'import venv' 2>/dev/null && ok "python venv module: yes" || warn "python venv module missing (apt install python3-venv, or use conda)"
command -v conda >/dev/null && ok "conda: $(conda --version)"
command -v git >/dev/null && ok "git: yes" || warn "git: no (the project is copied as an archive anyway)"
command -v docker >/dev/null && ok "Docker: $(docker --version 2>/dev/null)" || warn "Docker: not installed"
if command -v docker >/dev/null && docker info 2>/dev/null | grep -qi nvidia; then ok "Docker NVIDIA runtime: yes"
elif command -v docker >/dev/null; then warn "Docker NVIDIA runtime: not detected"; fi
command -v ollama >/dev/null && ok "Ollama: $(ollama --version 2>&1 | tail -1)" || warn "Ollama: not installed (setup.sh installs it)"
command -v dot >/dev/null && ok "Graphviz: yes" || warn "Graphviz 'dot': no (diagrams still render in the browser)"
command -v tmux >/dev/null && ok "tmux: yes" || { command -v screen >/dev/null && ok "screen: yes" || warn "tmux/screen: no (setup uses nohup)"; }

echo "== Disk (needs ~25 GB: models 12 GB, Python packages 8 GB, project 1 GB)"
df -h "$HOME" | awk 'NR==2{print "  [info] "$6": "$4" free of "$2}'

echo "== Network"
for url in https://ollama.com https://huggingface.co https://pypi.org https://download.pytorch.org https://en.wikipedia.org; do
  if curl -s -o /dev/null -m 10 -w '%{http_code}' "$url" | grep -qE '^[23]'; then ok "$url"
  else bad "$url unreachable (proxy? set HTTPS_PROXY before setup.sh)"; fi
done
for port in 8501 11434; do
  (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ":$port " && warn "Port $port already in use" || ok "Port $port free"
done

echo "== Suggested path"
if command -v nvidia-smi >/dev/null; then
  echo "  Native install: bash deploy/gpu/setup.sh   (works with or without sudo)"
  command -v docker >/dev/null && docker info 2>/dev/null | grep -qi nvidia \
    && echo "  Or Docker:      docker compose -f deploy/gpu/docker-compose.yml up -d"
else
  echo "  No GPU visible: fix the driver (or ask the admin) before installing."
fi
