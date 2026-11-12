#!/usr/bin/env bash
# One-time install of EduRoute AI on a Linux NVIDIA server. Safe to re-run.
# Needs no root: Ollama goes to ~/.local/ollama if it is not installed already.
#   bash deploy/gpu/setup.sh                 # everything
#   SKIP_7B=1 bash deploy/gpu/setup.sh       # without the 7B vision model (~6 GB less)
set -euo pipefail
. "$(dirname "$0")/common.sh"
cd "$PROJECT"
step() { printf '\n== %s\n' "$*"; }

step "1/7 GPU"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: no NVIDIA driver" >&2; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
CUDA=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
echo "Driver supports CUDA $CUDA${GPU_ID:+; using GPU $GPU_ID}"

step "2/7 Python environment ($VENV)"
if [ -z "${PYTHON:-}" ]; then
  for py in python3.12 python3.11 python3.10 python3; do command -v $py >/dev/null && { PYTHON=$py; break; }; done
fi
$PYTHON -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ needed"'
# Bare container images (pods) often lack venv/zstd; install them when running as root.
if ! $PYTHON -m venv --help >/dev/null 2>&1 || ! command -v zstd >/dev/null; then
  if [ "$(id -u)" = 0 ] && command -v apt-get >/dev/null; then
    apt-get update -qq && apt-get install -y -qq python3-venv zstd curl ca-certificates >/dev/null
  fi
fi
[ -d "$VENV" ] || $PYTHON -m venv "$VENV"
. "$VENV/bin/activate"
python -m pip install -q --upgrade pip wheel

# PyTorch wheel matching the driver (newest CUDA build the driver supports)
if [ -z "${TORCH_INDEX:-}" ]; then
  case 1 in
    $(awk "BEGIN{print ($CUDA >= 12.8)}")) TORCH_INDEX=cu128 ;;
    $(awk "BEGIN{print ($CUDA >= 12.6)}")) TORCH_INDEX=cu126 ;;
    $(awk "BEGIN{print ($CUDA >= 12.4)}")) TORCH_INDEX=cu124 ;;
    $(awk "BEGIN{print ($CUDA >= 12.1)}")) TORCH_INDEX=cu121 ;;
    *) TORCH_INDEX=cu118 ;;
  esac
fi
echo "PyTorch build: $TORCH_INDEX"
pip install -q torch --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
pip install -q -r requirements-gpu.txt
activate_venv
python - <<'PY'
import torch
assert torch.cuda.is_available(), "PyTorch cannot see the GPU"
print("PyTorch", torch.__version__, "on", torch.cuda.get_device_name(0))
PY

step "3/7 Ollama"
if ! command -v ollama >/dev/null; then
  echo "Installing Ollama into $OLLAMA_HOME (no root needed)"
  mkdir -p "$OLLAMA_HOME"
  ARCH=$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')
  if curl -fsSL "https://ollama.com/download/ollama-linux-$ARCH.tgz" -o "$RUN_DIR/ollama.tgz"; then
    tar -xzf "$RUN_DIR/ollama.tgz" -C "$OLLAMA_HOME"
  else
    curl -fsSL "https://ollama.com/download/ollama-linux-$ARCH.tar.zst" -o "$RUN_DIR/ollama.tar.zst"
    tar --zstd -xf "$RUN_DIR/ollama.tar.zst" -C "$OLLAMA_HOME"
  fi
  rm -f "$RUN_DIR"/ollama.t*
fi
ollama --version 2>&1 | tail -1
# Someone else's Ollama (e.g. a system service) on the port: use our own on the
# next port so our context-length settings apply.
if ollama_up && [ ! -f "$RUN_DIR/ollama.pid" ]; then
  echo "Port $OLLAMA_PORT already has an Ollama not started by EduRoute; using $((OLLAMA_PORT + 1))"
  OLLAMA_PORT=$((OLLAMA_PORT + 1))
  echo "OLLAMA_PORT=$OLLAMA_PORT" >>"$PROJECT/deploy/gpu/local.conf"
  export OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"
fi
start_ollama

MODELS="qwen2.5:3b qwen2.5vl:3b"
[ -z "${SKIP_7B:-}" ] && MODELS="$MODELS qwen2.5vl:7b"
step "4/7 Models ($MODELS)"
for m in $MODELS; do echo "Pulling $m"; ollama pull "$m" 2>&1 | tail -1; done

step "5/7 Settings (.env)"
if [ -f .env ]; then echo ".env exists: left unchanged"
else
  sed "s#127.0.0.1:11434#127.0.0.1:$OLLAMA_PORT#g" deploy/gpu/server.env >.env
  chmod 600 .env
  echo "Wrote .env from deploy/gpu/server.env"
fi
# Random access codes where none is set (shown once; they stay in .env, mode 600)
for key in EDUROUTE_ACCESS_CODE EDUROUTE_STUDIO_CODE; do
  if grep -q "^$key=$" .env; then
    code=$(python -c 'import secrets; print(secrets.token_urlsafe(9))')
    sed -i "s#^$key=\$#$key=$code#" .env
    echo "  $key = $code"
  fi
done
echo "  (change them any time in .env, then: bash deploy/gpu/services.sh restart)"

step "6/7 Embedding, NLI and Whisper models (first download only)"
python - <<'PY'
from config import settings
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device="cuda")
print("Embeddings:", settings.EMBEDDING_MODEL_NAME)
if settings.NLI_MODEL:
    CrossEncoder(settings.NLI_MODEL, device="cuda"); print("NLI:", settings.NLI_MODEL)
from faster_whisper import WhisperModel
WhisperModel(settings.WHISPER_MODEL_SIZE, device=settings.WHISPER_DEVICE, compute_type=settings.WHISPER_COMPUTE_TYPE)
print("Whisper:", settings.WHISPER_MODEL_SIZE, "on", settings.WHISPER_DEVICE)
PY

step "7/7 Self-check"
python -m pytest -q 2>&1 | tail -3
python -m deploy.smoke_test || echo "Smoke test failed: see the [FAIL] lines above"
echo
echo "Setup complete. Next:"
echo "  bash deploy/gpu/services.sh start        # Ollama + the app (127.0.0.1:$APP_PORT)"
echo "  bash deploy/gpu/run_evaluations.sh       # every benchmark, results zipped"
