# Sourced by the deploy/gpu scripts. Defines paths and the runtime environment.
# Overrides (export before running a script, or put them in deploy/gpu/local.conf):
#   GPU_ID=1            use only that GPU (shared servers)
#   OLLAMA_PORT=11435   if 11434 is taken by someone else's Ollama
#   APP_PORT=8502       if 8501 is taken
#   PYTHON=python3.11   interpreter for the virtual environment

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="$PROJECT/run"
mkdir -p "$RUN_DIR"
[ -f "$PROJECT/deploy/gpu/local.conf" ] && . "$PROJECT/deploy/gpu/local.conf"

VENV="$PROJECT/.venv"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
APP_PORT="${APP_PORT:-8501}"
OLLAMA_HOME="${OLLAMA_HOME:-$HOME/.local/ollama}"     # user-space install location
export PATH="$OLLAMA_HOME/bin:$PATH"

# Ollama: local only, long context (retrieved passages + 10-mark answers),
# text and vision models resident together.
export OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-3}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-2}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"

[ -n "${GPU_ID:-}" ] && export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

activate_venv() {
  [ -f "$VENV/bin/activate" ] || { echo "No virtual environment: run deploy/gpu/setup.sh first" >&2; exit 1; }
  . "$VENV/bin/activate"
  # cuDNN/cuBLAS from the nvidia-* wheels, for faster-whisper (CTranslate2)
  local site
  site="$(python -c 'import site; print(site.getsitepackages()[0])')"
  for lib in "$site"/nvidia/*/lib; do [ -d "$lib" ] && LD_LIBRARY_PATH="$lib:${LD_LIBRARY_PATH:-}"; done
  export LD_LIBRARY_PATH
}

ollama_up() { curl -s -m 3 "http://127.0.0.1:$OLLAMA_PORT/api/version" >/dev/null; }

start_ollama() {
  if ollama_up; then echo "Ollama already running on port $OLLAMA_PORT"; return; fi
  command -v ollama >/dev/null || { echo "Ollama not installed: run setup.sh" >&2; exit 1; }
  nohup ollama serve >"$RUN_DIR/ollama.log" 2>&1 &
  echo $! >"$RUN_DIR/ollama.pid"
  for _ in $(seq 60); do ollama_up && { echo "Ollama started on port $OLLAMA_PORT"; return; }; sleep 1; done
  echo "Ollama did not start; see $RUN_DIR/ollama.log" >&2; exit 1
}

warm() {  # load a model into GPU memory so the first real request is fast
  curl -s -m 900 "http://127.0.0.1:$OLLAMA_PORT/api/generate" \
    -d "{\"model\":\"$1\",\"prompt\":\"Say OK.\",\"stream\":false}" | grep -q '"response"' \
    && echo "  $1 loaded" || echo "  $1 failed to load (see $RUN_DIR/ollama.log)"
}
