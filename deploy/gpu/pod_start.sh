#!/usr/bin/env bash
# Start command for a GPU pod (RunPod, NVIDIA Brev, Kubernetes, ...).
# Pods lose everything outside their persistent volume on restart, and have no
# systemd. This keeps the venv, Ollama, models and caches on the volume, installs
# on first boot only, starts the services and then watches them.
#
# Put the project on the volume (e.g. /workspace/eduroute) and set the pod's
# start command to:   bash /workspace/eduroute/deploy/gpu/pod_start.sh
# Expose port 8501 (HTTP) in the pod settings. The platform's link is public,
# so the app requires the access code that setup.sh writes to .env.
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"
PERSIST="${PERSIST_DIR:-$(dirname "$PROJECT")}"

export OLLAMA_HOME="${OLLAMA_HOME:-$PERSIST/ollama}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$PERSIST/ollama/models}"
export HF_HOME="${HF_HOME:-$PERSIST/hf_cache}"
export APP_ADDRESS="${APP_ADDRESS:-0.0.0.0}"
mkdir -p "$OLLAMA_MODELS" "$HF_HOME" "$PROJECT/run"
cd "$PROJECT"
# Every later command (kubectl exec ..., run_evaluations.sh) reads these via common.sh.
cat >deploy/gpu/pod.conf <<EOF
OLLAMA_HOME="$OLLAMA_HOME"
OLLAMA_MODELS="$OLLAMA_MODELS"
HF_HOME="$HF_HOME"
APP_ADDRESS="$APP_ADDRESS"
EOF

# Pod restarts keep the venv but not system packages: re-run setup only when
# something is missing (it is idempotent and skips finished steps quickly).
if [ ! -x .venv/bin/python ] || ! { [ -x "$OLLAMA_HOME/bin/ollama" ] || command -v ollama >/dev/null; } || [ ! -f .env ]; then
  echo "First boot (or incomplete install): running setup"
  bash deploy/gpu/setup.sh 2>&1 | tee -a run/setup.log
fi

bash deploy/gpu/services.sh start
exec bash deploy/gpu/services.sh watch >>run/watch.log 2>&1
