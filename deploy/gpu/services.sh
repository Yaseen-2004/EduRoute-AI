#!/usr/bin/env bash
# Start, stop or check Ollama and the EduRoute app on the server.
#   bash deploy/gpu/services.sh start | stop | restart | status | logs
# The app listens on 127.0.0.1 only. Open it from your laptop through SSH:
#   ssh -N -L 8501:127.0.0.1:8501 user@server     then browse http://localhost:8501
set -euo pipefail
. "$(dirname "$0")/common.sh"
cd "$PROJECT"

app_up() { curl -s -m 3 "http://127.0.0.1:$APP_PORT/_stcore/health" | grep -q ok; }

stop_pid() {  # stop a process we started (and its children)
  local file="$RUN_DIR/$1.pid"
  [ -f "$file" ] || return 0
  local pid; pid=$(cat "$file")
  if kill -0 "$pid" 2>/dev/null; then
    pkill -TERM -P "$pid" 2>/dev/null || true; kill "$pid" 2>/dev/null || true
    echo "Stopped $1 (pid $pid)"
  fi
  rm -f "$file"
}

start() {
  activate_venv
  start_ollama
  echo "Loading models onto the GPU"
  warm "$(grep -oP '^EDUROUTE_LLM_MODEL=\K.*' .env)"
  warm "$(grep -oP '^EDUROUTE_VISION_MODEL=\K.*' .env)"
  if app_up; then echo "App already running on port $APP_PORT"; return; fi
  # Always a fresh process: a running Streamlit keeps old code in memory.
  nohup streamlit run app.py --server.address 127.0.0.1 --server.port "$APP_PORT" \
        --server.headless true >"$RUN_DIR/app.log" 2>&1 &
  echo $! >"$RUN_DIR/app.pid"
  for _ in $(seq 90); do app_up && break; sleep 1; done
  app_up && echo "App running: http://127.0.0.1:$APP_PORT (on the server)" \
         || { echo "App did not start; see $RUN_DIR/app.log" >&2; exit 1; }
  echo "From your laptop: ssh -N -L $APP_PORT:127.0.0.1:$APP_PORT $(id -un)@$(hostname -f 2>/dev/null || hostname)"
}

status() {
  ollama_up && echo "Ollama: running on $OLLAMA_PORT" || echo "Ollama: stopped"
  ollama_up && curl -s "http://127.0.0.1:$OLLAMA_PORT/api/ps" | python3 -c \
    'import json,sys; [print("  loaded:", m["name"], round(m.get("size_vram",0)/1e9,1), "GB on GPU") for m in json.load(sys.stdin).get("models",[])]' || true
  app_up && echo "App: running on 127.0.0.1:$APP_PORT" || echo "App: stopped"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
}

case "${1:-status}" in
  start)   start ;;
  stop)    stop_pid app; stop_pid ollama ;;
  restart) stop_pid app; sleep 2; start ;;
  status)  status ;;
  logs)    tail -n 40 "$RUN_DIR/app.log" "$RUN_DIR/ollama.log" 2>/dev/null ;;
  *) echo "usage: $0 start|stop|restart|status|logs" >&2; exit 2 ;;
esac
