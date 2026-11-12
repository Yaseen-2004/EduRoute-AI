#!/usr/bin/env bash
# Start, stop or check Ollama and the EduRoute app on the server.
#   bash deploy/gpu/services.sh start | stop | restart | status | logs | watch | backup
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

rotate_logs() {  # keep each log under ~50 MB: one previous copy is kept
  for log in "$RUN_DIR"/app.log "$RUN_DIR"/ollama.log "$RUN_DIR"/watch.log; do
    [ -f "$log" ] && [ "$(stat -c %s "$log")" -gt 52428800 ] && mv -f "$log" "$log.1"
  done
  return 0
}

backup() {  # work saved on the server by the team: grading, labels, reviews, results, logs
  local out="$RUN_DIR/eduroute_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
  tar -czf "$out" --ignore-failed-read evaluation/results config/branches_review outputs/logs 2>/dev/null
  echo "Backup: $out ($(du -h "$out" | cut -f1)). Copy it off the server: scp $(id -un)@$(hostname):$out ."
  ls -1t "$RUN_DIR"/eduroute_backup_*.tar.gz | tail -n +15 | xargs -r rm -f   # keep the last 14
}

start() {
  rotate_logs
  activate_venv
  start_ollama
  echo "Loading models onto the GPU"
  warm "$(grep -oP '^EDUROUTE_LLM_MODEL=\K.*' .env)"
  warm "$(grep -oP '^EDUROUTE_VISION_MODEL=\K.*' .env)"
  if app_up; then echo "App already running on port $APP_PORT"; return; fi
  if [ "$APP_ADDRESS" != "127.0.0.1" ] && ! grep -qE '^EDUROUTE_ACCESS_CODE=.+' .env; then
    echo "Refusing to listen on $APP_ADDRESS without EDUROUTE_ACCESS_CODE in .env (anyone could use the GPU)." >&2
    exit 1
  fi
  # Always a fresh process: a running Streamlit keeps old code in memory.
  nohup streamlit run app.py --server.address "$APP_ADDRESS" --server.port "$APP_PORT" \
        --server.headless true --server.enableXsrfProtection true \
        --server.fileWatcherType none --client.toolbarMode minimal >"$RUN_DIR/app.log" 2>&1 &
  echo $! >"$RUN_DIR/app.pid"
  for _ in $(seq 90); do app_up && break; sleep 1; done
  app_up && echo "App running: http://$APP_ADDRESS:$APP_PORT (on the server)" \
         || { echo "App did not start; see $RUN_DIR/app.log" >&2; exit 1; }
  if [ "$APP_ADDRESS" = "127.0.0.1" ]; then
    echo "From your laptop: ssh -N -L $APP_PORT:127.0.0.1:$APP_PORT $(id -un)@$(hostname -f 2>/dev/null || hostname)"
  else
    echo "Open it through the platform's link for port $APP_PORT (access code required)."
  fi
}

status() {
  ollama_up && echo "Ollama: running on $OLLAMA_PORT" || echo "Ollama: stopped"
  ollama_up && curl -s "http://127.0.0.1:$OLLAMA_PORT/api/ps" | python3 -c \
    'import json,sys; [print("  loaded:", m["name"], round(m.get("size_vram",0)/1e9,1), "GB on GPU") for m in json.load(sys.stdin).get("models",[])]' || true
  app_up && echo "App: running on 127.0.0.1:$APP_PORT" || echo "App: stopped"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
}

watch() {  # keep both services up (pods have no systemd); checks every 30 s
  echo "Watching Ollama and the app (Ctrl+C to stop watching; services keep running)"
  while true; do
    # Subshells: a failed restart is logged and retried, it does not end the watch.
    if ! ollama_up; then echo "$(date -Is) Ollama down: restarting"; rm -f "$RUN_DIR/ollama.pid"; (start_ollama) || true; fi
    if ! app_up; then
      sleep 20                     # it may just be starting
      app_up || { echo "$(date -Is) App down: restarting"; stop_pid app; (start) || true; }
    fi
    rotate_logs
    # Daily backup of the team's work (a pod's volume can be deleted with the pod)
    if [ "$(date +%H%M)" = "0300" ] || ! ls "$RUN_DIR"/eduroute_backup_*.tar.gz >/dev/null 2>&1; then
      backup; sleep 60
    fi
    sleep 30
  done
}

case "${1:-status}" in
  start)   start ;;
  watch)   watch ;;
  backup)  backup ;;
  stop)    stop_pid app; stop_pid ollama ;;
  restart) stop_pid app; sleep 2; start ;;
  status)  status ;;
  logs)    tail -n 40 "$RUN_DIR/app.log" "$RUN_DIR/ollama.log" 2>/dev/null ;;
  *) echo "usage: $0 start|stop|restart|status|logs|watch|backup" >&2; exit 2 ;;
esac
