#!/usr/bin/env bash
# EduRoute AI on the DGX Kubernetes cluster. Run on the login node (172.19.0.11)
# with eduroute_server_package.tar.gz and this folder in the home directory.
#
#   bash deploy.sh up        create storage + GPU pod + service, copy the project in, start it
#   bash deploy.sh status    pod, service, GPU and app health
#   bash deploy.sh logs      follow the install/start log inside the pod
#   bash deploy.sh shell     a shell inside the pod
#   bash deploy.sh eval      run every evaluation inside the pod (background)
#   bash deploy.sh results   copy the newest results zip to the login node
#   bash deploy.sh update    copy a new package in and restart the app (.env and models kept)
#   bash deploy.sh down      delete pod and service (the volume with models and results is kept)
#
# Settings (export before running): NAME NODEPORT GPUS STORAGE IMAGE PACKAGE
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME="${NAME:-eduroute}"
NODEPORT="${NODEPORT:-30851}"
GPUS="${GPUS:-1}"
STORAGE="${STORAGE:-60Gi}"
IMAGE="${IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime}"
PACKAGE="${PACKAGE:-$HOME/eduroute_server_package.tar.gz}"
MANIFEST="$HOME/$NAME-k8s.yaml"
NODE_IP="${NODE_IP:-172.19.0.11}"

inpod() { kubectl exec "$NAME" -- bash -c "$*"; }

copy_project() {
  [ -f "$PACKAGE" ] || { echo "Package not found: $PACKAGE" >&2; exit 1; }
  echo "Copying $(du -h "$PACKAGE" | cut -f1) package into the pod"
  kubectl cp "$PACKAGE" "$NAME:/workspace/eduroute_server_package.tar.gz"
  # Unpack over the old copy: .env, .venv and run/ are not in the package, so they stay.
  inpod 'mkdir -p /workspace/eduroute && tar -xzf /workspace/eduroute_server_package.tar.gz -C /workspace/eduroute && rm /workspace/eduroute_server_package.tar.gz && cat /workspace/eduroute/VERSION'
}

up() {
  sed -e "s#__NAME__#$NAME#g" -e "s#__STORAGE__#$STORAGE#g" -e "s#__GPUS__#$GPUS#g" \
      -e "s#__NODEPORT__#$NODEPORT#g" -e "s#__IMAGE__#$IMAGE#g" "$HERE/eduroute.yaml" >"$MANIFEST"
  kubectl apply -f "$MANIFEST"
  echo "Waiting for the pod to start (first image pull can take several minutes)"
  kubectl wait --for=jsonpath='{.status.phase}'=Running "pod/$NAME" --timeout=20m
  inpod 'nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader'
  copy_project
  # The container's start command only runs pod_start.sh at boot; start it now.
  inpod 'cd /workspace && nohup bash eduroute/deploy/gpu/pod_start.sh >/workspace/pod_start.log 2>&1 &'
  echo
  echo "Installing and starting inside the pod (first time ~20-30 min)."
  echo "  Follow:   bash $0 logs        (the access codes are printed there once)"
  echo "  App:      http://$NODE_IP:$NODEPORT   when 'App running' appears"
}

case "${1:-status}" in
  up) up ;;
  status)
    kubectl get pod "$NAME" -o wide; kubectl get svc "$NAME-svc"
    inpod 'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader; cd /workspace/eduroute && bash deploy/gpu/services.sh status' || true
    curl -s -m 5 "http://$NODE_IP:$NODEPORT/_stcore/health" | grep -q ok \
      && echo "App reachable at http://$NODE_IP:$NODEPORT" || echo "App not reachable on $NODE_IP:$NODEPORT yet" ;;
  logs)    kubectl exec "$NAME" -- tail -n 60 -f /workspace/pod_start.log ;;
  shell)   kubectl exec -it "$NAME" -- bash -c 'cd /workspace/eduroute; exec bash' ;;
  eval)    inpod 'cd /workspace/eduroute && nohup bash deploy/gpu/run_evaluations.sh >run/eval.log 2>&1 & echo "Started. Follow: kubectl exec '"$NAME"' -- tail -f /workspace/eduroute/run/eval.log"' ;;
  results)
    latest=$(inpod 'ls -1t /workspace/eduroute/run/eduroute_gpu_results_*.zip 2>/dev/null | head -1')
    [ -n "$latest" ] || { echo "No results zip yet"; exit 1; }
    kubectl cp "$NAME:$latest" "$HOME/$(basename "$latest")" && echo "Copied to ~/$(basename "$latest")" ;;
  update)
    copy_project
    inpod 'cd /workspace/eduroute && bash deploy/gpu/services.sh restart && .venv/bin/python -m deploy.smoke_test' ;;
  down)
    kubectl delete svc "$NAME-svc" --ignore-not-found; kubectl delete pod "$NAME" --ignore-not-found
    echo "Pod deleted. Volume $NAME-data (models, venv, results) kept; 'kubectl delete pvc $NAME-data' removes it." ;;
  *) sed -n '2,15p' "$0"; exit 2 ;;
esac
