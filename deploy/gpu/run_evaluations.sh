#!/usr/bin/env bash
# Every EduRoute evaluation on the GPU, then one zip of the new results.
#   bash deploy/gpu/run_evaluations.sh                       # all stages
#   STAGES="system images" bash deploy/gpu/run_evaluations.sh
#   nohup bash deploy/gpu/run_evaluations.sh >run/eval.log 2>&1 &    # survives logout
# Stages: tests smoke retrieval router system images voice claims
# The system benchmark resumes after an interruption: just run it again.
set -uo pipefail
. "$(dirname "$0")/common.sh"
cd "$PROJECT"
activate_venv
start_ollama

STAGES="${STAGES:-tests smoke retrieval router system images voice claims}"
VISION_MODELS="${VISION_MODELS:-qwen2.5vl:3b qwen2.5vl:7b}"
PARTIAL=evaluation/results/system_benchmark_partial_gpu.jsonl
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p evaluation/results
MARKER="$RUN_DIR/eval_start_$STAMP"; touch "$MARKER"
NOISE='Batches|Loading weights|FutureWarning|torch.jit|it/s\]'
FAILED=""

stage() {  # stage <name> <command...>: run, log, time, and keep going on failure
  local name=$1; shift
  echo; echo "===== $name ($(date +%H:%M:%S))"
  local t0=$SECONDS
  if "$@" 2>&1 | grep -v -E "$NOISE" | tee "$RUN_DIR/eval_${name}_$STAMP.log"; [ "${PIPESTATUS[0]}" -eq 0 ]; then
    echo "----- $name done in $((SECONDS - t0)) s"
  else
    echo "----- $name FAILED after $((SECONDS - t0)) s (log: run/eval_${name}_$STAMP.log)"; FAILED="$FAILED $name"
  fi
}
has() { [[ " $STAGES " == *" $1 "* ]]; }

{ echo "EduRoute GPU evaluation $STAMP"; hostname; nvidia-smi
  python -c 'import torch,sys; print("Python", sys.version.split()[0], "| torch", torch.__version__, "| CUDA", torch.version.cuda)'
  ollama --version 2>&1 | tail -1; ollama list; grep -v -i 'key' .env; git log -1 --oneline 2>/dev/null || cat VERSION 2>/dev/null
} >"evaluation/results/server_info_$STAMP.txt" 2>&1

has tests     && stage tests python -m pytest -q
has smoke     && stage smoke python -m deploy.smoke_test
has retrieval && stage retrieval_engineering python -m evaluation.evaluate_retrieval \
                   --configs engineering_v1:dense --benchmark retrieval_engineering_v1.jsonl
has retrieval && stage retrieval_aiml python -m evaluation.evaluate_retrieval \
                   --configs engineering_v1:dense --benchmark retrieval_aiml_v1.jsonl
has router    && stage router python -m evaluation.check_router_heldout --version engineering_v1

if has system; then
  warm "$(grep -oP '^EDUROUTE_LLM_MODEL=\K.*' .env)"
  touch "$PARTIAL"
  stage system python -m evaluation.run_system_benchmark --seed 7 --resume "$PARTIAL"
fi

if has images; then
  for vm in $VISION_MODELS; do
    if ! ollama list | grep -q "^$vm"; then echo "Skipping $vm: not pulled (ollama pull $vm)"; continue; fi
    warm "$vm"
    stage "images_${vm//[:.]/_}" env EDUROUTE_VISION_MODEL="$vm" python -m evaluation.run_image_benchmark
  done
fi

if has voice; then
  stage voice_synthetic python -m evaluation.evaluate_asr evaluation/voice_eval_synthetic
  [ -f evaluation/voice_eval/references.csv ] && stage voice_students python -m evaluation.evaluate_asr evaluation/voice_eval
fi

has claims && [ -s "$PARTIAL" ] && stage claims python -m evaluation.claim_labels export "$PARTIAL" --n 120

# Collect everything new. Grading keys go in a separate zip: keep them from raters.
cp "$RUN_DIR"/eval_*_"$STAMP".log evaluation/results/ 2>/dev/null
OUT="$RUN_DIR/eduroute_gpu_results_$STAMP.zip"
python - "$MARKER" "$OUT" <<'PY'
import sys, zipfile, os
from pathlib import Path
marker, out = os.path.getmtime(sys.argv[1]), Path(sys.argv[2])
new = [p for p in Path("evaluation/results").rglob("*") if p.is_file() and p.stat().st_mtime >= marker]
keys = [p for p in new if p.name.startswith("grading_key_")]
for target, files in ((out, [p for p in new if p not in keys]),
                      (out.with_name(out.stem.replace("results", "grading_keys") + ".zip"), keys)):
    if files:
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
            for p in files:
                z.write(p, p.as_posix())
        print(f"{len(files)} files -> {target}")
PY
rm -f "$MARKER"
echo
[ -n "$FAILED" ] && echo "Failed stages:$FAILED" || echo "All stages finished."
echo "Copy the zip to your laptop:  scp $(id -un)@$(hostname):$OUT ."
