@echo off
REM EduRoute AI - system benchmark on this laptop's CPU (about 30 hours).
REM Same 50 questions x 5 strategies (seed 7) as the Colab runs, so results compare.
REM Safe to re-run at any time: it resumes from the partial file and skips
REM answers that are already finished. Progress: outputs\cpu_benchmark.log
REM Needs Ollama running (it starts with Windows) with qwen2.5:3b.

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set EDUROUTE_RAG_VERSION=engineering_v1
set EDUROUTE_LLM_BASE_URL=http://localhost:11434/v1
set EDUROUTE_LLM_MODEL=qwen2.5:3b
REM A 10-mark answer takes ~11 minutes on this CPU; the GPU default (180 s) is too short.
set EDUROUTE_LLM_TIMEOUT=1800
set EDUROUTE_NLI_MODEL=cross-encoder/nli-deberta-v3-xsmall

set PARTIAL=evaluation\results\system_benchmark_partial_cpu_run1.jsonl
if not exist "%PARTIAL%" type nul > "%PARTIAL%"

echo ===== started %date% %time% >> outputs\cpu_benchmark.log
venv\Scripts\python -u -m evaluation.run_system_benchmark --sample 20 --seed 7 --keep-awake --resume "%PARTIAL%" >> outputs\cpu_benchmark.log 2>&1
echo ===== finished %date% %time% (exit %errorlevel%) >> outputs\cpu_benchmark.log
