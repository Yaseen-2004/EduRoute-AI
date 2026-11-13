@echo off
REM EduRoute AI - everything that runs on this laptop's CPU, in order, unattended:
REM   1. the full evaluation (smoke, 140-question benchmark, images, voice, claims)
REM   2. the with/without comparison of the agentic-RAG, context, memory and cache features
REM Resumable: run it again after any interruption (restart, closed lid) and it continues.
REM Progress: outputs\cpu_evaluation.log  and  outputs\feature_ablation.log
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
venv\Scripts\python -u -m evaluation.run_cpu_evaluation >> outputs\cpu_evaluation_console.log 2>&1
echo %date% %time%  ===== feature ablation: started >> outputs\cpu_evaluation.log
venv\Scripts\python -u -m evaluation.run_feature_ablation >> outputs\feature_ablation.log 2>&1
echo %date% %time%  ===== feature ablation: finished (exit %errorlevel%) >> outputs\cpu_evaluation.log
