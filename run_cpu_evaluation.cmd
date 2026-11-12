@echo off
REM EduRoute AI - the complete evaluation on this laptop's CPU (about 8-10 hours).
REM Keep the laptop plugged in with the lid open. Safe to close this window and
REM run it again at any time: finished stages are skipped and the benchmark resumes.
REM Progress: outputs\cpu_evaluation.log    Results: evaluation\results\
REM Optional 7B vision comparison (needs ~8 GB free RAM): run_cpu_evaluation.cmd --with-7b
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
venv\Scripts\python -u -m evaluation.run_cpu_evaluation %*
pause
