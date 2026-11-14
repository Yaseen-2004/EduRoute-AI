@echo off
REM EduRoute - pre-answer the likely exam questions so students get them instantly.
REM Two topics per subject (578 answers, ~12-14 h on this CPU). Resumable: run it again
REM after any interruption and it skips what is already in the bank.
REM Keep the laptop plugged in with the lid open. Progress: outputs\question_bank.log
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
venv\Scripts\python -u -m src.inference.question_bank --per-subject 2 %*
