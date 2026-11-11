@echo off
REM Start EduRoute AI (chat + Evaluation Studio) on this laptop.
REM Open http://localhost:8501 here, or http://<this laptop's IP>:8501 from
REM another laptop on the same Wi-Fi (the studio saves everything on this laptop).
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
venv\Scripts\streamlit run app.py --server.port 8501
