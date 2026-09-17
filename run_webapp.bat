@echo off
chcp 65001 >nul
cd /d "%~dp0"

set PYTHONPATH=%CD%;%PYTHONPATH%

echo ========================================
echo StatAgent Web Application
echo ========================================
echo PYTHONPATH=%PYTHONPATH%
echo.

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m streamlit run statagent\app\webapp.py --server.port 8501 --server.address localhost
) else if exist "E:\anconda\python.exe" (
  "E:\anconda\python.exe" -m streamlit run statagent\app\webapp.py --server.port 8501 --server.address localhost
) else (
  python -m streamlit run statagent\app\webapp.py --server.port 8501 --server.address localhost
)

pause
