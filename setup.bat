@echo off
REM ============================================================
REM GMO-BINN Setup Script for Windows
REM ============================================================
REM Run this script once to set up the project:
REM    setup.bat

setlocal enabledelayedexpansion

echo.
echo ==========================================
echo GMO-BINN Project Setup (Windows)
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo OK: Python %PYTHON_VERSION% found

REM Create virtual environment
echo.
echo Creating virtual environment...
if exist ".venv" (
    echo   (.venv already exists, skipping)
) else (
    python -m venv .venv
    echo   OK: .venv created
)

REM Activate venv
call .venv\Scripts\activate.bat
echo OK: Virtual environment activated

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel >nul 2>&1
echo OK: pip upgraded

REM Install dependencies
echo.
echo Installing dependencies ^(this may take 2-5 minutes^)...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Installation failed. Check your internet connection.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo OK: Setup Complete!
echo ==========================================
echo.
echo Next steps:
echo   1. Activate: .venv\Scripts\activate.bat
echo   2. Verify: python -c "import torch; print('OK: Ready')"
echo   3. Run Phase 1: python -m src.grn_simulator
echo   4. Run full: python run_all.py
echo.
echo For more details, see SETUP_GUIDE.md
echo.
pause