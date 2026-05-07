@echo off
echo ROCmPort AI - Starting Backend Server...
echo.

cd /d "%~dp0"

echo Installing dependencies...
if not exist .venv (
    python -m venv .venv
)
call ".venv\Scripts\activate.bat"
pip install -r backend\requirements.txt

echo.
echo Setting up environment...
if not exist .env (
    echo Creating .env file from template...
    copy .env.example .env
    echo Please edit .env file and add your GROQ_API_KEY
    echo.
)

echo.
echo Building frontend...
npm --prefix frontend install
npm --prefix frontend run build
if errorlevel 1 (
    echo Frontend build failed. Make sure Node.js and npm are installed.
    exit /b 1
)

echo.
echo Starting FastAPI server...
echo Server will be available at: http://localhost:8000
echo Frontend should be opened at: http://localhost:8000/index.html
echo.
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn backend.main:app --reload --port 8000 --host 0.0.0.0
