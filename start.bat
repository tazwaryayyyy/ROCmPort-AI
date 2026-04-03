@echo off
echo ROCmPort AI - Starting Backend Server...
echo.

cd /d "%~dp0backend"

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Setting up environment...
if not exist .env (
    echo Creating .env file from template...
    copy .env.example .env
    echo Please edit .env file and add your GROQ_API_KEY
    echo.
)

echo.
echo Starting FastAPI server...
echo Server will be available at: http://localhost:8000
echo Frontend should be opened at: http://localhost:8000/index.html
echo.
echo Press Ctrl+C to stop the server
echo.

uvicorn main:app --reload --port 8000 --host 0.0.0.0
