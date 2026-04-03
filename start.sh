#!/bin/bash

echo "ROCmPort AI - Starting Backend Server..."
echo

cd "$(dirname "$0")/backend"

echo "Installing dependencies..."
pip install -r requirements.txt

echo
echo "Setting up environment..."
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file and add your GROQ_API_KEY"
    echo
fi

echo
echo "Starting FastAPI server..."
echo "Server will be available at: http://localhost:8000"
echo "Frontend should be opened at: http://localhost:8000/index.html"
echo
echo "Press Ctrl+C to stop the server"
echo

uvicorn main:app --reload --port 8000 --host 0.0.0.0
