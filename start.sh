#!/bin/bash
set -e

echo "🚦 Starting Chaos Engine..."

# Start FastAPI OpenEnv server in background (needed for validation & inference.py)
echo "  → Launching OpenEnv server on :8000"
uvicorn server.app:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to be ready
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "  ✅ Backend ready"
        break
    fi
    sleep 1
done

# Start Gradio UI in foreground (uses environment directly, no HTTP needed)
echo "  → Launching Gradio UI on :7860"
python ui.py
