"""
MONITORING SERVER
Serves the JSONL logs through a simple API so Grafana can read them.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pathlib import Path
import json

app = FastAPI()

LOG_FILE = Path("monitoring_logs/logs.jsonl")

@app.get("/logs")
def get_logs():
    """Return all logs as JSON list."""
    if not LOG_FILE.exists():
        return []

    with LOG_FILE.open("r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f.readlines()]

    return JSONResponse(lines)

# Run with:
# uvicorn monitoring_server:app --reload --port 8001
