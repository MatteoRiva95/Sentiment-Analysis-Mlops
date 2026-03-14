"""
MONITORING SERVER
A simple FastAPI server that exposes prediction logs as a JSON endpoint.
This allows Grafana to read the data for real-time visualization.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import json

app = FastAPI(title="MachineInnovators Monitoring API")

LOG_FILE = Path("monitoring_logs/logs.jsonl")

@app.get("/health")
def health_check():
    """Simple health check for the monitoring server."""
    return {"status": "online", "file_exists": LOG_FILE.exists()}

@app.get("/logs")
def get_logs():
    """
    Reads the JSONL file and returns a list of log entries.
    """
    if not LOG_FILE.exists():
        # Instead of an empty list, we return a clear message or empty array
        return []

    try:
        with LOG_FILE.open("r", encoding="utf-8") as f:
            # Efficiently parse each line as a JSON object
            logs = [json.loads(line) for line in f if line.strip()]
        
        return JSONResponse(content=logs)
    
    except Exception as e:
        # If the file is being written to or is corrupted, return a 500 error
        raise HTTPException(status_code=500, detail=f"Error reading logs: {str(e)}")

# TO RUN:
# pip install uvicorn
# uvicorn monitoring_server:app --reload --port 8001