"""
MONITORING LOGGER
Logs each prediction into a JSONL file for Grafana visualization.
"""

import json
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("monitoring_logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "logs.jsonl"

def log_prediction(text: str, prediction: dict):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "text": text,
        "predicted_label": prediction["label"],
        "scores": prediction["scores"]
    }

    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
