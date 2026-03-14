"""
MONITORING LOGGER
Records predictions into a JSONL file. 
Designed for easy integration with Grafana or monitoring dashboards.
"""

import json
from datetime import datetime
from pathlib import Path

# Setup logging directory and file path
LOG_DIR = Path("monitoring_logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "logs.jsonl"

def log_prediction(text: str, prediction: dict):
    """
    Saves a single prediction entry to the JSONL log file.
    """
    # Create a structured entry with ISO timestamp
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z", # 'Z' indicates UTC time
        "text": text,
        "predicted_label": prediction["label"],
        "confidence": max(prediction["scores"].values()), # Extract highest score for quick monitoring
        "scores": prediction["scores"]
    }

    # Append the entry as a new line in the JSONL file
    # ensure_ascii=False is crucial for correctly saving emojis/special characters in tweets
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # Quick test to verify logging works
    test_pred = {"label": "positive", "scores": {"positive": 0.9, "neutral": 0.05, "negative": 0.05}}
    log_prediction("Test tweet for MachineInnovators!", test_pred)
    print(f"Log entry saved to {LOG_FILE}")