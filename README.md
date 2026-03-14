---
title: Sentiment Analysis with MLOps Practices
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.21.0
app_file: app.py
pinned: false
---

# Simple Sentiment Analysis with MLOps Practices
This project implements a lightweight sentiment analysis system using a HuggingFace model and a minimal MLOps workflow, for MachineInnovators Inc. to monitor corporate reputation on social media. It includes training, inference, monitoring, and a simple CI pipeline. The goal is to provide a clear, easy‑to‑understand structure suitable for learning or small‑scale deployments.

# Features
## Pretrained Model  
Uses "cardiffnlp/twitter-roberta-base-sentiment-latest" for sentiment classification (negative, neutral, positive).

## Training Script  
Fine‑tunes the model on a public dataset.

## Inference Script  
Loads the trained model and returns predictions with probabilities.

## Monitoring System  
Every prediction is logged into a JSONL file.
A small FastAPI server exposes these logs so Grafana can visualize them using the JSON API plugin.

## Web App  
A simple Gradio interface for interactive sentiment analysis.

## CI-CD Pipeline  
GitHub Actions workflow that installs dependencies and runs a basic integration test.

## Project Structure
```
SENTIMENT-ANALYSIS-MLOPS/
├── .github/
│   └── workflows/
│       └── ci_cd.yml          
├── tests/
│   └── test_logic.py          
├── app.py                     
├── inference.py               
├── monitoring.py              
├── monitoring_server.py       
├── README.md                  
├── requirements.txt           
└── train.py                   
```

# How to Run (GitHub Codespaces)
Install dependencies:
```
pip install -r requirements.txt
```

# Train the model:
```
python train.py
```

# Run inference:
```
python inference.py
```

# Start the monitoring server:

```
uvicorn monitoring_server:app --reload --port 8001
```

# Start the Gradio app:

```
python app.py
```

# Grafana Monitoring
## 1. Run Grafana (Codespaces supports port forwarding):
```
docker run -d -p 3000:3000 grafana/grafana
```

## 2. Install the JSON API plugin inside Grafana.

## 3. Add a new data source pointing to:
```
http://localhost:8001/logs
```

## 4. Build dashboards using fields like:

- predicted_label
- timestamp
- scores.positive, scores.neutral, scores.negative
