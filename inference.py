import torch
import time
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from monitoring import log_prediction

# --- Model Selection Logic ---
# Use the local fine-tuned model if available, otherwise fallback to the official one
LOCAL_MODEL = "sentiment_model"
OFFICIAL_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

if os.path.exists(LOCAL_MODEL):
    MODEL_PATH = LOCAL_MODEL
    print(f"Loading local model from: {LOCAL_MODEL}")
else:
    MODEL_PATH = OFFICIAL_MODEL
    print(f"Local model not found. Falling back to: {OFFICIAL_MODEL}")

# --- Load model and tokenizer once ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

def predict(text: str):
    """
    Predicts sentiment for a given text and logs the result.
    """
    start = time.time()

    # Prepare inputs
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get probabilities
    probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
    pred_id = int(torch.argmax(outputs.logits))

    # Map ID to label
    id2label = model.config.id2label
    
    result = {
        "label": id2label[pred_id],
        "scores": {id2label[i]: float(probs[i]) for i in range(len(id2label))},
        "elapsed_time": time.time() - start
    }

    # Log prediction for Phase 3 (Grafana monitoring)
    log_prediction(text, result)

    return result

if __name__ == "__main__":
    # Test call
    print(predict("MachineInnovators Inc. is great!"))