import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from monitoring import log_prediction

# --- Load model and tokenizer once at the start ---
# This makes the script much faster
MODEL_PATH = "sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

def predict(text: str):
    """
    Predicts sentiment for a given text.
    Simplified version: loads model once, runs fast.
    """
    start = time.time()

    # Prepare inputs for the model
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Inference without tracking gradients
    with torch.no_grad():
        outputs = model(**inputs)

    # Get probabilities and the highest score
    probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
    pred_id = int(torch.argmax(outputs.logits))

    # Map ID to human-readable label
    id2label = model.config.id2label
    
    result = {
        "label": id2label[pred_id],
        "scores": {id2label[i]: float(probs[i]) for i in range(3)},
        "elapsed_time": time.time() - start
    }

    # Log prediction for Phase 3
    log_prediction(text, result)

    return result

if __name__ == "__main__":
    print(predict("MachineInnovators Inc. is great!"))