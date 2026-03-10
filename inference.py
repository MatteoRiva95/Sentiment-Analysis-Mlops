"""
INFERENCE SCRIPT
Loads the trained model and predicts sentiment for a given text.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from monitoring import log_prediction
import time

MODEL_PATH = "model"

def predict(text: str):
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
    pred_id = int(torch.argmax(outputs.logits))

    id2label = model.config.id2label
    result = {
        "label": id2label[pred_id],
        "scores": {id2label[i]: float(probs[i]) for i in range(3)}
    }

    # Log prediction
    log_prediction(text, result)

    return result

if __name__ == "__main__":
    print(predict("I love this product"))
