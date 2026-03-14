"""
OPTIMIZED TRAINING SCRIPT
Fine-tunes a sentiment analysis model.
"""

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score

# 1. MODEL SELECTION
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

def compute_metrics(eval_pred):
    """Calculates metrics during the evaluation phase."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }

def main():
    # Check hardware availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running on: {device.upper()} ---")

    # 2. DATASET LOADING
    print("Loading tweet_eval dataset...")
    dataset = load_dataset("tweet_eval", "sentiment")

    # 3. FAST TOKENIZATION
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_function(examples):
        # Truncating to 64 instead of 128 
        return tokenizer(examples["text"], truncation=True, max_length=64)

    print("Mapping dataset (tokenization)...")
    # remove_columns speeds up data handling by dropping raw text after processing
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # 4. DYNAMIC PADDING: Only pads to the longest sentence in the current batch.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5. MODEL INITIALIZATION
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    # 6. TRAINING CONFIGURATION
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",       
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=16,
        num_train_epochs=1,           
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_steps=50,
        fp16=torch.cuda.is_available() 
    )

    # 7. TRAINER INITIALIZATION
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 8. EXECUTION
    print("Starting training loop...")
    trainer.train()

    # 9. SAVING THE MODEL
    print("Saving the fine-tuned model...")
    trainer.save_model("./sentiment_model")
    tokenizer.save_pretrained("./sentiment_model")
    print("Done! Model stored in './sentiment_model'")

if __name__ == "__main__":
    main()