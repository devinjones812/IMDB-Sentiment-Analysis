import torch
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import joblib
from pathlib import Path
import json
from tqdm.auto import tqdm
import os

# Check if MPS (Metal Performance Shaders) is available
device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")
NUM_WORKERS = os.cpu_count() - 1
print(f"Using {NUM_WORKERS} CPU cores for data loading")

class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='macro')
    }

def prepare_data():
    """Load raw data preserving IMDB's train/test split."""
    print("Loading data...")
    try:
        raw_data = joblib.load("processed/raw_data.pkl")
        
        return {
            'train_texts': raw_data["reviews"],        # 25,000 train reviews
            'train_labels': raw_data["labels"],        # 25,000 train labels
            'test_texts': raw_data["test_reviews"],    # Need to add these to raw_data.pkl
            'test_labels': raw_data["test_labels"],    # Need to add these to raw_data.pkl
            'unsup_texts': raw_data["unsupervised_reviews"]  # 50,000 unlabeled
        }
        
    except FileNotFoundError:
        raise FileNotFoundError("Raw data not found. Run preprocessing.py first.")

def tokenize_data(texts, tokenizer):
    """Tokenize texts with progress bar."""
    encodings = []
    for i in tqdm(range(0, len(texts), 100), desc="Tokenizing texts"):
        batch = texts[i:i + 100]
        batch_encodings = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        encodings.append(batch_encodings)
    
    # Combine batches
    combined_encodings = {
        key: torch.cat([enc[key] for enc in encodings])
        for key in encodings[0].keys()
    }
    return combined_encodings

class ProgressCallback(EarlyStoppingCallback):
    """Custom callback with progress bars."""
    def __init__(self, early_stopping_patience=3):
        super().__init__(early_stopping_patience=early_stopping_patience)
        self.training_bar = None
        self.epoch_bar = None
        self.current_epoch = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.epoch_bar = tqdm(total=args.num_train_epochs, desc="Epochs")
        print(f"\nStarting training: {args.num_train_epochs} epochs")
        return super().on_train_begin(args, state, control, **kwargs)

    def on_epoch_end(self, args, state, control, **kwargs):
        self.current_epoch += 1
        self.epoch_bar.update(1)
        print(f"\nEpoch {self.current_epoch}/{args.num_train_epochs} completed")
        return super().on_epoch_end(args, state, control, **kwargs)

    def on_train_end(self, args, state, control, **kwargs):
        self.epoch_bar.close()
        print("\nTraining completed!")
        return super().on_train_end(args, state, control, **kwargs)

def main():
    # Create directories
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    # Prepare data
    data = prepare_data()
    
    # Initialize tokenizer
    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize data with progress bars
    print("Tokenizing data...")
    train_encodings = tokenize_data(data['train_texts'], tokenizer)
    test_encodings = tokenize_data(data['test_texts'], tokenizer)
    
    # Create datasets
    train_dataset = IMDBDataset(train_encodings, data['train_labels'])
    test_dataset = IMDBDataset(test_encodings, data['test_labels'])
    
    print("\nInitializing BERT model...")
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    # Move model to appropriate device
    model = model.to(device)
    
    # Training arguments optimized for M3 Max
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=250,
        save_steps=250,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        save_total_limit=2,
        learning_rate=2e-5,
        gradient_accumulation_steps=2,
        fp16=True if device != "cpu" else False,
        dataloader_num_workers=NUM_WORKERS,
        dataloader_pin_memory=True,
        report_to="none"  # Disable wandb
    )
    
    # Initialize trainer with progress callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[ProgressCallback(early_stopping_patience=3)]
    )
    
    print("\nStarting BERT training...")
    trainer.train()
    
    print("\nEvaluating model...")
    eval_results = trainer.evaluate()
    
    # Save results
    results = {
        "eval_results": eval_results,
        "model_config": model.config.to_dict(),
        "training_args": training_args.to_dict(),
        "device_info": {
            "device": device,
            "num_workers": NUM_WORKERS
        }
    }
    
    with open("results/bert_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nSaving model...")
    model.save_pretrained("models/bert_model")
    tokenizer.save_pretrained("models/bert_tokenizer")
    
    print(f"\nFinal Results:")
    print(f"F1 Score: {eval_results['eval_f1']:.4f}")
    print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")

if __name__ == "__main__":
    main()