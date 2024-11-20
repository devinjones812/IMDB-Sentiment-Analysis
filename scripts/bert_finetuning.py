from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import joblib

# Load raw reviews and labels
data = joblib.load("preprocessed_data.pkl")
reviews = data["reviews"]  # Assuming reviews are stored in preprocessing.pkl
labels = data["labels"]

# Tokenize reviews
print("Tokenizing data...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encodings = tokenizer(reviews, truncation=True, padding=True, max_length=256, return_tensors="pt")

# Fine-tune BERT
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
training_args = TrainingArguments(output_dir='./results', num_train_epochs=3, per_device_train_batch_size=8)
trainer = Trainer(model=model, args=training_args, train_dataset=encodings, eval_dataset=encodings)

print("Fine-tuning BERT...")
trainer.train()

# Save the model
model.save_pretrained("models/bert_model")
tokenizer.save_pretrained("models/bert_tokenizer")
