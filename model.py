# model.py
import os
import time
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from dataset import EmailDataset  # Your custom torch Dataset class
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Globals ---
training_in_progress = False
MODEL_PATH = "models/bert_phishing"
TOKENIZER_PATH = "models/bert_phishing_tokenizer"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
tokenizer = None

# --- Dataset ---
DATA_PATH = "final_data/all_phishing_master_dataset.csv"
MAX_LEN = 512

def load_and_prepare_dataset(sample_frac=1.0):
    df = pd.read_csv(DATA_PATH, low_memory=False)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
    df = df.reset_index(drop=True)
    df['email_text'] = df['subject'].astype(str) + " " + df['body'].astype(str)
    return df

# --- Training function ---
def train_model(sample_frac=1.0, batch_size=8, epochs=1, lr=2e-5):
    global training_in_progress, model, tokenizer
    training_in_progress = True
    print("--- Starting BERT Model Training ---")

    df = load_and_prepare_dataset(sample_frac)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Load tokenizer & model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.to(device)

    # Encode data
    train_encodings = tokenizer(train_df['email_text'].tolist(), truncation=True, padding=True, max_length=MAX_LEN)
    val_encodings = tokenizer(val_df['email_text'].tolist(), truncation=True, padding=True, max_length=MAX_LEN)

    train_labels = train_df['label'].tolist()
    val_labels = val_df['label'].tolist()

    train_dataset = EmailDataset(train_encodings, train_labels)
    val_dataset = EmailDataset(val_encodings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    # --- Training loop ---
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        model.train()
        total_loss = 0
        start_time = time.time()

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            # Print progress every batch or every 10 batches
            if i % 10 == 0 or i == len(train_loader) - 1:
                elapsed = time.time() - start_time
                percent = (i + 1) / len(train_loader) * 100
                batches_per_sec = (i + 1) / elapsed
                eta = (len(train_loader) - (i + 1)) / batches_per_sec
                print(f"Batch {i + 1}/{len(train_loader)} - Loss: {loss.item():.4f} "
                      f"| {percent:.2f}% complete | ETA: {eta/60:.1f} min | Device: {device}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")

    # Save model & tokenizer
    os.makedirs(MODEL_PATH, exist_ok=True)
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(TOKENIZER_PATH)
    print(f"Model and tokenizer saved to {MODEL_PATH} / {TOKENIZER_PATH}")

    training_in_progress = False

# --- Load trained model ---
def load_trained_model():
    global model, tokenizer
    if os.path.exists(MODEL_PATH):
        tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        return True
    return False

# --- Quick test helper ---
def quick_test_train():
    """Train on 1% of dataset for fast test."""
    train_model(sample_frac=0.01, batch_size=4, epochs=1)

# --- Email Prediction Function ---
def predict_email(email_text):
    global model, tokenizer, device
    if model is None or tokenizer is None:
        raise ValueError("Model or tokenizer not loaded. Run `load_trained_model()` first.")

    inputs = tokenizer(email_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        phishing_prob = float(probs[1])
        safe_prob = float(probs[0])
        result = "phishing" if phishing_prob > safe_prob else "legitimate"

    return result, phishing_prob, safe_prob
