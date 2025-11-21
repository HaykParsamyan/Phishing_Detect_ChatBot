import os
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import EmailDataset
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

# --- Globals ---
training_in_progress = False
MODEL_PATH = "models/distilbert_phishing"
TOKENIZER_PATH = "models/distilbert_phishing_tokenizer"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
tokenizer = None
DATA_PATH = "final_data/all_phishing_master_dataset.csv"
MAX_LEN = 512


# ============================================================
# Load & Prepare Dataset
# ============================================================
def load_and_prepare_dataset(sample_frac=1.0):
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Fill missing
    df['subject'] = df['subject'].fillna('')
    df['body'] = df['body'].fillna('')

    # Combine subject + body
    df['email_text'] = df['subject'] + ' ' + df['body']

    # Keep only text and label
    df = df[['email_text', 'label']]

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)

    df = df.reset_index(drop=True)
    return df


# ============================================================
# Evaluation helpers
# ============================================================
def evaluate_model(model, loader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels']

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.detach().cpu()
            batch_preds = torch.argmax(logits, dim=1)

            preds.extend(batch_preds.tolist())
            labels.extend(batch_labels.tolist())

    return accuracy_score(labels, preds)


def get_predictions(model, loader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels']

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.detach().cpu()
            batch_preds = torch.argmax(logits, dim=1)

            preds.extend(batch_preds.tolist())
            labels.extend(batch_labels.tolist())

    return np.array(preds), np.array(labels)


# ============================================================
# Training Loop
# ============================================================
def train_model(sample_frac=1.0, batch_size=16, epochs=1, lr=2e-5):
    global training_in_progress, model, tokenizer
    training_in_progress = True

    print("--- Starting DistilBERT Model Training ---")

    df = load_and_prepare_dataset(sample_frac)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.to(device)

    train_encodings = tokenizer(train_df['email_text'].tolist(), truncation=True, padding=True, max_length=MAX_LEN)
    val_encodings = tokenizer(val_df['email_text'].tolist(), truncation=True, padding=True, max_length=MAX_LEN)
    test_encodings = tokenizer(test_df['email_text'].tolist(), truncation=True, padding=True, max_length=MAX_LEN)

    train_dataset = EmailDataset(train_encodings, train_df['label'].tolist())
    val_dataset = EmailDataset(val_encodings, val_df['label'].tolist())
    test_dataset = EmailDataset(test_encodings, test_df['label'].tolist())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.amp.autocast(device_type='cuda'):
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Train Avg Loss: {avg_loss:.4f}")
        val_acc = evaluate_model(model, val_loader)
        print(f"Validation Accuracy: {val_acc:.4f}")

    # Save
    os.makedirs(MODEL_PATH, exist_ok=True)
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(TOKENIZER_PATH)
    print("Model & Tokenizer Saved.")

    # Final Metrics
    print("\n==== FINAL METRICS ====")
    print(f"Train Accuracy: {evaluate_model(model, train_loader):.4f}")
    print(f"Val Accuracy:   {evaluate_model(model, val_loader):.4f}")
    print(f"Test Accuracy:  {evaluate_model(model, test_loader):.4f}")

    preds, labels = get_predictions(model, test_loader)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    cm = confusion_matrix(labels, preds)
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1-score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)

    training_in_progress = False


# ============================================================
# Load trained model
# ============================================================
def load_trained_model():
    global model, tokenizer
    if os.path.exists(MODEL_PATH):
        tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_PATH)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        return True
    return False


# ============================================================
# Prediction function
# ============================================================
def predict_email(email_text):
    global model, tokenizer
    if model is None or tokenizer is None:
        raise ValueError("Model/tokenizer not loaded. Run load_trained_model() first.")

    inputs = tokenizer(email_text, return_tensors='pt', truncation=True, padding=True, max_length=MAX_LEN).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits.detach().cpu(), dim=-1).numpy()[0]

    return ("phishing" if probs[1] > probs[0] else "legitimate", float(probs[1]), float(probs[0]))


# Quick test
def quick_test_train():
    train_model(sample_frac=0.01, epochs=1)
