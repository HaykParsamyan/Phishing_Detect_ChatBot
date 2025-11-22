import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from my_model.config import DEVICE, MAX_LEN, MODEL_PATH, TOKENIZER_PATH
from my_model.data_utils import load_and_prepare_dataset
from dataset import EmailDataset
from my_model.evaluation import evaluate_model, get_predictions, print_detailed_metrics
import os

training_in_progress = False
model = None
tokenizer = None

def train_model(sample_frac=1.0, batch_size=16, epochs=1, lr=2e-5):
    global training_in_progress, model, tokenizer
    training_in_progress = True

    print("--- Starting DistilBERT Model Training ---")
    df = load_and_prepare_dataset(sample_frac)
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.to(DEVICE)

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
    scaler = torch.amp.GradScaler()

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

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

    # Save model
    os.makedirs(MODEL_PATH, exist_ok=True)
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(TOKENIZER_PATH)
    print("Model & Tokenizer Saved.")

    # Final metrics
    print("\n==== FINAL METRICS ====")
    print(f"Train Accuracy: {evaluate_model(model, train_loader):.4f}")
    print(f"Val Accuracy: {evaluate_model(model, val_loader):.4f}")
    print(f"Test Accuracy: {evaluate_model(model, test_loader):.4f}")
    preds, labels = get_predictions(model, test_loader)
    print_detailed_metrics(labels, preds)

    training_in_progress = False
