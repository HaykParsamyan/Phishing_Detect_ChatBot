import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer

from my_model.config import DEVICE, MAX_LEN, MODEL_PATH, TOKENIZER_PATH, MODEL_NAME
from my_model.data_utils import load_and_prepare_dataset
from my_model.dataset import EmailDataset
from my_model.evaluation import evaluate_model, get_predictions, print_detailed_metrics

training_in_progress = False
model = None
tokenizer = None


def train_model(sample_frac=1.0, batch_size=2, epochs=1, lr=2e-5):
    global training_in_progress, model, tokenizer
    training_in_progress = True

    print("--- Starting DeBERTa-v3 Model Training ---")

    df = load_and_prepare_dataset(sample_frac)

    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
    )

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # ✅ Stable tokenizer (no tiktoken conversion)
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)

    # ✅ Force safetensors to bypass torch>=2.6 restriction
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        use_safetensors=True
    )
    model.to(DEVICE)

    # Tokenize
    train_encodings = tokenizer(
        train_df["email_text"].tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )
    val_encodings = tokenizer(
        val_df["email_text"].tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )
    test_encodings = tokenizer(
        test_df["email_text"].tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )

    train_dataset = EmailDataset(train_encodings, train_df["label"].tolist())
    val_dataset = EmailDataset(val_encodings, val_df["label"].tolist())
    test_dataset = EmailDataset(test_encodings, test_df["label"].tolist())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    use_amp = (DEVICE.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

        for batch in progress_bar:
            optimizer.zero_grad(set_to_none=True)

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            progress_bar.set_postfix({"loss": float(loss.item())})

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Train Avg Loss: {avg_loss:.4f}")

        val_acc = evaluate_model(model, val_loader, device=DEVICE)
        print(f"Validation Accuracy: {val_acc:.4f}")

    # Save model + tokenizer
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(TOKENIZER_PATH, exist_ok=True)

    model.save_pretrained(MODEL_PATH)        # will save safetensors if available
    tokenizer.save_pretrained(TOKENIZER_PATH)

    print("Model & Tokenizer Saved.")

    # Final metrics
    print("\n==== FINAL METRICS ====")
    print(f"Train Accuracy: {evaluate_model(model, train_loader, device=DEVICE):.4f}")
    print(f"Val Accuracy:   {evaluate_model(model, val_loader, device=DEVICE):.4f}")
    print(f"Test Accuracy:  {evaluate_model(model, test_loader, device=DEVICE):.4f}")

    preds, labels = get_predictions(model, test_loader, device=DEVICE)
    print_detailed_metrics(labels, preds)

    training_in_progress = False