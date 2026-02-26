import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

from transformers import (
    AutoModelForSequenceClassification,
    DebertaV2Tokenizer,
    get_linear_schedule_with_warmup,
)

from my_model.config import DEVICE, MAX_LEN, MODEL_PATH, TOKENIZER_PATH, MODEL_NAME
from my_model.data_utils import load_and_prepare_dataset
from my_model.dataset import EmailDataset
from my_model.evaluation import evaluate_model, get_predictions, print_detailed_metrics

training_in_progress = False
model = None
tokenizer = None


def _make_class_weights(df):
    # label 0=legit, 1=phish
    counts = df["label"].value_counts().sort_index()
    total = int(counts.sum())

    legit = int(counts.get(0, 1))
    phish = int(counts.get(1, 1))

    # Balanced weighting: total/(2*count)
    w0 = total / (2 * legit)
    w1 = total / (2 * phish)

    weights = torch.tensor([w0, w1], dtype=torch.float)
    return weights


def train_model(
    sample_frac: float = 1.0,
    batch_size: int = 2,
    epochs: int = 3,
    lr: float = 2e-5,
    accumulation_steps: int = 8,
):
    """
    Recommended for RTX 3050 6GB:
    batch_size=2, accumulation_steps=8 => effective_batch=16
    MAX_LEN should be 256.
    """
    global training_in_progress, model, tokenizer
    training_in_progress = True

    print("--- Starting DeBERTa-v3 Training (Weighted Loss) ---")
    print(
        f"DEVICE={DEVICE} | MAX_LEN={MAX_LEN} | batch_size={batch_size} | "
        f"accumulation_steps={accumulation_steps} | effective_batch={batch_size * accumulation_steps}"
    )

    df = load_and_prepare_dataset(sample_frac)

    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
    )

    print(f"Split sizes => Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Tokenizer (stable)
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)

    # Model (use safetensors)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        use_safetensors=True
    ).to(DEVICE)

    # Tokenize
    train_enc = tokenizer(train_df["email_text"].tolist(), truncation=True, padding=True, max_length=MAX_LEN)
    val_enc = tokenizer(val_df["email_text"].tolist(), truncation=True, padding=True, max_length=MAX_LEN)
    test_enc = tokenizer(test_df["email_text"].tolist(), truncation=True, padding=True, max_length=MAX_LEN)

    train_ds = EmailDataset(train_enc, train_df["label"].tolist())
    val_ds = EmailDataset(val_enc, val_df["label"].tolist())
    test_ds = EmailDataset(test_enc, test_df["label"].tolist())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr)

    # ✅ class-weighted loss (fix imbalance bias)
    class_weights = _make_class_weights(train_df).to(DEVICE)
    print("Class weights [legit, phish]:", class_weights.detach().cpu().tolist())
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Scheduler (count optimizer updates, not batches)
    updates_per_epoch = (len(train_loader) + accumulation_steps - 1) // accumulation_steps
    total_updates = max(1, updates_per_epoch * epochs)
    warmup_updates = max(1, int(0.1 * total_updates))

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_updates,
        num_training_steps=total_updates
    )

    use_amp = (DEVICE.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_acc = 0.0
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(TOKENIZER_PATH, exist_ok=True)

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        model.train()
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()
            total_loss += float(loss.item())
            progress.set_postfix({"loss": float(loss.item())})

            update_now = ((step + 1) % accumulation_steps == 0) or ((step + 1) == len(train_loader))
            if update_now:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Train Avg Loss: {avg_loss:.4f}")

        val_acc = evaluate_model(model, val_loader, device=DEVICE)
        print(f"Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(MODEL_PATH)
            tokenizer.save_pretrained(TOKENIZER_PATH)
            print(f"✅ Saved BEST model (val_acc={best_val_acc:.4f})")

    print("\n✅ Training finished. Loading BEST checkpoint for final evaluation...")
    best_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, use_safetensors=True
    ).to(DEVICE)

    print("\n==== FINAL METRICS (BEST CHECKPOINT) ====")
    print(f"Train Accuracy: {evaluate_model(best_model, train_loader, device=DEVICE):.4f}")
    print(f"Val Accuracy:   {evaluate_model(best_model, val_loader, device=DEVICE):.4f}")
    print(f"Test Accuracy:  {evaluate_model(best_model, test_loader, device=DEVICE):.4f}")

    preds, labels = get_predictions(best_model, test_loader, device=DEVICE)
    print_detailed_metrics(labels, preds)

    training_in_progress = False