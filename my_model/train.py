import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from my_model.config import DEVICE, MAX_LEN, MODEL_PATH, TOKENIZER_PATH, MODEL_NAME
from my_model.data_utils import load_and_prepare_dataset
from my_model.dataset import EmailDataset
from my_model.evaluation import get_predictions, print_detailed_metrics


training_in_progress = False
model = None
tokenizer = None


def _make_class_weights_from_counts(legit: int, phish: int) -> torch.Tensor:
    total = legit + phish
    # total/(2*count) classic balanced weights
    w0 = total / (2 * max(1, legit))
    w1 = total / (2 * max(1, phish))
    return torch.tensor([w0, w1], dtype=torch.float)


def train_model(
    sample_frac: float = 1.0,
    batch_size: int = 2,
    epochs: int = 3,
    lr: float = 2e-5,
    accumulation_steps: int = 8,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.10,
):
    """
    Recommended for RTX 3050 6GB:
      batch_size=2, accumulation_steps=8 => effective_batch=16
      MAX_LEN should be 256.
    """
    global training_in_progress, model, tokenizer
    training_in_progress = True

    print("--- Starting Training (DeBERTa-v3) ---")
    print(
        f"DEVICE={DEVICE} | MAX_LEN={MAX_LEN} | batch_size={batch_size} | "
        f"accumulation_steps={accumulation_steps} | effective_batch={batch_size * accumulation_steps}"
    )

    # -------- Load dataset --------
    df = load_and_prepare_dataset(sample_frac)

    # Hard-enforce binary labels
    df = df[df["label"].isin([0, 1])].copy()

    # Report distribution
    counts = df["label"].value_counts().to_dict()
    legit_n = int(counts.get(0, 0))
    phish_n = int(counts.get(1, 0))
    total_n = legit_n + phish_n
    print("\nLabel distribution:", counts)
    if total_n == 0:
        raise ValueError("Dataset is empty after cleaning.")
    if legit_n == 0 or phish_n == 0:
        raise ValueError(f"Need both classes. legit={legit_n}, phish={phish_n}")

    # -------- Split (stratified) --------
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
    )

    print(f"\nSplit sizes => Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # -------- Tokenizer + Model --------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        use_safetensors=True
    ).to(DEVICE)

    # -------- Tokenize --------
    train_enc = tokenizer(
        train_df["email_text"].tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )
    val_enc = tokenizer(
        val_df["email_text"].tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )
    test_enc = tokenizer(
        test_df["email_text"].tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )

    train_ds = EmailDataset(train_enc, train_df["label"].tolist())
    val_ds = EmailDataset(val_enc, val_df["label"].tolist())
    test_ds = EmailDataset(test_enc, test_df["label"].tolist())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # -------- Optimizer --------
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # -------- Loss (use class weights ONLY if imbalance is meaningful) --------
    train_counts = train_df["label"].value_counts().to_dict()
    train_legit = int(train_counts.get(0, 0))
    train_phish = int(train_counts.get(1, 0))

    ratio = max(train_legit, train_phish) / max(1, min(train_legit, train_phish))
    use_weights = ratio >= 1.5  # only if imbalance is big enough

    if use_weights:
        class_weights = _make_class_weights_from_counts(train_legit, train_phish).to(DEVICE)
        print("Using class-weighted loss. Weights [legit, phish]:", class_weights.detach().cpu().tolist())
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        print("Using standard CrossEntropyLoss (dataset roughly balanced).")
        loss_fn = torch.nn.CrossEntropyLoss()

    # -------- Scheduler (count optimizer updates, not batches) --------
    updates_per_epoch = (len(train_loader) + accumulation_steps - 1) // accumulation_steps
    total_updates = max(1, updates_per_epoch * epochs)
    warmup_updates = max(1, int(warmup_ratio * total_updates))

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_updates,
        num_training_steps=total_updates
    )

    # -------- AMP --------
    use_amp = (DEVICE.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # -------- Save dirs --------
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(TOKENIZER_PATH, exist_ok=True)

    best_val_f1 = -1.0

    # -------- Train loop --------
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        model.train()
        optimizer.zero_grad(set_to_none=True)

        total_raw_loss = 0.0
        progress = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                raw_loss = loss_fn(outputs.logits, labels)
                loss = raw_loss / accumulation_steps

            scaler.scale(loss).backward()
            total_raw_loss += float(raw_loss.item())
            progress.set_postfix({"raw_loss": float(raw_loss.item())})

            update_now = ((step + 1) % accumulation_steps == 0) or ((step + 1) == len(train_loader))
            if update_now:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        avg_loss = total_raw_loss / max(1, len(train_loader))
        print(f"Train Avg Loss: {avg_loss:.4f}")

        # -------- Validation (choose best by F1 for phishing class) --------
        val_preds, val_labels = get_predictions(model, val_loader, device=DEVICE)
        val_f1 = f1_score(val_labels, val_preds, pos_label=1)

        print(f"Validation F1 (phish=1): {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(MODEL_PATH)
            tokenizer.save_pretrained(TOKENIZER_PATH)
            print(f"✅ Saved BEST model (val_f1={best_val_f1:.4f})")

    # -------- Final evaluation on best checkpoint --------
    print("\n✅ Training finished. Loading BEST checkpoint for final evaluation...")

    best_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, use_safetensors=True
    ).to(DEVICE)

    print("\n==== FINAL METRICS (BEST CHECKPOINT) ====")

    train_preds, train_labels = get_predictions(best_model, train_loader, device=DEVICE)
    val_preds, val_labels = get_predictions(best_model, val_loader, device=DEVICE)
    test_preds, test_labels = get_predictions(best_model, test_loader, device=DEVICE)

    print("\n--- Train ---")
    print_detailed_metrics(train_labels, train_preds)

    print("\n--- Val ---")
    print_detailed_metrics(val_labels, val_preds)

    print("\n--- Test ---")
    print_detailed_metrics(test_labels, test_preds)

    training_in_progress = False
    return True