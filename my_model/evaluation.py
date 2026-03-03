import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

def evaluate_model(model, data_loader, device=None) -> float:
    """
    Returns accuracy for quick monitoring.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    return accuracy_score(all_labels, all_preds)

def get_predictions(model, data_loader, device=None):
    """
    Returns (preds, labels) for detailed metrics.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    return all_preds, all_labels

def print_detailed_metrics(labels, preds):
    """
    Prints precision/recall/f1 + confusion matrix + report.
    Assumes: 0 = legitimate, 1 = phishing (match your predict mapping).
    """
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", pos_label=1)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix [ [TN FP], [FN TP] ]:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["legitimate", "phishing"]))
