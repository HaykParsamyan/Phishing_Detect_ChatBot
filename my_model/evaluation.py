import torch
import numpy as np
from sklearn.metrics import accuracy_score

from my_model.config import DEVICE

def evaluate_model(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
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
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            batch_labels = batch['labels']

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.detach().cpu()
            batch_preds = torch.argmax(logits, dim=1)

            preds.extend(batch_preds.tolist())
            labels.extend(batch_labels.tolist())
    return np.array(preds), np.array(labels)

def print_detailed_metrics(labels, preds):
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    cm = confusion_matrix(labels, preds)
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1-score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
