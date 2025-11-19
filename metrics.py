import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score


def evaluate_model(X, y_true, model, dataset_name="Dataset"):
    """
    Calculates and prints key performance metrics for the trained model.

    Args:
        X (sparse matrix or array): Feature matrix.
        y_true (array): True labels.
        model (XGBClassifier): The trained classification model.
        dataset_name (str): Name of the dataset being evaluated (e.g., 'Validation', 'Test').
    """
    if model is None:
        print(f"Evaluation skipped for {dataset_name}: Model is not initialized.")
        return

    try:
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]  # Probability of the positive class (phishing)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)

        # Calculate True Positives, False Positives, etc.
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print(f"\n--- Evaluation Results ({dataset_name}) ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC Score: {auc:.4f}")
        print("-" * 35)
        print(f"Confusion Matrix:\n[[TN: {tn}, FP: {fp}],\n [FN: {fn}, TP: {tp}]]")
        print("---------------------------------")

    except Exception as e:
        print(f"An error occurred during evaluation of {dataset_name}: {e}")

# Note: This file must be present to resolve the ModuleNotFoundError in model.py