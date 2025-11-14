# metrics.py

from sklearn.metrics import classification_report, accuracy_score, f1_score


def evaluate_model(X_set, y_set, clf, name):
    """Calculates and prints performance metrics for a given dataset."""

    y_pred = clf.predict(X_set)

    print(f"\n{name} Accuracy: {accuracy_score(y_set, y_pred):.4f}")
    print(f"{name} F1 Score: {f1_score(y_set, y_pred):.4f}")
    print(
        f"{name} Classification Report:\n{classification_report(y_set, y_pred, target_names=['legitimate', 'phishing'])}")