import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from my_model.config import MODEL_PATH, TOKENIZER_PATH, DEVICE

model = None
tokenizer = None
training_in_progress = False


def load_trained_model():
    """
    Loads the trained model + tokenizer.
    Raises clear errors instead of silently failing like before.
    """
    global model, tokenizer

    # Validate both paths
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"[ERROR] MODEL_PATH not found: {MODEL_PATH}")

    if not os.path.exists(TOKENIZER_PATH):
        raise RuntimeError(f"[ERROR] TOKENIZER_PATH not found: {TOKENIZER_PATH}")

    # Load tokenizer
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_PATH)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to load tokenizer: {e}")

    # Load model
    try:
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(DEVICE)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to load model: {e}")

    print("[INFO] Trained model successfully loaded.")
    return True
