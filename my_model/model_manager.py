import os
from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer
from my_model.config import MODEL_PATH, TOKENIZER_PATH, DEVICE

model = None
tokenizer = None
training_in_progress = False


def load_trained_model():
    """
    Loads trained model + tokenizer from disk.
    Uses safetensors to bypass torch>=2.6 restriction.
    """
    global model, tokenizer

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"[ERROR] MODEL_PATH not found: {MODEL_PATH}")

    if not os.path.exists(TOKENIZER_PATH):
        raise RuntimeError(f"[ERROR] TOKENIZER_PATH not found: {TOKENIZER_PATH}")

    # ✅ Stable tokenizer (no tiktoken)
    try:
        tokenizer = DebertaV2Tokenizer.from_pretrained(TOKENIZER_PATH)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to load tokenizer: {e}")

    # ✅ Force safetensors
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            use_safetensors=True
        )
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to load model: {e}")

    print("[INFO] Trained model successfully loaded.")
    return True