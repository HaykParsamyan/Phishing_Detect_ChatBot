from my_model import model_manager
import torch
from my_model.config import DEVICE, MAX_LEN

def predict_email(email_text):
    if model_manager.model is None or model_manager.tokenizer is None:
        raise ValueError("models/tokenizer not loaded. Run load_trained_model() first.")

    inputs = model_manager.tokenizer(
        email_text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    ).to(DEVICE)

    model_manager.model.eval()
    with torch.no_grad():
        outputs = model_manager.model(**inputs)
        probs = torch.softmax(outputs.logits.detach().cpu(), dim=-1)[0].numpy()

    return ("phishing" if probs[1] > probs[0] else "legitimate", float(probs[1]), float(probs[0]))
