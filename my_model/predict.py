import torch
from my_model.model_manager import model, tokenizer
from my_model.config import DEVICE, MAX_LEN

def predict_email(email_text):
    if model is None or tokenizer is None:
        raise ValueError("Model/tokenizer not loaded. Run load_trained_model() first.")

    inputs = tokenizer(email_text, return_tensors='pt', truncation=True, padding=True, max_length=MAX_LEN).to(DEVICE)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits.detach().cpu(), dim=-1).numpy()[0]

    return ("phishing" if probs[1] > probs[0] else "legitimate", float(probs[1]), float(probs[0]))
