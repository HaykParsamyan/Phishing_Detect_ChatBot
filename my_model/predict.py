import torch
from my_model import model_manager
from my_model.config import DEVICE, MAX_LEN

def predict_email(email_text: str):
    if model_manager.model is None or model_manager.tokenizer is None:
        raise ValueError("Model/tokenizer not loaded. Run load_trained_model() first.")

    inputs = model_manager.tokenizer(
        email_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    model_manager.model.eval()
    with torch.no_grad():
        outputs = model_manager.model(**inputs)
        probs = torch.softmax(outputs.logits.detach(), dim=-1)[0].detach().cpu().numpy()

    # Mapping: 0=legitimate, 1=phishing
    predicted_label = "phishing" if probs[1] > probs[0] else "legitimate"
    return predicted_label, float(probs[1]), float(probs[0])