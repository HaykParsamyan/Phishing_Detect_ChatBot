import torch
from my_model import model_manager
from my_model.config import DEVICE, MAX_LEN

PHISH_HIGH = 0.90
LEGIT_LOW  = 0.30
MARGIN     = 0.25

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
        probs = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu().numpy()

    legit_prob = float(probs[0])
    phishing_prob = float(probs[1])

    if (phishing_prob >= PHISH_HIGH) and ((phishing_prob - legit_prob) >= MARGIN):
        predicted_label = "phishing"
    elif phishing_prob <= LEGIT_LOW:
        predicted_label = "legitimate"
    else:
        predicted_label = "uncertain"

    return predicted_label, phishing_prob, legit_prob