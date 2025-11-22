import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from my_model.config import MODEL_PATH, TOKENIZER_PATH, DEVICE

model = None
tokenizer = None
training_in_progress = False

def load_trained_model():
    global model, tokenizer
    if os.path.exists(MODEL_PATH):
        tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_PATH)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(DEVICE)
        return True
    return False
