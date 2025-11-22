import torch

MODEL_PATH = "models/distilbert_phishing"
TOKENIZER_PATH = "models/distilbert_phishing_tokenizer"
DATA_PATH = "final_data/all_phishing_master_dataset.csv"
MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
