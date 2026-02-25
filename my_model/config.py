import torch

# Choose one:
# MODEL_NAME = "microsoft/deberta-v3-small"
MODEL_NAME = "microsoft/deberta-v3-base"

MODEL_PATH = "models/deberta_v3_phishing"
TOKENIZER_PATH = "models/deberta_v3_phishing_tokenizer"
DATA_PATH = "final_data/all_phishing_master_dataset.csv"

MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")