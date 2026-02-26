import torch

# Choose one:
# MODEL_NAME = "microsoft/deberta-v3-small"
MODEL_NAME = "microsoft/deberta-v3-base"

MODEL_PATH = "models/deberta_v3_phishing"
TOKENIZER_PATH = "models/deberta_v3_phishing_tokenizer"
DATA_PATH = "final_data/merged_email_url_dataset_ai.csv"

MAX_LEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")