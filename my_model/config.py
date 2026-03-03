from pathlib import Path
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_NAME = "microsoft/deberta-v3-base"

MODEL_PATH = str(PROJECT_DIR / "models" / "deberta_v3_phishing")
TOKENIZER_PATH = str(PROJECT_DIR / "models" / "deberta_v3_phishing_tokenizer")

DATA_PATH = str(PROJECT_DIR / "final_data" / "url_plus_email_148k_cleaned.csv")

# ✅ add these back
DATA_MODE = "mixed"   # change to "url_only" or "email_only" if needed
URL_TAG = "[URL]"

MAX_LEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")