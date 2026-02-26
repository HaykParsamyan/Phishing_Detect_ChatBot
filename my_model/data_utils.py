import pandas as pd
from my_model.config import DATA_PATH


def load_and_prepare_dataset(sample_frac: float = 1.0) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)

    if "label" not in df.columns:
        raise ValueError("Dataset must contain column: label")

    # Build email_text from whichever columns exist
    if "email_text" in df.columns:
        df["email_text"] = df["email_text"].fillna("").astype(str)
    else:
        # fallback: subject + body
        if "subject" not in df.columns or "body" not in df.columns:
            raise ValueError("Dataset must contain either email_text OR (subject and body)")
        df["subject"] = df["subject"].fillna("").astype(str)
        df["body"] = df["body"].fillna("").astype(str)
        df["email_text"] = (df["subject"] + " " + df["body"]).str.strip()

    # Keep only needed columns
    df = df[["email_text", "label"]].copy()

    # Label cleanup: supports 1/0, 1.0/0.0, "phishing"/"legitimate"
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].replace({"phishing": "1", "legitimate": "0"})

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Clean text
    df["email_text"] = df["email_text"].fillna("").astype(str).str.strip()
    df = df[df["email_text"].str.len() > 0]

    # Optional sampling
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)

    df = df.reset_index(drop=True)

    print("Label distribution:\n", df["label"].value_counts())
    return df