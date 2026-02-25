import pandas as pd
from my_model.config import DATA_PATH

def load_and_prepare_dataset(sample_frac: float = 1.0) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # --- Validate required columns ---
    required_cols = {"subject", "body", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError("Dataset must contain columns: subject, body, label")

    # --- Clean text fields ---
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)

    # --- Combine subject + body ---
    df["email_text"] = (df["subject"] + " " + df["body"]).str.strip()

    # --- Keep only needed columns ---
    df = df[["email_text", "label"]].copy()

    # --- Clean labels safely ---
    # Convert anything non-numeric to NaN
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    # Drop rows with missing labels
    df = df.dropna(subset=["label"])

    # Convert to int
    df["label"] = df["label"].astype(int)

    # Keep only binary labels (0 and 1)
    df = df[df["label"].isin([0, 1])]

    # --- Optional: Debug distribution ---
    print("Label distribution:\n", df["label"].value_counts())

    # --- Optional sampling (smoke test mode) ---
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)

    df = df.reset_index(drop=True)

    # Safety check
    if df.empty:
        raise ValueError("Dataset is empty after cleaning. Check label column.")

    return df