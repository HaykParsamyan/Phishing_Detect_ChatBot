import pandas as pd
from my_model.config import DATA_PATH, DATA_MODE, URL_TAG

def load_and_prepare_dataset(sample_frac: float = 1.0, seed: int = 42) -> pd.DataFrame:
    print("===== DATA LOADER =====")
    print("DATA_PATH:", DATA_PATH)
    print("DATA_MODE:", DATA_MODE)

    df = pd.read_csv(DATA_PATH, low_memory=False)
    print("RAW rows:", len(df))

    if "label" not in df.columns:
        raise ValueError("Dataset must contain column: label")

    print("\nRAW label counts (incl NaN):")
    print(df["label"].value_counts(dropna=False).head(20))

    # ---------- Build unified text field ----------
    if "email_text" in df.columns:
        df["email_text"] = df["email_text"].fillna("").astype(str)
    else:
        if "subject" not in df.columns or "body" not in df.columns:
            raise ValueError("Dataset must contain either email_text OR (subject and body)")
        df["subject"] = df["subject"].fillna("").astype(str)
        df["body"] = df["body"].fillna("").astype(str)
        df["email_text"] = (df["subject"] + " " + df["body"]).str.strip()

    # Keep only needed columns
    df = df[["email_text", "label"]].copy()

    # ---------- Clean labels ----------
    before = len(df)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].replace({"phishing": "1", "legitimate": "0"})
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    dropped_bad_label = before - len(df)

    # Keep only binary labels BEFORE int cast
    before = len(df)
    df = df[df["label"].isin([0, 1, 0.0, 1.0])].copy()
    dropped_nonbinary = before - len(df)

    df["label"] = df["label"].astype(int)

    # ---------- Clean text ----------
    before = len(df)
    df["email_text"] = df["email_text"].fillna("").astype(str).str.strip()
    df = df[df["email_text"].str.len() > 0].copy()
    dropped_empty_text = before - len(df)

    # ---------- Filter by mode ----------
    before = len(df)
    if DATA_MODE == "url_only":
        df = df[df["email_text"].str.contains(URL_TAG, na=False)].copy()
    elif DATA_MODE == "email_only":
        df = df[~df["email_text"].str.contains(URL_TAG, na=False)].copy()
    elif DATA_MODE != "mixed":
        raise ValueError(f"Unknown DATA_MODE: {DATA_MODE}")
    dropped_by_mode = before - len(df)

    # ---------- Optional sampling (STRATIFIED) ----------
    if sample_frac < 1.0:
        before_s = len(df)
        parts = []
        for label_val in [0, 1]:
            part = df[df["label"] == label_val]
            if len(part) == 0:
                continue
            parts.append(part.sample(frac=sample_frac, random_state=seed))
        df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
        print(f"\nSampled frac={sample_frac} -> {before_s} -> {len(df)} rows")

    print("\n===== DROPS SUMMARY =====")
    print("Dropped bad label (NaN/invalid):", dropped_bad_label)
    print("Dropped non-binary labels:", dropped_nonbinary)
    print("Dropped empty text:", dropped_empty_text)
    print("Dropped by DATA_MODE filter:", dropped_by_mode)

    print("\n===== FINAL DATA =====")
    print("Final rows:", len(df))
    print("Final label distribution:")
    print(df["label"].value_counts())

    return df