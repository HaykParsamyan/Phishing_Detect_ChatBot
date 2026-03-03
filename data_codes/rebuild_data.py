from pathlib import Path
import pandas as pd

# ================= CONFIG =================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

URL_DATA_PATH = PROJECT_DIR / "final_data" / "url_plus_email_148k_balanced.csv"
EMAIL_DATA_PATH = PROJECT_DIR / "cleaned_data" / "dataset_tragmanutyun_cleaned.csv"

OUTPUT_PATH = PROJECT_DIR / "final_data" / "url_plus_email_148k_balanced.csv"

LABEL_COL = "label"
TEXT_COL = "body"
URL_TAG = "[URL]"

N_PER_CLASS = 1241
RANDOM_SEED = 1
# ==========================================

def main():
    # ---------- Load URL dataset ----------
    if not URL_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing URL dataset: {URL_DATA_PATH}")
    url_df = pd.read_csv(URL_DATA_PATH)

    # ---------- Load Email dataset ----------
    if not EMAIL_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing email dataset: {EMAIL_DATA_PATH}")
    email_df = pd.read_csv(EMAIL_DATA_PATH)

    # ---------- Validate columns ----------
    for df, name in [(url_df, "URL dataset"), (email_df, "Email dataset")]:
        if LABEL_COL not in df.columns or TEXT_COL not in df.columns:
            raise ValueError(f"{name} columns: {list(df.columns)} (need '{LABEL_COL}', '{TEXT_COL}')")

    # ---------- Clean labels to int 0/1 ----------
    def clean_labels(df):
        df = df.copy()
        df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
        df = df.dropna(subset=[LABEL_COL])
        df = df[df[LABEL_COL].isin([0, 1])]
        df[LABEL_COL] = df[LABEL_COL].astype(int)
        return df

    url_df = clean_labels(url_df)
    email_df = clean_labels(email_df)

    # ---------- Keep ONLY emails (no [URL] tag) ----------
    email_only = email_df[~email_df[TEXT_COL].astype(str).str.contains(URL_TAG, na=False)].copy()

    legit_emails = email_only[email_only[LABEL_COL] == 0]
    phish_emails = email_only[email_only[LABEL_COL] == 1]

    print("===== AVAILABLE EMAILS (after filtering) =====")
    print("Legit emails:", len(legit_emails))
    print("Phish emails:", len(phish_emails))

    if len(legit_emails) < N_PER_CLASS or len(phish_emails) < N_PER_CLASS:
        raise ValueError(
            f"Not enough emails to sample {N_PER_CLASS} per class.\n"
            f"Have legit={len(legit_emails)}, phish={len(phish_emails)}"
        )

    # ---------- Sample 37k + 37k ----------
    legit_sample = legit_emails.sample(n=N_PER_CLASS, random_state=RANDOM_SEED)
    phish_sample = phish_emails.sample(n=N_PER_CLASS, random_state=RANDOM_SEED)

    sampled_emails = pd.concat([legit_sample, phish_sample], ignore_index=True)

    # ---------- Merge with URL dataset ----------
    merged = pd.concat([url_df, sampled_emails], ignore_index=True)
    merged = merged.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # ---------- Print final stats ----------
    total = len(merged)
    counts = merged[LABEL_COL].value_counts().sort_index()

    print("\n===== FINAL DATASET =====")
    print("Total:", total)
    print("Legit (0):", counts.get(0, 0))
    print("Phish (1):", counts.get(1, 0))

    # Optional: show type breakdown if you want
    merged["data_type"] = merged[TEXT_COL].apply(lambda x: "url" if URL_TAG in str(x) else "email")
    print("\n===== TYPE BREAKDOWN =====")
    print(merged.groupby([LABEL_COL, "data_type"]).size())
    merged = merged.drop(columns=["data_type"])

    # ---------- Save ----------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)
    print("\nSaved to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()