import re
from pathlib import Path
import pandas as pd

# ================== CONFIG ==================
LABEL_COL = "label"
TEXT_COL = "body"   # change if your column name is different

TARGET_PHISH = 0.55
TARGET_LEGIT = 0.45

RANDOM_SEED = 42

# Input/Output (script is in data_codes/)
BASE_DIR = Path(__file__).resolve().parent                 # .../data_codes
PROJECT_DIR = BASE_DIR.parent                              # .../Phishing_Detector_AI
INPUT_PATH = PROJECT_DIR / "final_data" / "merged_email_url_dataset_ai.csv" # change filename if needed
OUTPUT_PATH = PROJECT_DIR / "final_data" / "final_data_balanced_55_45.csv"
# ===========================================

url_regex = re.compile(r"(https?://|www\.)", re.IGNORECASE)

def detect_type(x) -> str:
    """Classify row content roughly as url vs email text."""
    if not isinstance(x, str):
        return "unknown"
    s = x.strip()
    if url_regex.search(s):
        return "url"
    return "email"

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_PATH}\n"
            f"Fix INPUT_PATH filename or put your csv into final_data/"
        )

    df = pd.read_csv(INPUT_PATH)

    # Validate columns
    missing = [c for c in [LABEL_COL, TEXT_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Your columns are: {list(df.columns)}\n"
            f"Fix LABEL_COL / TEXT_COL at the top."
        )

    # Keep only labels 0/1
    df = df[df[LABEL_COL].isin([0, 1])].copy()

    # Add type column (url/email)
    df["data_type"] = df[TEXT_COL].apply(detect_type)

    legit = df[df[LABEL_COL] == 0].copy()
    phish = df[df[LABEL_COL] == 1].copy()

    L = len(legit)
    P = len(phish)

    if L == 0 or P == 0:
        raise ValueError(f"Bad dataset: legit={L}, phish={P}. You need both classes.")

    # Keep all legit, compute needed phishing to reach target ratio
    P_target = int(round(L * (TARGET_PHISH / TARGET_LEGIT)))

    if P_target > P:
        raise ValueError(
            f"Not enough phishing to reach target.\nHave phishing={P}, need={P_target}."
        )

    print("===== CURRENT DISTRIBUTION =====")
    print(f"Total: {L+P}")
    print(f"Legit (0): {L} ({L/(L+P)*100:.2f}%)")
    print(f"Phish (1): {P} ({P/(L+P)*100:.2f}%)")

    print("\n===== TARGET =====")
    print(f"Keep all legit: {L}")
    print(f"Keep phishing:  {P_target} (downsample from {P})")

    # Sample phishing proportionally by type to preserve url/email mix
    phish_type_counts = phish["data_type"].value_counts(dropna=False)
    phish_type_props = phish_type_counts / phish_type_counts.sum()

    sampled_parts = []
    for t, prop in phish_type_props.items():
        k = int(round(P_target * prop))
        subset = phish[phish["data_type"] == t]

        # If subset is tiny, just take all, but this shouldn't happen usually
        k = min(k, len(subset))
        if k > 0:
            sampled_parts.append(subset.sample(n=k, random_state=RANDOM_SEED))

    phish_sampled = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else phish.sample(
        n=P_target, random_state=RANDOM_SEED
    )

    # Fix rounding mismatch exactly to P_target
    if len(phish_sampled) > P_target:
        phish_sampled = phish_sampled.sample(n=P_target, random_state=RANDOM_SEED)
    elif len(phish_sampled) < P_target:
        need = P_target - len(phish_sampled)
        # add extra from remaining phishing not already selected (approx)
        extra_pool = phish.copy()
        extra = extra_pool.sample(n=min(need, len(extra_pool)), random_state=RANDOM_SEED)
        phish_sampled = pd.concat([phish_sampled, extra], ignore_index=True)
        # ensure exact
        phish_sampled = phish_sampled.sample(n=P_target, random_state=RANDOM_SEED)

    # Combine and shuffle
    balanced = pd.concat([legit, phish_sampled], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Final stats
    final_counts = balanced[LABEL_COL].value_counts().sort_index()
    total = len(balanced)

    print("\n===== FINAL DISTRIBUTION =====")
    print(f"Total: {total}")
    print(f"Legit (0): {final_counts.get(0,0)} ({final_counts.get(0,0)/total*100:.2f}%)")
    print(f"Phish (1): {final_counts.get(1,0)} ({final_counts.get(1,0)/total*100:.2f}%)")

    print("\n===== TYPE MIX (FINAL) =====")
    print(balanced.groupby([LABEL_COL, "data_type"]).size())

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    balanced.drop(columns=["data_type"]).to_csv(OUTPUT_PATH, index=False)
    print("\nSaved to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()