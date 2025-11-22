import pandas as pd
from my_model.config import DATA_PATH

def load_and_prepare_dataset(sample_frac=1.0):
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Fill missing
    df['subject'] = df['subject'].fillna('')
    df['body'] = df['body'].fillna('')

    # Combine subject + body
    df['email_text'] = df['subject'] + ' ' + df['body']

    # Keep only email_text and label
    df = df[['email_text', 'label']]

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)

    df = df.reset_index(drop=True)
    return df
