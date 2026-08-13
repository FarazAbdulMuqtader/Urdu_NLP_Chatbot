import pandas as pd
import re
from datasets import load_dataset

# ── 1. Load ───────────────────────────────────────────────
dataset = load_dataset("Khubaib01/RomanUrdu-NLP-Sentiment-Corpus")
df = pd.DataFrame(dataset['train'])
original_len=len(df)
print("Label distribution before cleaning: ")
print(df["label"].value_counts())

print(f"Before cleaning: {len(df)} rows")

print(f"Before cleaning: {len(df)} rows")

# ── 2. Clean function ─────────────────────────────────────
def clean_text(text):
    # Handle NaN/empty values
    if not isinstance(text, str):
        return ''
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtag symbol but keep word
    text = re.sub(r'#', '', text)
    # Remove emojis,punctuations
    text = re.sub(r'[^\w\s]', '', text)
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── 3. Apply cleaning ─────────────────────────────────────
df['clean_message'] = df['message'].apply(clean_text)
print(f"Duplicates:{df.duplicated(subset="clean_message").sum()}")
# ── 4. Remove bad rows ────────────────────────────────────
# Remove empty messages
df = df[df['clean_message'].str.len() > 0]

# Remove messages under 2 words (too short to carry sentiment)
df = df[df['clean_message'].str.split().str.len() >= 2]

# Remove extreme outliers (over 100 words)
df = df[df['clean_message'].str.split().str.len() <= 100]

print(f"After cleaning: {len(df)} rows")
print(f"Removed: {original_len - len(df)} rows")

print("Label distribution after cleaning: ")
print(df["label"].value_counts())

# ── 5. Preview ────────────────────────────────────────────
print("\n=== BEFORE vs AFTER ===")
for i in range(3):
    print(f"\nOriginal : {df['message'].iloc[i]}")
    print(f"Cleaned  : {df['clean_message'].iloc[i]}")
    print(f"Label    : {df['label'].iloc[i]}")

# ── 6. Save ───────────────────────────────────────────────
df[['clean_message', 'label']].to_csv('roman_urdu_clean.csv', index=False)
print("\n✅ Saved to roman_urdu_clean.csv")
