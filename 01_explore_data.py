from datasets import load_dataset
import pandas as pd

# Load dataset
dataset = load_dataset("Khubaib01/RomanUrdu-NLP-Sentiment-Corpus")

# Convert to pandas
df = pd.DataFrame(dataset['train'])

print("=== BASIC INFO ===")
print(f"Total samples: {len(df)}")
print(f"\nColumn names: {df.columns.tolist()}")

print("\n=== LABEL DISTRIBUTION ===")
print(df['label'].value_counts())

print("\n=== SAMPLE TEXTS PER LABEL ===")
for label in df['label'].unique():
    sample = df[df['label'] == label]['message'].iloc[0]
    print(f"\n{label}: {sample}")

print("\n=== TEXT LENGTH STATS ===")
print(f"Average word count: {df['word_length'].mean():.1f}")
print(f"Shortest message: {df['word_length'].min():.0f} words")
print(f"Longest message: {df['word_length'].max():.0f} words")

print("\n=== FIRST 5 ROWS ===")
print(df[['message', 'label']].head())