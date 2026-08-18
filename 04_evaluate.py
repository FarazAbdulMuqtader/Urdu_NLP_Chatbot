import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ── 1. Load cleaned data (same split logic as training) ───
df = pd.read_csv('roman_urdu_clean.csv')
label_map = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
df['label_id'] = df['label'].map(label_map)
df = df.dropna(subset=['label_id'])
df['label_id'] = df['label_id'].astype(int)

df = df.sample(3000, random_state=42)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ── 2. Load the SAVED model + tokenizer ────────────────────
model_path = './trained_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

print(f"Evaluating on: {device}")

# ── 3. Dataset class (same as training) ────────────────────
class UrduSentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts  = texts.tolist()
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        return {
            'input_ids'      : encoding['input_ids'].squeeze(),
            'attention_mask' : encoding['attention_mask'].squeeze(),
            'label'          : torch.tensor(self.labels[idx], dtype=torch.long)
        }

test_dataset = UrduSentimentDataset(test_df['clean_message'], test_df['label_id'])
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ── 4. Run predictions + keep the ORIGINAL TEXT alongside ──
all_predictions = []
all_true_labels = []
all_texts       = test_df['clean_message'].tolist()

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_true_labels.extend(labels.cpu().numpy())

# ── 5. Classification report (same as before, for reference)
label_names = ['Positive', 'Negative', 'Neutral']
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(all_true_labels, all_predictions, target_names=label_names))

# ── 6. Confusion matrix ─────────────────────────────────────
print("\n=== CONFUSION MATRIX ===")
print("Rows = actual label, Columns = predicted label")
cm = confusion_matrix(all_true_labels, all_predictions)
print(pd.DataFrame(cm, index=label_names, columns=label_names))

# ── 7. Pull actual misclassified examples ──────────────────
results_df = pd.DataFrame({
    'text'      : all_texts,
    'true_label': [label_names[i] for i in all_true_labels],
    'pred_label': [label_names[i] for i in all_predictions]
})

misclassified = results_df[results_df['true_label'] != results_df['pred_label']]

print(f"\n=== MISCLASSIFIED EXAMPLES ({len(misclassified)} total) ===")
print(misclassified.sample(min(10, len(misclassified)), random_state=42).to_string(index=False))

# ── 8. Save full results for later inspection ──────────────
results_df.to_csv('evaluation_results.csv', index=False)
print("\n✅ Full results saved to evaluation_results.csv")
