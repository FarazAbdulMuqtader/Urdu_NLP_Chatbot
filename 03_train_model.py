import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.optim import AdamW

# ── 1. Load CSV ───────────────────────────────────────────
df = pd.read_csv('roman_urdu_clean.csv')

# ── 2. Encode labels ──────────────────────────────────────
label_map = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
df['label_id'] = df['label'].map(label_map)
df = df.dropna(subset=['label_id'])
df['label_id'] = df['label_id'].astype(int)

# ── 3. Use small sample for today ─────────────────────────
# 3000 rows only — just to verify training works
# We'll do full training after this runs clean
df = df.sample(3000, random_state=42)
print(f"Using {len(df)} rows for test run")

# ── 4. Split 80/20 ────────────────────────────────────────
train_df, test_df = train_test_split(df,test_size=0.2,random_state=42)
print(f"Train: {len(train_df)} | Test: {len(test_df)}")

# ── 5. Load tokenizer ─────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

# ── 6. Dataset class ──────────────────────────────────────
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

# ── 7. Create datasets ────────────────────────────────────
train_dataset = UrduSentimentDataset(train_df['clean_message'], train_df['label_id'])
test_dataset  = UrduSentimentDataset(test_df['clean_message'],  test_df['label_id'])

# ── 8. Create DataLoaders ─────────────────────────────────
# batch_size=16 means feed 16 samples at a time to the model
# shuffle=True means mix up the training data each epoch
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Test batches : {len(test_loader)}")

# ── 9. Load the model ─────────────────────────────────────
# AutoModelForSequenceClassification loads xlm-roberta
# with a classification head on top — 3 outputs for our 3 labels
# num_labels=3 tells it we have 3 categories
print("\nLoading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=3
)

# ── 10. Set up optimizer ──────────────────────────────────
# AdamW is the standard optimizer for transformers
# lr is learning rate — 2e-5 is the standard for fine-tuning
optimizer = AdamW(model.parameters(), lr=2e-5)

# ── 11. Detect CPU or GPU ─────────────────────────────────
# If you have a GPU it uses that, otherwise falls back to CPU
# You likely have CPU only — that's fine for this test run
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

# Move model to the device
model = model.to(device)

# ── 12. Training loop ─────────────────────────────────────
# We train for 2 epochs on this small run
EPOCHS = 2

print("\n=== TRAINING STARTED ===\n")

for epoch in range(EPOCHS):

    # --- Training phase ---
    # model.train() tells the model it's in training mode
    model.train()
    total_loss = 0

    for batch_num, batch in enumerate(train_loader):

        # Move batch data to same device as model
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        # optimizer.zero_grad() clears old gradients
        # If you don't do this they accumulate and corrupt training
        optimizer.zero_grad()

        # Forward pass — model makes predictions
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        # outputs.loss is how wrong the model was
        loss = outputs.loss

        # backward() calculates how to adjust the model
        loss.backward()

        # optimizer.step() actually applies the adjustments
        optimizer.step()

        total_loss += loss.item()

        # Print progress every 50 batches
        if (batch_num + 1) % 50 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_num+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch+1} complete | Average Loss: {avg_loss:.4f}\n")

print("=== TRAINING COMPLETE ===")
print("\nNext step: evaluate accuracy on test set")
# ── 13. Evaluation ────────────────────────────────────────
from sklearn.metrics import accuracy_score, classification_report

print("\n=== EVALUATING ON TEST SET ===\n")

# model.eval() switches off dropout — consistent predictions
model.eval()

all_predictions = []
all_true_labels = []

# torch.no_grad() tells PyTorch don't calculate gradients
# saves memory since we're just predicting, not training
with torch.no_grad():

    for batch in test_loader:

        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        # Forward pass — get predictions
        # No labels passed this time — just predicting
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # outputs.logits are the raw scores for each class
        # shape: [batch_size, 3] — 3 scores per sample
        logits = outputs.logits

        # argmax picks the index of the highest score
        # dim=1 means pick across the 3 label columns
        predictions = torch.argmax(logits, dim=1)

        # Move to CPU and convert to list for sklearn
        all_predictions.extend(predictions.cpu().numpy())
        all_true_labels.extend(labels.cpu().numpy())

# ── 14. Print results ─────────────────────────────────────
accuracy = accuracy_score(all_true_labels, all_predictions)
print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")

# classification_report shows accuracy per label
# so you can see which sentiment the model handles best
label_names = ['Positive', 'Negative', 'Neutral']
print("Per-label breakdown:")
print(classification_report(
    all_true_labels,
    all_predictions,
    target_names=label_names
))
# ── Save the trained model ────────────────────────────────
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')
print("\n✅ Model saved to ./trained_model")
