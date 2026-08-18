import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── 1. Load the saved model + tokenizer ────────────────────
model_path = './trained_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

print(f"Model loaded. Running on: {device}\n")

# ── 2. Label map (must match training order exactly) ───────
label_names = ['Positive', 'Negative', 'Neutral']

# ── 3. Prediction function ──────────────────────────────────
def predict_sentiment(text):
    encoding = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    input_ids      = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=1).squeeze()
    predicted_id = torch.argmax(probs).item()
    predicted_label = label_names[predicted_id]
    confidence = probs[predicted_id].item()

    return predicted_label, confidence, probs

# ── 4. Test on fixed example sentences ──────────────────────
test_sentences = [
    "ye movie bohut acha tha",
    "mujhe bilkul pasand nahi aya",
    "kal market gaya tha",
    "bohut bura din tha aj",
    "chalo choro is baat ko",
    "wah kya baat hai zabardast",
]

print("=== Roman Urdu Sentiment Demo ===\n")

for sentence in test_sentences:
    label, confidence, probs = predict_sentiment(sentence)
    print(f"Text       : {sentence}")
    print(f"Prediction : {label} ({confidence*100:.1f}% confidence)")
    print(f"All scores : " + ", ".join(f"{n}: {p.item()*100:.1f}%" for n, p in zip(label_names, probs)))
    print()    predicted_label = label_names[predicted_id]
    confidence = probs[predicted_id].item()

    return predicted_label, confidence, probs

# ── 4. Interactive loop ──────────────────────────────────────
print("=== Roman Urdu Sentiment Demo ===")
print("Type a sentence to check its sentiment. Type 'quit' to exit.\n")

while True:
    user_input = input("Enter text: ")

    if user_input.lower() == 'quit':
        print("Exiting demo.")
        break

    if not user_input.strip():
        print("Please enter some text.\n")
        continue

    label, confidence, probs = predict_sentiment(user_input)

    print(f"\nPrediction : {label}")
    print(f"Confidence : {confidence*100:.1f}%")
    print("All scores :")
    for name, p in zip(label_names, probs):
        print(f"  {name}: {p.item()*100:.1f}%")
    print()
