import torch
from torch import nn
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import json
import os

# ============================================================
# ⚙️ Configuration
# ============================================================
embedding_model_path = "output/fine_tuned_20251030_114220"  # your fine-tuned sentence transformer
classifier_path = "output/classifier_model.pt"               # trained classifier
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

# ============================================================
# 🧩 Define Classifier Architecture (same as training)
# ============================================================
class SMSClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ============================================================
# 📦 Load Saved Classifier & Metadata
# ============================================================
checkpoint = torch.load(classifier_path, map_location=device)

labels_map = checkpoint["labels_map"]
input_dim = checkpoint["input_dim"]
reverse_labels_map = {v: k for k, v in labels_map.items()}

num_classes = len(labels_map)

classifier = SMSClassifier(input_dim, num_classes).to(device)
classifier.load_state_dict(checkpoint["classifier_state_dict"], strict=True)
classifier.eval()

print(f"✅ Classifier loaded successfully with {num_classes} classes.")
print("📚 Label Map:", json.dumps(labels_map, indent=2))

# ============================================================
# 🧠 Load Sentence Transformer Model
# ============================================================
embedding_model = SentenceTransformer(embedding_model_path, device=device)
print("✅ Loaded fine-tuned embedding model")

# ============================================================
# 🔮 Prediction Function
# ============================================================
def predict_sms(text):
    # Encode text into embedding
    embedding = embedding_model.encode([text], convert_to_tensor=True).to(device)
    
    # Forward pass through classifier
    with torch.no_grad():
        logits = classifier(embedding)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    # Top-3 predictions
    top_indices = probs.argsort()[-3:][::-1]
    top_classes = [(reverse_labels_map[i], float(probs[i])) for i in top_indices]

    # Print details
    print("\n👉 SMS:", text)
    print("🔍 Raw logits:", logits.cpu().numpy())
    print("📈 Probabilities:", probs)
    print("🏷️ Top Predictions:")
    for i, (label, p) in enumerate(top_classes):
        print(f"   {i+1}. {label} — {p:.3f}")

    best_label = top_classes[0][0]
    print(f"🎯 Predicted Category: {best_label}")
    return best_label

# ============================================================
# 💬 Interactive Prediction Loop
# ============================================================
if __name__ == "__main__":
    print("\n💡 Type your SMS or message below. Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("📩 Enter SMS: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Exiting prediction mode.")
            break
        if not user_input:
            print("⚠️ Please enter a valid message.")
            continue

        predict_sms(user_input)
