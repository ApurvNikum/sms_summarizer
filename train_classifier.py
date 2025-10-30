import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm
from collections import Counter
import numpy as np

# ============================================================
# ⚙️ Configuration
# ============================================================
base_model_path = "output/fine_tuned_20251030_114220"  # your fine-tuned embedding model
csv_path = "annotated_data_set.csv"
output_path = "output/classifier_model.pt"

batch_size = 32
epochs = 8
learning_rate = 2e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

# ============================================================
# 📂 Load Dataset
# ============================================================
df = pd.read_csv(csv_path)

# Replace these column names with your actual dataset columns
df = df[['body_cleaned', 'annotation_category']].dropna()

# Map categories to numeric labels
labels_map = {label: idx for idx, label in enumerate(sorted(df['annotation_category'].unique()))}
df['label_id'] = df['annotation_category'].map(labels_map)
print(f"✅ Loaded {len(df)} samples")
print("🧾 Classes:", labels_map)

texts = df['body_cleaned'].tolist()
labels = df['label_id'].tolist()

# ============================================================
# 🧠 Load Fine-Tuned SentenceTransformer
# ============================================================
model = SentenceTransformer(base_model_path, device=device)

# Compute embeddings
print("🔍 Encoding text samples...")
embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)

# ============================================================
# ⚖️ Compute Class Weights (for imbalance correction)
# ============================================================
label_counts = Counter(labels)
num_samples = len(labels)
num_classes = len(label_counts)
weights = [num_samples / (num_classes * label_counts[i]) for i in range(num_classes)]
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

print("\n📊 Class Distribution:")
for label, count in label_counts.items():
    print(f"  {list(labels_map.keys())[list(labels_map.values()).index(label)]}: {count} samples")

print("\n⚖️ Class Weights:")
for i, w in enumerate(weights):
    print(f"  {list(labels_map.keys())[list(labels_map.values()).index(i)]}: {w:.3f}")

# ============================================================
# 📦 Prepare DataLoader with Weighted Sampling
# ============================================================
dataset = TensorDataset(embeddings, torch.tensor(labels, dtype=torch.long))
samples_weight = [weights[label] for label in labels]
sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

# ============================================================
# 🧩 Define Neural Classifier (3-layer MLP)
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

# Initialize classifier
input_dim = embeddings.shape[1]
num_classes = len(labels_map)
classifier = SMSClassifier(input_dim, num_classes).to(device)

# ============================================================
# ⚡ Training Setup
# ============================================================
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate, weight_decay=1e-5)

# ============================================================
# 🚀 Training Loop
# ============================================================
print("\n🚀 Starting classifier training with class balancing...")
for epoch in range(epochs):
    classifier.train()
    total_loss = 0.0
    for batch_x, batch_y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        logits = classifier(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"✅ Epoch {epoch+1}/{epochs} — Balanced Loss: {avg_loss:.4f}")

# ============================================================
# 💾 Save Model and Label Map
# ============================================================
os.makedirs(os.path.dirname(output_path), exist_ok=True)
torch.save({
    "classifier_state_dict": classifier.state_dict(),
    "labels_map": labels_map,
    "input_dim": input_dim
}, output_path)

print(f"\n🎯 Classifier saved successfully to: {output_path}")
