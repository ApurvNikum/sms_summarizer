import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import pandas as pd
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from datetime import datetime

# ---------- CONFIG ----------
csv_path = "annotated_dataset_v5.csv"
fine_tuned_model_dir = "output/fine_tuned_20251102_010016"   # update to your fine-tuned folder
output_dir = "output"
batch_size = 64
epochs = 8
lr = 2e-4
# ----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("✅ Using device:", device)

# ---------- Load data ----------
df = pd.read_csv(csv_path)
text_col = (
    "body_lower"
    if "body_lower" in df.columns
    else "body_cleaned"
    if "body_cleaned" in df.columns
    else "message"
)
df = df[[text_col, "annotation_category"]].dropna()
df[text_col] = df[text_col].astype(str)

labels_map = {label: idx for idx, label in enumerate(sorted(df["annotation_category"].unique()))}
df["label_id"] = df["annotation_category"].map(labels_map)

print(f"✅ Loaded {len(df)} samples")
print(f"✅ Labels map ({len(labels_map)} classes): {labels_map}")

# ---------- Load fine-tuned encoder ----------
encoder = SentenceTransformer(fine_tuned_model_dir, device=device)
print("✅ Encoder loaded from:", fine_tuned_model_dir)

# ---------- Generate embeddings ----------
print("Encoding texts to embeddings...")
embeddings = encoder.encode(df[text_col].tolist(), convert_to_tensor=True, show_progress_bar=True)
embeddings = embeddings.detach().clone().to(device)
labels = torch.tensor(df["label_id"].tolist(), dtype=torch.long).to(device)

print(f"✅ Embedding shape: {embeddings.shape}")
print(f"✅ First 5 embedding values: {embeddings[0][:5].tolist()}")
print(f"✅ Mean embedding value: {embeddings.mean().item():.6f}")

# ---------- Balanced sampler ----------
label_counts = Counter(df["label_id"].tolist())
num_classes = len(label_counts)
num_samples = len(df)
class_weights = [num_samples / (num_classes * label_counts[i]) for i in range(num_classes)]
print("✅ Class weights:", class_weights)

samples_weight = [class_weights[int(l.item())] for l in labels]
sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)
dataloader = DataLoader(TensorDataset(embeddings, labels), batch_size=batch_size, sampler=sampler)

# ---------- Classifier definition ----------
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


input_dim = embeddings.shape[1]
num_classes = len(labels_map)
classifier = SMSClassifier(input_dim, num_classes).to(device)

print(f"✅ Input dim: {input_dim}, Num classes: {num_classes}")

# ---------- Training setup ----------
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-5)

# ---------- Training loop ----------
print("🚀 Training classifier...")
for epoch in range(epochs):
    classifier.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        batch_x = batch_x.detach().clone()
        logits = classifier(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs} → Loss: {total_loss/len(dataloader):.4f} | Acc: {acc:.2f}%")

print("✅ Training complete.")

# ---------- Save checkpoint ----------
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, f"classifier_model_{ts}.pt")

torch.save(
    {
        "classifier_state_dict": classifier.state_dict(),
        "labels_map": labels_map,
        "input_dim": input_dim,
        "fine_tuned_model_path": fine_tuned_model_dir,
    },
    out_path,
)

print(f"✅ Classifier saved to: {out_path}")
