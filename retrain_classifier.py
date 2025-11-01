import os
import torch
import pandas as pd
from torch import nn, optim
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
import numpy as np

# ==========================================================
# ✅ Config
# ==========================================================
csv_path = "annotated_dataset_v5.csv"
fine_tuned_model_dir = "output/fine_tuned_20251031_162336"  # ← replace if different
batch_size = 16
epochs = 6
learning_rate = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"✅ Using device: {device.upper()}")

# ==========================================================
# ✅ Load Dataset
# ==========================================================
df = pd.read_csv(csv_path)
df = df[['body_lower', 'annotation_category']].dropna()

label_mapping = {label: idx for idx, label in enumerate(sorted(df['annotation_category'].unique()))}
df['label_id'] = df['annotation_category'].map(label_mapping)

print(f"✅ Loaded {len(df)} samples")
print(f"🧾 Label mapping: {label_mapping}")

# ==========================================================
# ✅ Dataset Loader
# ==========================================================
class TextDataset(Dataset):
    def __init__(self, df):
        self.texts = df['body_lower'].tolist()
        self.labels = df['label_id'].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# ==========================================================
# ✅ Load fine-tuned SentenceTransformer
# ==========================================================
encoder = SentenceTransformer(fine_tuned_model_dir, device=device)

# Freeze encoder weights
for param in encoder.parameters():
    param.requires_grad = False

# ==========================================================
# ✅ Compute class weights
# ==========================================================
classes = np.unique(df['label_id'])
weights = compute_class_weight(class_weight='balanced', classes=classes, y=df['label_id'])
weights = torch.tensor(weights, dtype=torch.float).to(device)
print(f"⚖️ Class Weights: {weights}")

# ==========================================================
# ✅ Classifier model
# ==========================================================
class SentenceClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super(SentenceClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(encoder.get_sentence_embedding_dimension(), 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, texts):
        # Run encoding inside the computation graph
        encoded = self.encoder.tokenize(texts)
        encoded = {k: v.to(device) for k, v in encoded.items()}  # ✅ move all to GPU
        features = self.encoder[0](encoded)                      # Transformer layer
        sentence_embeddings = self.encoder[1](features)["sentence_embedding"]
        logits = self.classifier(sentence_embeddings)

        return logits

num_classes = len(label_mapping)
model = SentenceClassifier(encoder, num_classes).to(device)

# ==========================================================
# ✅ Dataloader
# ==========================================================
dataset = TextDataset(df)
dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

# ==========================================================
# ✅ Training Setup
# ==========================================================
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# ==========================================================
# ✅ Training Loop
# ==========================================================
print("🚀 Training classifier with balanced class weights...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct, total = 0, 0

    for texts, labels in dataloader:
        labels = torch.tensor(labels, dtype=torch.long).to(device)
        optimizer.zero_grad()

        logits = model(texts)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(dataloader):.4f} | Acc: {acc:.2f}%")

# ==========================================================
# ✅ Save classifier
# ==========================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join("output", f"classifier_model_{timestamp}.pt")

torch.save({
    'model_state_dict': model.state_dict(),
    'label_mapping': label_mapping
}, output_path)

print(f"🎯 Classifier saved successfully → {output_path}")
print(f"🔗 Trained using fine-tuned model → {fine_tuned_model_dir}")
