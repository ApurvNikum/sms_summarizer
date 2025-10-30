import os
import torch
import pandas as pd
from torch import nn
from datetime import datetime
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader

# ==========================================================
# ✅ Configuration
# ==========================================================
csv_path = "annotated_data_set.csv"
base_model_path = "sentence-transformers/all-MiniLM-L6-v2"  # directly from Hugging Face
batch_size = 16
epochs = 3

# ==========================================================
# ✅ Device setup
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device.upper()}")

# ==========================================================
# ✅ Load dataset
# ==========================================================
df = pd.read_csv(csv_path)
df = df[['body_cleaned', 'annotation_category']].dropna()

# Map categories to numeric labels
label_mapping = {label: idx for idx, label in enumerate(sorted(df['annotation_category'].unique()))}
df['label_id'] = df['annotation_category'].map(label_mapping)
print(f"✅ Loaded {len(df)} samples from {csv_path}")
print(f"🧾 Classes: {label_mapping}")

# Convert to training examples
train_samples = [
    InputExample(texts=[row['body_cleaned']], label=int(row['label_id']))
    for _, row in df.iterrows()
]
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)

# ==========================================================
# ✅ Load base model
# ==========================================================
model = SentenceTransformer(base_model_path, device=device)

# ==========================================================
# ✅ Custom classification loss
# ==========================================================
class CustomClassificationLoss(nn.Module):
    def __init__(self, model, num_classes):
        super(CustomClassificationLoss, self).__init__()
        self.model = model
        self.classifier = nn.Linear(model.get_sentence_embedding_dimension(), num_classes)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, features, labels):
        # features is a list of dicts
        embeddings = self.model(features[0])["sentence_embedding"]
        logits = self.classifier(embeddings)
        return self.loss_fct(logits, labels)

num_classes = len(label_mapping)
train_loss = CustomClassificationLoss(model, num_classes=num_classes)

# ==========================================================
# ✅ Output directory
# ==========================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("output", f"fine_tuned_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# ==========================================================
# ✅ Training
# ==========================================================
print("🚀 Starting training...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=epochs,
    output_path=output_dir,
    show_progress_bar=True
)
print(f"✅ Training complete! Model saved to: {output_dir}")
