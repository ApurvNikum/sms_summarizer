import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os
import json

# ---- CONFIG ----
CSV_PATH = "annotated_data_set.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_DIR = "fine_tuned_sms_classifier"
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
# ----------------

# 1️⃣ Load dataset
df = pd.read_csv(CSV_PATH)
df = df[['body_cleaned', 'annotation_category']].dropna()

print(f"✅ Loaded {len(df)} samples from {CSV_PATH}")

# 2️⃣ Encode labels
labels = sorted(df['annotation_category'].unique())
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}
df['label_id'] = df['annotation_category'].map(label2id)
num_labels = len(labels)

print(f"🧾 Classes: {label2id}")

# 3️⃣ Prepare data for training
train_samples = [
    InputExample(texts=[text], label=int(label))
    for text, label in zip(df['body_cleaned'], df['label_id'])
]

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)

# 4️⃣ Load base model
base_model = SentenceTransformer(MODEL_NAME)
embedding_dim = base_model.get_sentence_embedding_dimension()

# 5️⃣ Define classifier head (custom dense layer)
class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )

    def forward(self, features):
        x = features["sentence_embedding"]
        logits = self.classifier(x)
        return {"logits": logits}

# Attach custom head
classifier = ClassificationHead(embedding_dim, num_labels)
base_model.add_module("classification_head", classifier)

# 6️⃣ Define training loss (CrossEntropy)
class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, features, labels):
        logits = features["logits"]
        return self.loss_fct(logits, labels.long())

loss = ClassificationLoss()

# 7️⃣ Train model
base_model.fit(
    train_objectives=[(train_dataloader, loss)],
    epochs=EPOCHS,
    warmup_steps=int(0.1 * len(train_dataloader)),
    output_path=OUTPUT_DIR,
    optimizer_params={'lr': LR}
)

# 8️⃣ Save label mapping
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "label_mapping.json"), "w") as f:
    json.dump(label2id, f, indent=2)

print(f"🎯 Fine-tuned model saved to: {OUTPUT_DIR}")
print(f"📁 Labels mapping saved: {label2id}")
