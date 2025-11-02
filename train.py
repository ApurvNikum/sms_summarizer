# train_finetune.py
import os
import shutil
import torch
import pandas as pd
from torch import nn
from datetime import datetime
from sentence_transformers import SentenceTransformer, InputExample
from torch.utils.data import DataLoader

# ---------- CONFIG ----------
csv_path = "annotated_dataset_v5.csv"
base_model_name = "sentence-transformers/all-MiniLM-L6-v2"
batch_size = 16
epochs = 3
output_parent = "output"
# ----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# helper: ensure SBERT folder has a 0_Transformer subfolder (safetensors/pytorch bin handling)
def ensure_sbert_structure(model_dir):
    """
    If model.safetensors (or tokenizer files) are at model_dir root (not inside 0_Transformer),
    move the model files into a new 0_Transformer folder so SentenceTransformer can load properly.
    """
    files_at_root = os.listdir(model_dir)
    need_to_move = False
    model_files = ["model.safetensors", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json", "vocab.txt"]
    for f in model_files:
        if f in files_at_root:
            need_to_move = True
            break

    if not need_to_move:
        return

    target = os.path.join(model_dir, "0_Transformer")
    os.makedirs(target, exist_ok=True)

    for fname in files_at_root:
        # move only relevant files (weights + tokenizer + config) into 0_Transformer
        if any(fname.startswith(x) for x in ["model.safetensors", "pytorch_model.bin", "tokenizer", "vocab", "config", "sentence_bert"]):
            src = os.path.join(model_dir, fname)
            dst = os.path.join(target, fname)
            if not os.path.exists(dst):
                try:
                    shutil.move(src, dst)
                    print(f"Moved: {src} -> {dst}")
                except Exception as e:
                    print(f"Could not move {src}: {e}")

# Load dataset
df = pd.read_csv(csv_path)
df = df[['body_lower', 'annotation_category']].dropna()

label_mapping = {label: idx for idx, label in enumerate(sorted(df['annotation_category'].unique()))}
df['label_id'] = df['annotation_category'].map(label_mapping)
print(f"Loaded {len(df)} samples, classes: {label_mapping}")

train_examples = [InputExample(texts=[r['body_lower']], label=int(r['label_id'])) for _, r in df.iterrows()]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

# Load base model (from HF)
model = SentenceTransformer(base_model_name, device=device)
print("Loaded base model:", base_model_name)

# Custom classification loss as before (works with model.fit)
class CustomClassificationLoss(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        self.classifier = nn.Linear(model.get_sentence_embedding_dimension(), num_classes)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, features, labels):
        # features is like [batch_texts] because we passed InputExample(texts=[...])
        embeddings = self.model(features[0])["sentence_embedding"]
        logits = self.classifier(embeddings)
        return self.loss_fct(logits, labels)

num_classes = len(label_mapping)
train_loss = CustomClassificationLoss(model, num_classes=num_classes)

# Output folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(output_parent, f"fine_tuned_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

print("Starting finetune...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=epochs,
    output_path=output_dir,
    show_progress_bar=True
)

# After saving, ensure structure is proper (move files if necessary)
ensure_sbert_structure(output_dir)
print("Finetune complete. Model saved to:", output_dir)
