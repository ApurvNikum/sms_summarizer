import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.models import Transformer, Pooling
from torch.utils.data import DataLoader
import torch

# --- Configuration ---
DATA_FILE = 'annotated_data_set.csv'

# Path to your local model folder (downloaded or cached)
MODEL_NAME_PATH = './local_model/all-MiniLM-L6-v2'

OUTPUT_MODEL_PATH = 'sms_classification_model'
NUM_EPOCHS = 4
BATCH_SIZE = 16

def prepare_data_and_train_model():
    """
    Loads data, preprocesses labels, prepares DataLoader, and fine-tunes the S-BERT model.
    """
    print("--- 1. Loading and Filtering Data ---")
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"✅ Data loaded from {DATA_FILE}.")
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Data file '{DATA_FILE}' not found. Aborting.")
        return

    # Filter out non-core categories for training
    CORE_CATEGORIES = [
        'Travel', 'Telecom', 'Retail', 'Banking',
        'Education', 'Transactional/Security', 'Status/Alert'
    ]
    df_train = df[df['annotation_category'].isin(CORE_CATEGORIES)].copy()

    # Ensure all categories have at least 2 samples
    valid_categories = df_train['annotation_category'].value_counts()
    valid_categories = valid_categories[valid_categories >= 2].index
    df_train = df_train[df_train['annotation_category'].isin(valid_categories)]

    print(f"Data loaded. Total messages for training: {len(df_train)}")

    # Combine sender and message text (feature engineering)
    df_train['text_input'] = df_train['sender'].astype(str) + " [SEP] " + df_train['body_cleaned'].astype(str)

    # --- 2. Encode Labels ---
    print("\n--- 2. Encoding Labels ---")
    label_encoder = LabelEncoder()
    df_train['label_id'] = label_encoder.fit_transform(df_train['annotation_category'])

    # --- 3. Split into Train/Validation ---
    print("\n--- 3. Splitting and Preparing Data Loaders ---")
    train_df, val_df = train_test_split(
        df_train,
        test_size=0.1,
        random_state=42,
        stratify=df_train['annotation_category']
    )

    # SoftmaxLoss requires two identical texts for each sample
    train_examples_fixed = [
        InputExample(texts=[row['text_input'], row['text_input']], label=row['label_id'])
        for _, row in train_df.iterrows()
    ]

    train_dataloader_fixed = DataLoader(
        train_examples_fixed,
        shuffle=True,
        batch_size=BATCH_SIZE,
        pin_memory=False
    )

    # --- 4. Model Setup and Fine-Tuning ---
    print("\n--- 4. Model Setup and Fine-Tuning ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # ✅ Proper SBERT-compatible model loading (fixes tokenize error)
        word_embedding_model = Transformer(MODEL_NAME_PATH)

        pooling_model = Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )

        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

        print("✅ Model successfully constructed using Transformer wrapper.")

    except Exception as e:
        print(f"CRITICAL ERROR during model construction: {e}")
        print("Please ensure local model directory contains config.json, tokenizer.json, pytorch_model.bin, etc.")
        return

    # --- 5. Define Loss Function ---
    train_loss = losses.SoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=len(label_encoder.classes_)
    )

    # --- 6. Start Training ---
    print("\n🚀 Starting training...\n")
    model.fit(
        train_objectives=[(train_dataloader_fixed, train_loss)],
        epochs=NUM_EPOCHS,
        warmup_steps=100,
        output_path=OUTPUT_MODEL_PATH,
        show_progress_bar=True,
        use_amp=False   # Disable mixed precision for CPU
    )

    print(f"\n🎉 Training complete! Model saved to '{OUTPUT_MODEL_PATH}'")
    print(f"Model trained on classes: {list(label_encoder.classes_)}")


if __name__ == '__main__':
    prepare_data_and_train_model()
