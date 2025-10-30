import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling
import torch
import os
import re

# --- Configuration ---
MODEL_NAME_PATH = './sms_classification_model' # Folder where your model was saved
INPUT_DATA_FILE = 'annotated_data_set.csv'     # File to load labels from (for decoder)
# Cosine similarity threshold for considering two promotional messages a duplicate
DUPLICATION_THRESHOLD = 0.95 
# ---

def load_encoder(data_file):
    """
    Loads the LabelEncoder classes used during training to map numerical IDs back to labels.
    """
    try:
        df = pd.read_csv(data_file)
        
        # Ensure we only use the categories the model was trained on
        CORE_CATEGORIES = ['Personal', 'Transactional/Security', 'Status/Alert', 'Travel', 'Telecom', 'Retail', 'Banking', 'Education']
        df_train = df[df['annotation_category'].isin(CORE_CATEGORIES)].copy()
        
        # Ensure all categories have at least 2 samples for encoder fitting
        valid_categories = df_train['annotation_category'].value_counts()
        valid_categories = valid_categories[valid_categories >= 2].index
        df_train = df_train[df_train['annotation_category'].isin(valid_categories)]

        label_encoder = LabelEncoder()
        label_encoder.fit(df_train['annotation_category'])
        return label_encoder
    except Exception as e:
        print(f"Error loading or fitting encoder: {e}")
        return None

def load_and_classify(model, texts_to_classify, label_encoder):
    """
    Classifies a list of texts using the trained S-BERT model.
    Returns: DataFrame with text, predicted category, and embeddings.
    """
    if not texts_to_classify:
        return pd.DataFrame()

    print(f"\n--- 1. Generating Embeddings for {len(texts_to_classify)} Messages ---")
    
    # 1. Generate embeddings (vector representations)
    embeddings = model.encode(texts_to_classify, convert_to_tensor=True, show_progress_bar=True)
    
    # --- 2. Get the classification layer (FIXED) ---
    # The classification layer (Dense layer) is typically saved alongside the model weights
    # when SoftmaxLoss is used. We load it manually here.
    
    # The saved model directory usually contains a '0_Transformer', '1_Pooling', and '2_Dense' (the classifier)
    # Since we manually constructed the base model, the classification layer is separate.
    
    # We must load the classification weights directly from the checkpoint
    try:
        # Load the state dictionary for the classification layer which SoftmaxLoss saved
        # The SoftmaxLoss structure is usually a Dense layer (classifier) followed by a softmax activation.
        
        # We assume the model saved the classification layer weights inside the 'sms_classification_model' folder
        # We need to find the Dense layer weights (saved as model.bin or similar) and apply them to a temporary Dense layer
        
        # NOTE: SBERT saves the final Softmax layer weights in a 'softmax_loss_model.bin' or similar file if saved correctly.
        
        # Since standard SBERT training saves the classifier under '2_Dense' (or similar structure) 
        # when loading the whole model, let's revert to the simplest check that often works:
        
        # Check if the model has the classifier component directly added as an attribute
        if hasattr(model, 'classifier'):
             # This means the Dense layer was loaded directly onto the model object
             logits = model.classifier(embeddings).cpu().numpy()
        elif hasattr(model[-1], 'classifier'):
             # This means the Dense layer is inside the last module (e.g., SoftmaxLoss)
             logits = model[-1].classifier(embeddings).cpu().numpy()
        else:
             # If the layer is not an attribute of the model or the last module,
             # we assume the model only contains the embedding layers (Transformer, Pooling).
             # We need to load the weights of the Dense layer from the output path manually.
             
             # Fallback: We try to load the classification layer manually from the checkpoint.
             classification_layer_path = os.path.join(MODEL_NAME_PATH, '2_Dense', 'pytorch_model.bin')
             
             # This requires defining the Dense layer structure first (input dim = output dim of pooling, output dim = num_labels)
             num_labels = len(label_encoder.classes_)
             
             # Load the initial model to get the embedding dimension
             embedding_dim = model.get_sentence_embedding_dimension()
             
             # Define the Dense layer structure
             from torch import nn
             classifier = nn.Linear(embedding_dim, num_labels)
             
             # Load the saved state dict
             if os.path.exists(classification_layer_path):
                 print(f"Attempting to load classification layer from {classification_layer_path}")
                 state_dict = torch.load(classification_layer_path, map_location=torch.device('cpu'))
                 classifier.load_state_dict(state_dict)
                 
                 # Calculate logits using the loaded classifier
                 logits = classifier(embeddings).cpu().numpy()
             else:
                 # If we cannot find the Dense layer, we cannot classify.
                 raise AttributeError("Could not find the necessary classification layer (Dense weights).")

    except AttributeError as e:
        # Re-raise specific error related to classification layer access
        raise AttributeError(f"Failed to access classification layer: {e}. Check if training saved the final layer correctly.")
    except Exception as e:
        # Catch all other errors
        raise Exception(f"An unexpected error occurred during classification: {e}")

    # Convert logits to probability scores, then find the class with the highest score
    predicted_ids = np.argmax(logits, axis=1)
    
    # Decode the numerical IDs back to the original category names
    predicted_categories = label_encoder.inverse_transform(predicted_ids)

    df_results = pd.DataFrame({
        'text_input': texts_to_classify,
        'category': predicted_categories,
        'embedding': embeddings.cpu().numpy().tolist() # Store embeddings as list
    })
    
    print(f"✅ Classification complete. Found {len(df_results)} classified messages.")
    return df_results

def apply_de_duplication(df_promo):
    """
    Applies hierarchical clustering (using cosine similarity) to group duplicate offers.
    """
    if df_promo.empty:
        return df_promo
    
    print(f"\n--- 2. Applying De-Duplication (Threshold: {DUPLICATION_THRESHOLD}) ---")
    
    # Convert embedding lists back to a NumPy array for fast calculation
    embeddings_matrix = np.array(df_promo['embedding'].tolist())
    
    # Calculate Cosine Similarity Matrix
    similarity_matrix = cosine_similarity(embeddings_matrix)
    
    # De-duplication logic: Hierarchical Grouping
    df_promo['cluster_id'] = -1
    current_cluster_id = 0
    
    for i in range(len(df_promo)):
        if df_promo.iloc[i]['cluster_id'] == -1:
            # Start a new cluster
            df_promo.iloc[i, df_promo.columns.get_loc('cluster_id')] = current_cluster_id
            
            # Find all similar messages in the unclustered data
            similar_indices = np.where(similarity_matrix[i] >= DUPLICATION_THRESHOLD)[0]
            
            # Assign the cluster ID to all similar messages
            for j in similar_indices:
                if df_promo.iloc[j]['cluster_id'] == -1:
                    df_promo.iloc[j, df_promo.columns.get_loc('cluster_id')] = current_cluster_id
            
            current_cluster_id += 1
            
    # Select only the first message from each cluster as the representative (de-duplicated) message
    df_dedup = df_promo.sort_values(by='cluster_id').groupby('cluster_id').first().reset_index()
    
    print(f"✅ De-duplication complete. Reduced from {len(df_promo)} messages to {len(df_dedup)} unique clusters.")
    
    # Count how many duplicates were in each cluster
    cluster_counts = df_promo['cluster_id'].value_counts().reset_index()
    cluster_counts.columns = ['cluster_id', 'duplicate_count']
    
    df_dedup = pd.merge(df_dedup, cluster_counts, on='cluster_id')
    
    # Add number of duplicates to the representative text
    df_dedup['summary_text'] = df_dedup.apply(
        lambda row: f"{row['text_input']} ({row['duplicate_count']} total messages)", axis=1
    )
    
    return df_dedup

def generate_digest(df_classified, original_messages_count):
    """
    Generates the final categorized digest, separating immediate alerts from the summary.
    """
    
    # Separate Immediate Alerts (Security Filter)
    ALERT_CATEGORIES = ['Personal', 'Transactional/Security', 'Status/Alert']
    df_alerts = df_classified[df_classified['category'].isin(ALERT_CATEGORIES)]
    
    # Promotional messages for de-duplication
    df_promo = df_classified[~df_classified['category'].isin(ALERT_CATEGORIES)].reset_index(drop=True)
    
    # De-duplicate promotional messages
    df_dedup = apply_de_duplication(df_promo)
    
    # Group remaining unique promotional messages by category
    digest = {}
    
    for category in df_dedup['category'].unique():
        df_cat = df_dedup[df_dedup['category'] == category]
        digest[category] = df_cat['summary_text'].tolist()

    # --- Print Final Digest ---
    print("\n" + "="*50)
    print(f"| INBOX CLARITY DAILY DIGEST | (Processed: {original_messages_count} messages) ")
    print("="*50)
    
    # 1. Immediate Alerts
    if not df_alerts.empty:
        print("\n[ IMMEDIATE INBOX (SECURITY FILTER BYPASS) ]")
        for index, row in df_alerts.iterrows():
            print(f"- [ {row['category']} ]: {row['text_input']}")
    
    # 2. Daily Summary
    if digest:
        print("\n[ DAILY PROMOTIONAL SUMMARY ]")
        for category, summaries in digest.items():
            print(f"\n--- {category.upper()} OFFERS ({len(summaries)} unique deals) ---")
            for summary in summaries:
                # Remove sender and [SEP] for cleaner summary view
                clean_summary = re.sub(r'\[SEP\].*$', '', summary).strip()
                print(f"  * {clean_summary}")
    else:
        print("\nNo promotional messages found for summarization.")
    
    print("\n" + "="*50)


if __name__ == '__main__':
    # Define sample messages simulating new SMS data
    # NOTE: You would typically get the sender and body separately, but for inference, we combine them
    SAMPLE_MESSAGES = [
        "AX-HDFCBK [SEP] Your account has been credited with AMOUNT_MASK on 29/10/2025. Ref: 9876. Thank you.",
        "DM-MYNTRA [SEP] FLAT 50% OFF on all new arrivals! Use code SALE50. Shop now.",
        "AM-FLIPKT [SEP] FLAT 50% OFF on all new arrivals! Use code SALE50. Shop now before they run out!", # Duplicate of above
        "QP-EDUINST [SEP] Admit card for JEE 2026 available. Login to the portal using your ID.",
        "VI-APP [SEP] Get 1GB free data pack with every recharge today. Expires soon.",
        "VI-APP [SEP] Enjoy unlimited music and video streaming on the Vi app for free this month!", # New Telecom service promo
        "VM-TRAIN [SEP] Your train ticket booking is confirmed. PNR: TKT1234. Happy journey.",
        "9876543210 [SEP] Hey, are we meeting for dinner tomorrow? Call me back.", # Personal message
        "AX-HDFCBK [SEP] Special offer: Get 5% extra cashback on spends over AMOUNT_MASK this Diwali.", # New Banking Offer
    ]

    # 0. Load Model and Encoder
    try:
        # We need the local path for the SentenceTransformer constructor
        model = SentenceTransformer(MODEL_NAME_PATH)
        label_encoder = load_encoder(INPUT_DATA_FILE)
        
        if label_encoder is None:
            raise Exception("Could not load Label Encoder. Check data file.")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: Failed to load model or encoder. Ensure '{MODEL_NAME_PATH}' exists.")
        print(f"Details: {e}")
        exit()
        
    # 1. Classify all incoming messages
    df_classified = load_and_classify(model, SAMPLE_MESSAGES, label_encoder)
    
    # 2. Generate the final digest
    generate_digest(df_classified, len(SAMPLE_MESSAGES))
