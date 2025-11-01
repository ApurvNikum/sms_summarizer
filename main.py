import os
import torch
import streamlit as st
from torch import nn
from sentence_transformers import SentenceTransformer
from datetime import datetime

# ==========================================================
# ✅ Config
# ==========================================================
st.set_page_config(page_title="📱 SMS Summarizer", layout="wide")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Change this to your latest model paths
classifier_path = "output/classifier_model_20251101_220130.pt"
fine_tuned_model_dir = "output/fine_tuned_20251031_162336"

# ==========================================================
# ✅ Load Encoder (Fine-Tuned SentenceTransformer)
# ==========================================================
@st.cache_resource
def load_encoder():
    encoder = SentenceTransformer(fine_tuned_model_dir, device=device)
    return encoder

encoder = load_encoder()

# ==========================================================
# ✅ Define Model (Matches retrain_classifier.py)
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
        with torch.no_grad():
            sentence_embeddings = self.encoder.encode(
                texts,
                convert_to_tensor=True,
                device=device,
                show_progress_bar=False
            )
        embeddings = sentence_embeddings.detach().clone()
        logits = self.classifier(embeddings)
        return logits

# ==========================================================
# ✅ Safe Load Classifier
# ==========================================================
@st.cache_resource
def load_classifier():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier_path = "output/classifier_model_20251101_220130.pt"  # update if needed

    # Safe load
    checkpoint = torch.load(classifier_path, map_location=device, weights_only=True)
    
   

    # Handle multiple possible key names
    label_mapping = checkpoint.get("label_mapping", checkpoint.get("labels_map", {}))
    inv_label_map = {v: k for k, v in label_mapping.items()}

    fine_tuned_model_path = checkpoint.get("fine_tuned_model_path", "output/fine_tuned_20251031_162336")
    input_dim = checkpoint.get("input_dim", 384)

    # Load encoder
    encoder = SentenceTransformer(fine_tuned_model_path, device=device)

    # Define classifier architecture
    import torch.nn as nn
    class SMSClassifier(nn.Module):
        def __init__(self, encoder, input_dim, num_classes):
            super(SMSClassifier, self).__init__()
            self.encoder = encoder
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )

        def forward(self, texts):
            with torch.no_grad():
                emb = self.encoder.encode(texts, convert_to_tensor=True, device=device)
            emb = emb.detach().clone().to(device)
            return self.net(emb)


    # Initialize classifier
    classifier = SMSClassifier(encoder, input_dim, num_classes=len(label_mapping)).to(device)
    classifier.load_state_dict(checkpoint["classifier_state_dict"], strict=False)
    classifier.eval()

    return classifier, inv_label_map


# ==========================================================
# ✅ Load Classifier Before UI (Fixes NameError)
# ==========================================================
classifier, inv_label_map = load_classifier()


# ==========================================================
# ✅ Streamlit Frontend
# ==========================================================
st.title("📩 SMS Summarizer Prototype")
st.markdown("Categorizes your SMS messages and generates a quick **daily summary.**")

# Example mock SMS messages (you can later connect to real data)
default_sms = [
    "Your OTP for login is 982134. Do not share it with anyone.",
    "Hi Rahul, let's meet at 7pm near the cafe!",
    "Airtel: Your data pack of 2GB/day has been activated.",
    "Your payment of ₹499 to Amazon has been received.",
    "IRCTC: Train 12345 is delayed by 20 minutes.",
    "Final reminder: Your tuition fees are due tomorrow.",
    "Get 40% off on your next shopping trip at Big Bazaar!"
]

st.sidebar.header("📨 SMS Inbox Simulator")
user_input_mode = st.sidebar.radio("Choose input mode:", ["Sample Messages", "Enter Custom Messages"])

if user_input_mode == "Sample Messages":
    messages = default_sms
else:
    user_sms = st.sidebar.text_area("Paste or type messages (one per line):", height=200)
    messages = [m.strip() for m in user_sms.split("\n") if m.strip()]

if st.sidebar.button("Categorize Messages") and messages:
    with st.spinner("Analyzing messages..."):
        logits = classifier(messages)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        categorized = [(msg, inv_label_map[p]) for msg, p in zip(messages, preds)]

    st.subheader("📊 Categorized Messages")
    for msg, label in categorized:
        if label in ["Transactional/Security", "Personal"]:
            continue  # Skip private and OTP/security messages
        st.markdown(f"**🗂️ {label}:** {msg}")

    # Summary Section
    st.subheader("🧠 Daily Summary")
    summary = {}
    for _, label in categorized:
        if label in ["Transactional/Security", "Personal"]:
            continue
        summary[label] = summary.get(label, 0) + 1

    if summary:
        for category, count in summary.items():
            st.write(f"- {category}: {count} messages")
    else:
        st.info("No summarizable messages found (only personal or OTP-related).")
