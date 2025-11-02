import os
import torch
import streamlit as st
from torch import nn
from sentence_transformers import SentenceTransformer

# ==========================================================
# ✅ Config
# ==========================================================
st.set_page_config(page_title="📱 SMS Summarizer", layout="wide")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Update to your latest model paths
CLASSIFIER_PATH = "output/classifier_model_20251102_015350.pt"
FINE_TUNED_DIR = "output/fine_tuned_20251102_010016"

# ==========================================================
# ✅ Safe Load Classifier + Encoder
# ==========================================================
@st.cache_resource
def load_models():
    try:
        checkpoint = torch.load(CLASSIFIER_PATH, map_location=device)

        # Label mapping
        label_mapping = checkpoint.get("label_mapping", checkpoint.get("labels_map", {}))
        inv_label_map = {v: k for k, v in label_mapping.items()}

        fine_tuned_model_path = checkpoint.get("fine_tuned_model_path", FINE_TUNED_DIR)
        input_dim = checkpoint.get("input_dim", 384)

        # Load encoder
        encoder = SentenceTransformer(fine_tuned_model_path, device=device)

        # ✅ Define classifier with same architecture as training
        class SMSClassifier(nn.Module):
            def __init__(self, input_dim, num_classes):
                super(SMSClassifier, self).__init__()
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
            def forward(self, emb):
                return self.net(emb)

        classifier = SMSClassifier(input_dim, len(label_mapping)).to(device)
        classifier.load_state_dict(checkpoint["classifier_state_dict"], strict=True)
        classifier.eval()

        # Sanity check
        w = list(classifier.net[0].weight.detach().cpu().numpy().flatten()[:5])
        st.write("✅ Classifier loaded successfully! First few weights:", w)

        return encoder, classifier, inv_label_map
    except Exception as e:
        st.error(f"❌ Failed to load classifier/encoder: {e}")
        return None, None, None


encoder, classifier, inv_label_map = load_models()

# ==========================================================
# ✅ Streamlit Frontend
# ==========================================================
st.title("📩 SMS Summarizer Prototype")
st.markdown("Categorizes your SMS messages and generates a quick **daily summary.**")

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
mode = st.sidebar.radio("Choose input mode:", ["Sample Messages", "Enter Custom Messages"])

if mode == "Sample Messages":
    messages = default_sms
else:
    raw_input = st.sidebar.text_area("Paste or type messages (one per line):", height=200)
    messages = [m.strip() for m in raw_input.split("\n") if m.strip()]

# ==========================================================
# ✅ Classification Logic
# ==========================================================
if st.sidebar.button("Categorize Messages") and messages:
    if encoder is None or classifier is None:
        st.error("❌ Models not loaded properly.")
    else:
        with st.spinner("Analyzing messages..."):
            # Encode
            emb = encoder.encode(
                messages,
                convert_to_tensor=True,
                device=device,
                normalize_embeddings=False
            )

            # Classify
            with torch.no_grad():
                logits = classifier(emb)
                probs = torch.softmax(logits, dim=1)
                top2 = torch.topk(probs, k=2, dim=1)

            categorized = []
            for i, msg in enumerate(messages):
                top_labels = [inv_label_map[idx.item()] for idx in top2.indices[i]]
                top_scores = top2.values[i].tolist()

                # ✅ Skip or mask "Noise/Unlabeled"
                if top_labels[0] == "Noise/Unlabeled" and len(top_labels) > 1:
                    label = top_labels[1]
                    conf = top_scores[1]
                else:
                    label = top_labels[0]
                    conf = top_scores[0]

                categorized.append((msg, label, conf))

        # ======================================================
        # ✅ Show Results
        # ======================================================
        st.subheader("📊 Categorized Messages")
        for msg, label, conf in categorized:
            if label in ["Transactional/Security", "Personal"]:
                continue
            st.markdown(f"**🗂️ {label} ({conf*100:.1f}%):** {msg}")

        st.subheader("🧠 Daily Summary")
        summary = {}
        for _, label, _ in categorized:
            if label in ["Transactional/Security", "Personal"]:
                continue
            summary[label] = summary.get(label, 0) + 1

        if summary:
            for cat, count in summary.items():
                st.write(f"- {cat}: {count} messages")
        else:
            st.info("No summarizable messages found (only personal or OTP-related).")
