import streamlit as st
import torch
import torch.nn.functional as F
from torch import nn
from sentence_transformers import SentenceTransformer
import pandas as pd
import re
import os
import joblib
from sklearn.cluster import KMeans

embedding_model_path = "models/fine_tuned_embedding"
classifier_path = "models/classifier_model.pt"
summary_model_path = "models/sms_summary_model.pkl"

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(classifier_path, map_location=device)
labels_map = checkpoint["labels_map"]
reverse_labels_map = {v: k for k, v in labels_map.items()}
num_classes = len(labels_map)
input_dim = checkpoint["input_dim"]

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

classifier = SMSClassifier(input_dim, num_classes).to(device)
classifier.load_state_dict(checkpoint["classifier_state_dict"], strict=False)
classifier.eval()

embedding_model = SentenceTransformer(embedding_model_path, device=device)
summary_model = joblib.load(summary_model_path)

if isinstance(summary_model, tuple):
    vectorizer, summary_clf, sender_map = summary_model
else:
    vectorizer = summary_model["vectorizer"]
    summary_clf = summary_model["model"]
    sender_map = summary_model.get("sender_map", None)

def predict_sms(text):
    embedding = embedding_model.encode([text], convert_to_tensor=True).to(device)
    with torch.no_grad():
        logits = classifier(embedding)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred_label = reverse_labels_map[int(probs.argmax())]
    return pred_label

def is_otp(text):
    return bool(re.search(r'\b\d{4,8}\b', text)) and "otp" in text.lower()

def summarize_sender_texts(messages):
    joined = " ".join(messages)
    X_vec = vectorizer.transform([joined])
    label_pred = summary_clf.predict(X_vec)[0]
    top_words = " ".join(sorted(set(joined.split()[:60]), key=len))
    return f"Summary ({label_pred}): {top_words[:300]}..."

def auto_summarize(messages, n_clusters=5):
    embeddings = embedding_model.encode(messages, convert_to_tensor=True)
    n_clusters = min(n_clusters, len(messages))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings.cpu().numpy())

    grouped = pd.DataFrame({"Message": messages, "Cluster": clusters})
    summaries = []
    for c in sorted(grouped["Cluster"].unique()):
        group_msgs = grouped[grouped["Cluster"] == c]["Message"].tolist()
        joined = " ".join(group_msgs)
        X_vec = vectorizer.transform([joined])
        label_pred = summary_clf.predict(X_vec)[0]
        top_words = " ".join(sorted(set(joined.split()[:60]), key=len))
        summaries.append({
            "Group": f"Cluster {c+1}",
            "Summary": f"Summary ({label_pred}): {top_words[:300]}...",
            "Messages": len(group_msgs)
        })
    return pd.DataFrame(summaries)

st.set_page_config(page_title="📩 Smart SMS Insights Dashboard", layout="wide", page_icon="📱")

theme_choice = st.sidebar.radio("🎨 Theme", ["Light", "Dark"], index=0)

if theme_choice == "Dark":
    st.markdown("""
        <style>
        body, .stApp { background-color: #0E1117; color: #EAEAEA; }
        .stButton>button { background-color: #262730; color: #EAEAEA; border: 1px solid #444; }
        .stButton>button:hover { background-color: #333; color: white; }
        .stTextArea textarea { background-color: #1E1E1E; color: #EAEAEA; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body, .stApp { background-color: white; color: black; }
        </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>📩 Smart SMS Insights Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Summarize and categorize your SMS messages automatically</p>", unsafe_allow_html=True)

input_mode = st.radio("Select Input Mode", ["📂 Upload CSV File", "📝 Write / Paste Messages"])
messages = []
df = None

if input_mode == "📂 Upload CSV File":
    uploaded_file = st.file_uploader("Upload a CSV (must contain a 'Message' column)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        msg_col = None
        for possible in ["Message", "message", "body_lower", "text", "sms"]:
            if possible in df.columns:
                msg_col = possible
                break
        if msg_col is None:
            msg_col = df.columns[0]
        df[msg_col] = df[msg_col].astype(str)
        messages = df[msg_col].tolist()
        st.success(f"✅ Loaded {len(messages)} messages from file.")
else:
    st.markdown("""
    ✍️ **How to write your messages:**
    - One message per line in the format: Sender: Message
    """)
    user_input = st.text_area("Paste or write messages below:", height=250)

    if user_input.strip():
        raw_lines = [m.strip() for m in user_input.strip().split("\n") if len(m.strip()) > 0]
        senders, msgs = [], []
        for line in raw_lines:
            if ":" in line:
                sender, msg = line.split(":", 1)
                senders.append(sender.strip())
                msgs.append(msg.strip())
            else:
                senders.append("Unknown")
                msgs.append(line.strip())
        df = pd.DataFrame({"Sender": senders, "Message": msgs})
        messages = msgs
        st.success(f"✅ {len(messages)} messages ready for analysis (with senders).")

if st.button("🚀 Analyze Messages") and messages:
    with st.spinner("Analyzing and summarizing messages... please wait"):
        results = []
        valid_msgs = []
        for msg in messages:
            if is_otp(msg):
                continue
            label = predict_sms(msg)
            valid_msgs.append(msg)
            results.append({"Message": msg, "Category": label})

        if not results:
            st.warning("No valid messages found after filtering OTP content.")
        else:
            df_results = pd.DataFrame(results)
            summary = df_results["Category"].value_counts().reset_index()
            summary.columns = ["Category", "Count"]
            st.markdown("### 📊 Category Overview")
            st.bar_chart(summary.set_index("Category"))
            st.markdown("### 🧾 Summarized Insights")

            if df is not None and any(col.lower() == "sender" for col in df.columns):
                sender_col = [col for col in df.columns if col.lower() == "sender"][0]
                summaries = []
                for sender, group in df.groupby(sender_col):
                    msgs = group["Message"].dropna().astype(str).tolist()
                    summary_text = summarize_sender_texts(msgs)
                    summaries.append({
                        "Sender": sender,
                        "Summary": summary_text,
                        "Messages": len(msgs)
                    })
                st.dataframe(pd.DataFrame(summaries))
            else:
                auto_summary = auto_summarize(valid_msgs)
                st.dataframe(auto_summary)

            st.markdown("### 🔍 Detailed Message Analysis")
            st.dataframe(df_results, use_container_width=True)

            csv_out = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="💾 Download Categorized Messages",
                data=csv_out,
                file_name="sms_categorized.csv",
                mime="text/csv"
            )
            st.success("✅ Analysis complete! Summaries and categories ready.")
else:
    st.info("👆 Choose an input mode and provide messages, then click **Analyze Messages**.")
