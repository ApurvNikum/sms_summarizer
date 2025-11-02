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

# ============================================================
# ⚙ Model Paths
# ============================================================
embedding_model_path = r"output\fine_tuned_20251102_010016"
classifier_path = r"output\classifier_model_20251102_015350.pt"
summary_model_path = r"output\sms_summary_model.pkl"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 🧠 Load Classifier Model
# ============================================================
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

# ============================================================
# 🔤 Embedding + Summary Model
# ============================================================
embedding_model = SentenceTransformer(embedding_model_path, device="cpu")
if device == "cuda":
    embedding_model = embedding_model.to(torch.device("cuda"))

summary_model = joblib.load(summary_model_path)
if isinstance(summary_model, tuple):
    vectorizer, summary_clf, sender_map = summary_model
else:
    vectorizer = summary_model["vectorizer"]
    summary_clf = summary_model["model"]
    sender_map = summary_model.get("sender_map", None)

# ============================================================
# 🔮 Prediction + Summarization Logic
# ============================================================
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

# ============================================================
# 🎨 Streamlit UI Setup
# ============================================================
st.set_page_config(page_title="📩 Smart SMS Insights Dashboard", layout="wide", page_icon="📱")

theme_choice = st.sidebar.radio("🎨 Theme", ["Light", "Dark"], index=0)

# ============================================================
# 💡 THEME STYLES
# ============================================================
if theme_choice == "Dark":
    st.markdown("""
        <style>
        body, .stApp { background-color: #0E1117; color: #EAEAEA; }
        h1,h2,h3,h4,h5,h6,label,.stMarkdown,p,span,div { color: #EAEAEA !important; }
        .card { background-color: #161A1F; border-radius: 12px; padding: 20px; margin: 15px 0; box-shadow: 0 0 8px rgba(255,255,255,0.1); }
        input, textarea, .stTextInput>div>div>input, .stTextArea textarea {
            background-color: #1E1E1E !important;
            color: #EAEAEA !important;
            border: 1px solid #555 !important;
            caret-color: #EAEAEA !important;
        }
        .stButton>button {
            background-color: #333 !important; color: #EAEAEA !important;
            border: 1px solid #777 !important; border-radius: 10px; padding: 6px 16px;
        }
        .stButton>button:hover { background-color: #555 !important; border-color: #999 !important; }
        .stSidebar { background-color: #161A1F !important; color: #EAEAEA !important; }
        .stDataFrame table, .stDataFrame, .css-1d391kg { color: #EAEAEA !important; background-color: #1A1D23 !important; }
        .stRadio label, .stFileUploader label, .stSelectbox label { color: #EAEAEA !important; }
        </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
        <style>
        body, .stApp {
            background-color: #F4F6FA;
            color: #111111;
        }
        h1,h2,h3,h4,h5,h6,label,.stMarkdown,p,span,div {
            color: #111111 !important;
        }
        .card {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        input, textarea,
        .stTextInput>div>div>input,
        .stTextArea textarea {
            background-color: #FFFFFF !important;
            color: #111111 !important;
            border: 1px solid #CCCCCC !important;
            caret-color: #111111 !important;
            border-radius: 8px !important;
        }
        .stButton>button {
            background-color: #007BFF !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px;
            padding: 8px 18px;
            font-weight: 500;
            transition: background-color 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #0056b3 !important;
        }
        .stSidebar {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E0E0E0 !important;
        }
        .stDataFrame table,
        .stDataFrame,
        .css-1d391kg {
            color: #111111 !important;
            background-color: #FFFFFF !important;
        }
        .stRadio label,
        .stFileUploader label,
        .stSelectbox label {
            color: #111111 !important;
            font-weight: 500 !important;
        }
        ::placeholder {
            color: #777777 !important;
        }

        /* ✅ File uploader specific fix */
        .stFileUploader {
            background-color: #FFFFFF !important;
            border: 1px dashed #CCCCCC !important;
            border-radius: 10px !important;
            padding: 1rem !important;
        }
        .stFileUploader div[data-testid="stFileUploadDropzone"] {
            background-color: #FFFFFF !important;
            color: #111111 !important;
            border-radius: 10px !important;
        }
        .stFileUploader div[data-testid="stFileUploadDropzone"] * {
            color: #111111 !important;
        }
        .stFileUploader label {
            color: #111111 !important;
        }
        </style>
    """, unsafe_allow_html=True)
# ✅ Universal file uploader visibility fix for both light & dark themes
st.markdown("""
    <style>
    /* Make Streamlit file uploader clearly visible in both themes */
    .stFileUploader {
        background-color: transparent !important;
    }
    div[data-testid="stFileUploadDropzone"] {
        background-color: #1E88E5 !important;  /* balanced blue for both themes */
        border-radius: 10px !important;
        border: none !important;
        color: #FFFFFF !important;
        transition: background-color 0.2s ease;
    }
    div[data-testid="stFileUploadDropzone"]:hover {
        background-color: #1565C0 !important;
    }
    div[data-testid="stFileUploadDropzone"] * {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }
    </style>
""", unsafe_allow_html=True)




# ============================================================
# 🏷 Header
# ============================================================
st.markdown("""
<h1 style='text-align:center; margin-bottom:5px;'>📩 Smart SMS Insights Dashboard</h1>
<p style='text-align:center; color:gray; margin-bottom:30px;'>Summarize and categorize your SMS messages automatically</p>
""", unsafe_allow_html=True)

# ============================================================
# 📥 Input Options
# ============================================================
input_mode = st.radio("Select Input Mode", ["📂 Upload CSV File", "📝 Write / Paste Messages"])
messages = []
df = None

st.markdown("<div class='card'>", unsafe_allow_html=True)

if input_mode == "📂 Upload CSV File":
    uploaded_file = st.file_uploader("Upload a CSV (must contain a 'Message' column)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        msg_col = next((c for c in ["Message","message","body_lower","text","sms"] if c in df.columns), df.columns[0])
        df[msg_col] = df[msg_col].astype(str)
        messages = df[msg_col].tolist()
        st.success(f"✅ Loaded {len(messages)} messages from file.")
else:
    st.markdown("""
    ✍ How to write your messages:
    - One message per line in the format Sender: Message
    - Example:
      
      Amazon: Your package will arrive tomorrow  
      HDFC Bank: Your OTP is 283716 for login  
      HR Dept: Meeting rescheduled to 5 PM today
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

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 🚀 Run Analysis
# ============================================================
if st.button("🚀 Analyze Messages") and messages:
    with st.spinner("Analyzing and summarizing messages... please wait"):
        results, valid_msgs = [], []
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

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 📊 Category Overview")
            st.bar_chart(summary.set_index("Category"))
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🧾 Summarized Insights")
            if df is not None and any(col.lower() == "sender" for col in df.columns):
                sender_col = [col for col in df.columns if col.lower() == "sender"][0]
                summaries = []
                for sender, group in df.groupby(sender_col):
                    msgs = group["Message"].dropna().astype(str).tolist()
                    summary_text = summarize_sender_texts(msgs)
                    summaries.append({"Sender": sender, "Summary": summary_text, "Messages": len(msgs)})
                st.dataframe(pd.DataFrame(summaries))
            else:
                auto_summary = auto_summarize(valid_msgs)
                st.dataframe(auto_summary)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🔍 Detailed Message Analysis")
            st.dataframe(df_results, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            csv_out = df_results.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Download Categorized Messages", data=csv_out, file_name="sms_categorized.csv", mime="text/csv")

            st.success("✅ Analysis complete! Summaries and categories ready.")
else:
    st.info("👆 Choose an input mode and provide messages, then click Analyze Messages.")