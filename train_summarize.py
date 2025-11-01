import os
import csv
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from collections import Counter
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import MBartForConditionalGeneration
try:
    from transformers import MBart50TokenizerFast as MBartTokenizer
except ImportError:
    from transformers import MBartTokenizer
from langdetect import detect
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# --------------------- CONFIG ---------------------
xml_file = r"C:\Users\Rishita\Downloads\sms\sms.xml"
csv_file = r"C:\Users\Rishita\Downloads\sms\sms_output.csv"
summary_csv = r"C:\Users\Rishita\Downloads\sms\summarized_multilingual_sms.csv"
model_save_path = r"C:\Users\Rishita\Downloads\sms\sms_summary_model.pkl"

# --------------------- HELPERS ---------------------
def ms_to_iso(ms):
    try:
        ms = int(ms)
        dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        return dt.isoformat()
    except Exception:
        return ""

# --------------------- XML → CSV ---------------------
if not os.path.exists(csv_file):
    print("🔹 Converting XML → CSV...")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    rows = []

    for sms in root.findall('sms'):
        contact_name = sms.get('contact_name')
        address = sms.get('address', 'Unknown')
        sender = address if contact_name in [None, '', '(Unknown)'] else contact_name

        message = sms.get('body', '').replace('\n', ' ').strip()
        date_ms = sms.get('date', '')
        date_iso = ms_to_iso(date_ms)
        typ = sms.get('type', '')
        direction = "Received" if typ == "1" else "Sent" if typ == "2" else "Unknown"

        rows.append({
            'Sender': sender,
            'Direction': direction,
            'Date': date_iso,
            'Message': message
        })

    with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Sender', 'Direction', 'Date', 'Message'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ XML converted to CSV: {csv_file} ({len(rows)} messages)")
else:
    print(f"✅ CSV already exists: {csv_file}")

# --------------------- LOAD CSV ---------------------
df = pd.read_csv(csv_file)
df = df[df['Message'].notna() & df['Message'].str.strip().astype(bool)]

# --------------------- LOAD MODELS ---------------------
print("🔹 Loading models...")
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model_name = "facebook/mbart-large-50-many-to-one-mmt"
try:
    tokenizer = MBartTokenizer.from_pretrained(model_name)
except Exception as e:
    print("⚠️ Falling back to slow tokenizer:", e)
    from transformers import MBartTokenizer as SlowMBartTokenizer
    tokenizer = SlowMBartTokenizer.from_pretrained(model_name)

model = MBartForConditionalGeneration.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def summarize_text(text, lang_code="en_XX"):
    tokenizer.src_lang = lang_code
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    summary_ids = model.generate(inputs["input_ids"], max_length=40, min_length=10, num_beams=4)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# --------------------- GROUP & SUMMARIZE ---------------------
final_summaries = []
print("🔹 Summarizing messages...")

for sender, group in df.groupby('Sender'):
    messages = group['Message'].astype(str).tolist()
    embeddings = embedder.encode(messages, convert_to_tensor=True)

    visited = set()
    grouped_msgs = []
    for i in range(len(messages)):
        if i in visited:
            continue
        sim_scores = util.pytorch_cos_sim(embeddings[i], embeddings)[0]
        similar = [j for j, s in enumerate(sim_scores) if s > 0.80]
        visited.update(similar)
        grouped_msgs.append([messages[j] for j in similar])

    sender_summary_parts = []
    for cluster in grouped_msgs:
        combined = " ".join(cluster)
        combined = " ".join(combined.split()[:300])
        try:
            lang = detect(combined)
        except:
            lang = "en"
        lang_map = {
            "en": "en_XX", "hi": "hi_IN", "bn": "bn_IN", "ta": "ta_IN",
            "te": "te_IN", "ml": "ml_IN", "gu": "gu_IN", "mr": "mr_IN"
        }
        lang_code = lang_map.get(lang, "en_XX")
        summary = summarize_text(combined, lang_code)
        sender_summary_parts.append(summary)

    final_summaries.append({
        "Sender": sender,
        "Summary": " | ".join(sender_summary_parts)
    })

summary_df = pd.DataFrame(final_summaries)
summary_df.to_csv(summary_csv, index=False)
print(f"✅ Summarized CSV saved: {summary_csv}")
print(summary_df.head())

# --------------------- TRAIN MODEL ---------------------
print("\n🔹 Training model on summaries...")

X = summary_df["Summary"].astype(str)
y = summary_df["Sender"].astype(str)

label_counts = Counter(y)
if all(v >= 2 for v in label_counts.values()):
    stratify_opt = y
else:
    print("⚠️ Not enough samples for stratified split, using random split.")
    stratify_opt = None

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_opt
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_tfidf, y_train)
y_pred = clf.predict(X_test_tfidf)

print("\n🔹 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# --------------------- SAVE MODEL ---------------------
joblib.dump({"vectorizer": vectorizer, "model": clf}, model_save_path)
print(f"✅ Model saved to: {model_save_path}")
