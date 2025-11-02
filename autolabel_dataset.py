import re
import pandas as pd

df = pd.read_csv("raw_sms.csv")
print(f"Loaded {len(df)} messages")

def clean_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip().lower()

df["clean_body"] = df["message"].apply(clean_text)
df["clean_sender"] = df["address"].astype(str).str.strip()

auth_keywords = [
    r"\botp\b", r"one[-\s]?time[-\s]?password", r"verification\s?code",
    r"login\s?code", r"auth", r"verify", r"2fa"
]

bank_keywords = [
    r"\bbank\b", r"\ba/c\b", r"\baccount\b", r"\bbalance\b",
    r"\bcredit\b", r"\bdebit\b", r"\btxn\b", r"\btransaction\b",
    r"\bupi\b", r"\bifsc\b", r"\bloan\b", r"\bemi\b", r"\bkyc\b",
    r"\bnetbanking\b"
]

telecom_keywords = [
    r"\bvi\b", r"\bjio\b", r"\bairtel\b", r"\bbsnl\b", r"\bvodafone\b",
    r"data\s?pack", r"recharge", r"plan", r"tariff", r"199\b", r"balance\s?check"
]

retail_keywords = [
    r"amazon", r"flipkart", r"myntra", r"ajio", r"nykaa", r"bigbasket",
    r"offer", r"discount", r"sale", r"shopping", r"deal", r"cashback",
    r"zomato", r"swiggy", r"bookmyshow", r"zee5", r"netmeds", r"pharmeasy"
]

travel_keywords = [
    r"flight", r"train", r"pnr", r"boarding", r"check[-\s]?in", r"ticket",
    r"cab", r"uber", r"ola", r"hotel", r"booking", r"journey", r"bus"
]

education_keywords = [
    r"school", r"college", r"university", r"exam", r"test", r"admission",
    r"result", r"marks", r"tuition", r"fees", r"student", r"degree", r"class"
]

def auto_label(sender, body):
    sender = str(sender).strip()
    body = clean_text(body)

    if re.fullmatch(r"\d{10}", sender):
        if not any(re.search(k, body) for k in auth_keywords + bank_keywords + telecom_keywords):
            return "Personal"

    if any(re.search(k, body) for k in auth_keywords + bank_keywords):
        return "Transactional/Security"

    if any(re.search(k, body) for k in telecom_keywords):
        return "Telecom"

    if any(re.search(k, body) for k in retail_keywords):
        return "Retail"

    if any(re.search(k, body) for k in travel_keywords):
        return "Travel"

    if any(re.search(k, body) for k in education_keywords):
        return "Education"

    return "Noise/Unlabeled"

df["annotation_category"] = df.apply(lambda x: auto_label(x["clean_sender"], x["clean_body"]), axis=1)

print("Category distribution:")
print(df["annotation_category"].value_counts())

output_path = "output/annotated_dataset.csv"
df.to_csv(output_path, index=False)
print("Annotated dataset saved to:", output_path)
