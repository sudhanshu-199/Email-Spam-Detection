"""
Email Spam Detection System - Model Training Script
Authors     : Sudhanshu Ranjan & Khushwinder Kaur
Internship  : Solitaire Infosystems
Description : Loads SMS + Email datasets, cleans text, trains a LinearSVC
              model using TF-IDF vectorization, and saves the model to disk.
"""

import os
import re
import joblib
import warnings
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

# ── File paths ────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SMS_PATH   = os.path.join(DATA_DIR, "spam.csv")
EMAIL_PATH = os.path.join(DATA_DIR, "spam_email_dataset.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "spam_model.pkl")
VEC_PATH   = os.path.join(MODELS_DIR, "vectorizer.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)


# ── Step 1: Load datasets ─────────────────────────────────────
def load_data():
    print("Loading datasets...")

    # SMS dataset — columns are v1 (label) and v2 (message text)
    sms = pd.read_csv(SMS_PATH, encoding="latin-1", usecols=["v1", "v2"])
    sms.columns = ["label", "text"]
    sms["source"] = "sms"
    print(f"  SMS: {len(sms)} records")

    # Email dataset — combine Subject + Body as the text
    email = pd.read_csv(EMAIL_PATH)
    email["Subject"] = email["Subject"].fillna("")
    email["Body"]    = email["Body"].fillna("")
    email["text"]    = email["Subject"] + " " + email["Body"]
    email["label"]   = email["Label"].str.lower().str.strip()
    email["source"]  = "email"
    email = email[["label", "text", "source"]]
    print(f"  Email: {len(email)} records")

    return sms, email


# ── Step 2: Merge and clean ───────────────────────────────────
def merge_and_clean(sms, email):
    print("Merging and cleaning...")
    df = pd.concat([sms, email], ignore_index=True)
    df.drop_duplicates(subset=["text"], inplace=True)
    df.dropna(subset=["text", "label"], inplace=True)
    df["text"]  = df["text"].astype(str).str.strip()
    df["label"] = df["label"].str.lower().str.strip()
    df = df[df["label"].isin(["spam", "ham"])].reset_index(drop=True)
    print(f"  Final: {len(df)} records — {df['label'].value_counts().to_dict()}")
    return df


# ── Step 3: Text preprocessing ────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)            # replace URLs
    text = re.sub(r"[\w.+-]+@[\w-]+\.[a-z]+", " email ", text) # replace email addresses
    text = re.sub(r"\d+", " num ", text)                       # replace numbers
    text = re.sub(r"[^a-z\s]", " ", text)                     # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Step 4: Vectorize ─────────────────────────────────────────
def vectorize(df):
    print("Vectorizing with TF-IDF...")
    df["clean_text"] = df["text"].apply(clean_text)

    le = LabelEncoder()
    y  = le.fit_transform(df["label"])   # ham = 0, spam = 1

    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),      # unigrams + bigrams
        sublinear_tf=True,
        stop_words="english",
    )
    X = vectorizer.fit_transform(df["clean_text"])
    print(f"  Feature matrix: {X.shape}  |  Classes: {list(le.classes_)}")
    return X, y, le, vectorizer


# ── Step 5: Train and evaluate ────────────────────────────────
def train(X, y, le):
    print("Training LinearSVC model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    model.fit(X_train, y_train)

    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    cm       = confusion_matrix(y_test, y_pred)

    print(f"\n  Accuracy : {accuracy * 100:.2f}%")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred, target_names=le.classes_)}")
    print(f"  Confusion Matrix:\n{cm}\n")

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report,
        "confusion_matrix": cm,
    }
    return model, metrics


# ── Step 6: Save model ────────────────────────────────────────
def save(model, vectorizer, le, metrics):
    print("Saving model artifacts...")
    metadata = {
        "accuracy":         round(float(metrics["accuracy"]), 4),
        "precision":        round(float(metrics["precision"]), 4),
        "recall":           round(float(metrics["recall"]), 4),
        "f1":               round(float(metrics["f1"]), 4),
        "report":           metrics["report"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "classes":          list(le.classes_),
        "model_type":       "LinearSVC",
    }
    joblib.dump({"model": model, "le": le, "metadata": metadata}, MODEL_PATH)
    joblib.dump(vectorizer, VEC_PATH)
    print(f"  Model saved   : {MODEL_PATH}")
    print(f"  Vectorizer    : {VEC_PATH}")


# ── Main ──────────────────────────────────────────────────────
def main():
    print("\n=== Email Spam Detection — Training Pipeline ===\n")
    sms, email       = load_data()
    df               = merge_and_clean(sms, email)

    # Generate static word cloud images
    print("Generating static word cloud images...")
    spam_text = " ".join(df[df["label"] == "spam"]["text"].dropna().astype(str).tolist())
    ham_text  = " ".join(df[df["label"] == "ham"]["text"].dropna().astype(str).tolist())
    WordCloud(width=600, height=300, background_color="white", colormap="Reds", max_words=100).generate(spam_text).to_file(os.path.join(MODELS_DIR, "spam_wc.png"))
    WordCloud(width=600, height=300, background_color="white", colormap="Blues", max_words=100).generate(ham_text).to_file(os.path.join(MODELS_DIR, "ham_wc.png"))
    print("  Word clouds saved to models/")

    X, y, le, vec    = vectorize(df)
    model, metrics   = train(X, y, le)
    save(model, vec, le, metrics)
    print("\n✅ Training complete!\n")
    return metrics["accuracy"]


if __name__ == "__main__":
    main()
