"""
Email Spam Detection System - Streamlit App
Authors     : Sudhanshu Ranjan & Khushwinder Kaur
Internship  : Solitaire Infosystems
Description : A simple Streamlit dashboard for spam detection
              using a LinearSVC model trained on SMS + Email data.
"""

import io
import os
import re
import joblib
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)
from sklearn.model_selection import train_test_split

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="🛡️",
    layout="wide",
)

# ── File paths ────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SMS_PATH   = os.path.join(DATA_DIR, "spam.csv")
EMAIL_PATH = os.path.join(DATA_DIR, "spam_email_dataset.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "spam_model.pkl")
VEC_PATH   = os.path.join(MODELS_DIR, "vectorizer.pkl")


# ── Text cleaning ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)         # replace URLs
    text = re.sub(r"[\w.+-]+@[\w-]+\.[a-z]+", " email ", text)  # replace emails
    text = re.sub(r"\d+", " num ", text)                    # replace numbers
    text = re.sub(r"[^a-z\s]", " ", text)                  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Load dataset ──────────────────────────────────────────────
def load_data():
    # SMS dataset
    sms = pd.read_csv(SMS_PATH, encoding="latin-1", usecols=["v1", "v2"])
    sms.columns = ["label", "text"]
    sms["source"] = "SMS"

    # Email dataset — combine Subject + Body into one text field
    email = pd.read_csv(EMAIL_PATH)
    email["Subject"] = email["Subject"].fillna("")
    email["Body"]    = email["Body"].fillna("")
    email["text"]    = email["Subject"] + " " + email["Body"]
    email["label"]   = email["Label"].str.lower().str.strip()
    email["source"]  = "Email"
    email = email[["label", "text", "source"]]

    # Merge both datasets
    df = pd.concat([sms, email], ignore_index=True)
    df.drop_duplicates(subset=["text"], inplace=True)
    df.dropna(subset=["text", "label"], inplace=True)
    df["text"]  = df["text"].astype(str).str.strip()
    df["label"] = df["label"].str.lower().str.strip()
    df = df[df["label"].isin(["spam", "ham"])].reset_index(drop=True)

    # Extra columns used in visualizations
    df["clean_text"]    = df["text"].apply(clean_text)
    df["msg_length"]    = df["text"].apply(len)
    df["word_count"]    = df["text"].apply(lambda x: len(str(x).split()))
    return df


# ── Load model ────────────────────────────────────────────────
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
        bundle     = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VEC_PATH)
        return bundle["model"], vectorizer, bundle["le"], bundle["metadata"]

    # If model not found, train it first
    st.warning("Model not found — running training pipeline...")
    import train_model as tm
    tm.main()
    bundle     = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    return bundle["model"], vectorizer, bundle["le"], bundle["metadata"]


# ── Compute evaluation metrics (loaded from model metadata) ───


# ── Load everything ───────────────────────────────────────────
with st.spinner("Loading data and model..."):
    df = load_data()
    model, vectorizer, le, metadata = load_model()

spam_df = df[df["label"] == "spam"]
ham_df  = df[df["label"] == "ham"]

# ── Sidebar navigation ────────────────────────────────────────
st.sidebar.title("🛡️ Spam Detector")
st.sidebar.caption("Sudhanshu Ranjan & Khushwinder Kaur | Solitaire Infosystems")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Data Visualization", "Model Performance", "Spam Detector", "About"],
)
st.sidebar.markdown("---")
st.sidebar.info(
    f"**Model:** LinearSVC\n\n"
    f"**Accuracy:** {metadata['accuracy'] * 100:.2f}%\n\n"
    f"**Dataset:** {len(df):,} messages"
)


# ═════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═════════════════════════════════════════════════════════════
if page == "Home":
    st.title("📧 Email Spam Detection System")
    st.markdown(
        "A machine learning system that classifies emails and SMS messages as **Spam** or **Ham** "
        "using Natural Language Processing (NLP) and a LinearSVC model trained on "
        f"**{len(df):,} messages**."
    )
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Messages", f"{len(df):,}")
    col2.metric("Spam Messages",  f"{len(spam_df):,}")
    col3.metric("Ham Messages",   f"{len(ham_df):,}")
    col4.metric("Model Accuracy", f"{metadata['accuracy'] * 100:.2f}%")

    st.divider()
    st.subheader("About the Project")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**📂 Dataset**")
        st.markdown(
            "- SMS Spam Collection (UCI) — 5,572 records\n"
            "- Email Spam Dataset — 5,000 records\n"
            "- Combined after removing duplicates\n"
            "- Labels: `spam` / `ham`"
        )
    with c2:
        st.markdown("**🛠️ Tech Stack**")
        st.markdown(
            "- Python 3.10+\n"
            "- Scikit-Learn (LinearSVC)\n"
            "- TF-IDF Vectorizer\n"
            "- Pandas & NumPy\n"
            "- Streamlit (this app)"
        )
    with c3:
        st.markdown("**🤖 Model**")
        st.markdown(
            "- Algorithm: LinearSVC\n"
            "- Features: TF-IDF (15K, bigrams)\n"
            "- Train/Test: 80% / 20%\n"
            "- Regularization: C = 1.0"
        )


# ═════════════════════════════════════════════════════════════
# PAGE 2 — DATA VISUALIZATION
# ═════════════════════════════════════════════════════════════
elif page == "Data Visualization":
    st.title("📊 Data Visualization")
    st.caption("Visual exploration of the merged SMS + Email spam dataset.")
    st.divider()

    # Row 1: Class distribution
    c1, c2 = st.columns(2)
    counts = df["label"].value_counts().reset_index()
    counts.columns = ["Label", "Count"]

    with c1:
        st.subheader("Spam vs Ham Count")
        fig_bar = px.bar(
            counts,
            x="Label",
            y="Count",
            color="Label",
            color_discrete_map={"ham": "#4A90D9", "spam": "#E05C5C"},
            text="Count"
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(showlegend=False, yaxis_title="Count")
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        st.subheader("Class Proportion")
        fig_pie = px.pie(
            counts,
            names="Label",
            values="Count",
            color="Label",
            color_discrete_map={"ham": "#4A90D9", "spam": "#E05C5C"},
            hole=0.3
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # Row 2: Word clouds
    st.subheader("☁️ Word Clouds")
    wc1, wc2 = st.columns(2)

    SPAM_WC_PATH = os.path.join(MODELS_DIR, "spam_wc.png")
    HAM_WC_PATH  = os.path.join(MODELS_DIR, "ham_wc.png")

    with wc1:
        st.subheader("Spam Messages — Top Words")
        if os.path.exists(SPAM_WC_PATH):
            st.image(SPAM_WC_PATH, use_container_width=True)
        else:
            st.warning("Spam Word Cloud image not found.")
    with wc2:
        st.subheader("Ham Messages — Top Words")
        if os.path.exists(HAM_WC_PATH):
            st.image(HAM_WC_PATH, use_container_width=True)
        else:
            st.warning("Ham Word Cloud image not found.")

    st.divider()

    # Row 3: Message length histogram
    st.subheader("📏 Message Length Distribution")
    df_clipped = df.copy()
    df_clipped["msg_length"] = df_clipped["msg_length"].clip(upper=1000)
    fig_hist = px.histogram(
        df_clipped,
        x="msg_length",
        color="label",
        barmode="overlay",
        color_discrete_map={"ham": "#4A90D9", "spam": "#E05C5C"},
        nbins=50,
        labels={"msg_length": "Message Length (characters)", "label": "Label"}
    )
    fig_hist.update_layout(yaxis_title="Frequency")
    st.plotly_chart(fig_hist, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.title("📈 Model Performance")
    st.caption("Evaluation of the LinearSVC model on a 20% held-out test set.")
    st.divider()

    # Load pre-computed evaluation metrics directly from the model metadata
    metrics = metadata

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{metrics['accuracy']  * 100:.2f}%")
    c2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
    c3.metric("Recall",    f"{metrics['recall']    * 100:.2f}%")
    c4.metric("F1 Score",  f"{metrics['f1']        * 100:.2f}%")

    st.divider()

    # Classification report table
    st.subheader("Classification Report")
    rows = []
    for cls in le.classes_:
        r = metrics["report"][cls]
        rows.append({
            "Class":     cls.upper(),
            "Precision": f"{r['precision'] * 100:.2f}%",
            "Recall":    f"{r['recall']    * 100:.2f}%",
            "F1-Score":  f"{r['f1-score']  * 100:.2f}%",
            "Support":   int(r["support"]),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.divider()

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"Actual {c.upper()}" for c in le.classes_],
        columns=[f"Predicted {c.upper()}" for c in le.classes_]
    )
    st.dataframe(cm_df, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PAGE 4 — SPAM DETECTOR
# ═════════════════════════════════════════════════════════════
elif page == "Spam Detector":
    st.title("🔍 Spam Detector")
    st.caption("Paste any email or SMS below to check if it's spam.")
    st.divider()

    # Keep prediction history across button clicks
    if "history" not in st.session_state:
        st.session_state.history = []

    # Quick example buttons
    col_ex1, col_ex2, _ = st.columns([1, 1, 2])
    with col_ex1:
        if st.button("Try Spam Example"):
            st.session_state["example_text"] = (
                "WINNER! You've been selected to receive a FREE iPhone. "
                "Call 1-800-WIN-NOW or click win.prize.com to claim!"
            )
    with col_ex2:
        if st.button("Try Ham Example"):
            st.session_state["example_text"] = (
                "Hi, just confirming our meeting tomorrow at 3 PM is still on. "
                "Let me know if you need to reschedule!"
            )

    default_text = st.session_state.get("example_text", "")
    user_input = st.text_area("Enter message text:", value=default_text, height=150,
                              placeholder="Paste your email or SMS here...")

    if st.button("🔍 Check Message", type="primary"):
        if not user_input.strip():
            st.warning("Please enter a message first.")
        else:
            # Preprocess → vectorize → predict
            cleaned = clean_text(user_input)
            vec     = vectorizer.transform([cleaned])
            pred    = model.predict(vec)[0]
            label   = le.inverse_transform([pred])[0]

            # Confidence from decision function distance
            try:
                score      = model.decision_function(vec)[0]
                confidence = min(100.0, abs(float(score)) / 2 * 100)
            except Exception:
                confidence = 80.0

            st.divider()
            if label == "spam":
                st.error(f"🚨 **SPAM** — This message looks like spam! (Confidence: {confidence:.1f}%)")
            else:
                st.success(f"✅ **HAM** — This looks like a legitimate message. (Confidence: {confidence:.1f}%)")

            col1, col2, col3 = st.columns(3)
            col1.metric("Result",     label.upper())
            col2.metric("Confidence", f"{confidence:.1f}%")
            col3.metric("Length",     f"{len(user_input)} chars")

            # Save to session history
            st.session_state.history.append({
                "Time":     datetime.datetime.now().strftime("%H:%M:%S"),
                "Result":   label.upper(),
                "Confidence": f"{confidence:.1f}%",
                "Preview":  user_input[:60] + "..." if len(user_input) > 60 else user_input,
            })

    # Prediction history table
    if st.session_state.history:
        st.divider()
        st.subheader("Prediction History")
        st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True, use_container_width=True)
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()


# ═════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ═════════════════════════════════════════════════════════════
elif page == "About":
    st.title("ℹ️ About This Project")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👨‍💻 Developers")
        st.markdown(
            "**Sudhanshu Ranjan** & **Khushwinder Kaur**\n\n"
            "🏢 Intern at Solitaire Infosystems\n\n"
            "💻 Interests: ML · Data Science · Web Dev"
        )
        st.divider()
        st.subheader("🎯 Project Goal")
        st.markdown(
            "Build an end-to-end spam detection pipeline — from raw CSV data "
            "to a trained ML model deployed as a web app. "
            "The project covers data cleaning, NLP preprocessing, model training, "
            "evaluation, and Streamlit deployment."
        )

    with col2:
        st.subheader("⚙️ ML Pipeline Steps")
        st.markdown(
            "1. **Load** SMS + Email CSV datasets\n"
            "2. **Merge** and remove duplicates\n"
            "3. **Clean text** — lowercase, remove URLs, numbers, punctuation\n"
            "4. **Vectorize** — TF-IDF with bigrams (15K features)\n"
            "5. **Train** — LinearSVC on 80% of data\n"
            "6. **Evaluate** — Accuracy, Precision, Recall, F1 on 20% test split\n"
            "7. **Save** model with `joblib` → load in this app"
        )
        st.divider()
        st.subheader("🚀 Future Improvements")
        st.markdown(
            "- Use BERT or other transformer models\n"
            "- Multi-language spam detection\n"
            "- Gmail / Outlook API integration\n"
            "- Docker deployment on cloud (AWS / GCP)"
        )

    st.divider()
    st.caption(
        f"Email Spam Detection System · Sudhanshu Ranjan & Khushwinder Kaur · "
        f"Solitaire Infosystems · {datetime.datetime.now().year}"
    )
