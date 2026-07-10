# 🛡️ Email Spam Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-00C896?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-6C63FF?style=for-the-badge)

**A production-grade Email and SMS Spam Detection system built with Machine Learning, NLP, and a modern Streamlit dashboard.**

*Developers: Sudhanshu Ranjan & Khushwinder Kaur | Intern at Solitaire Infosystems*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Features](#-features)
- [Technology Stack](#️-technology-stack)
- [Datasets](#-datasets)
- [ML Pipeline](#️-ml-pipeline)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [App Pages](#-app-pages)
- [Results](#-results)
- [Future Scope](#-future-scope)

---

## 🌟 Overview

The **Email Spam Detection System** is an end-to-end machine learning project that classifies email and SMS messages as **Spam** or **Ham (Legitimate)**. It merges two real-world datasets, applies NLP preprocessing, trains a high-accuracy LinearSVC classifier with TF-IDF vectorization, and exposes all functionality through a sleek, dark-mode Streamlit dashboard.

This project is designed to serve as:
- ✅ A GitHub portfolio showcase
- ✅ A college mini-project / internship project
- ✅ A placement interview demonstration
- ✅ A learning reference for ML + NLP + Streamlit

---

## 📁 Project Structure

```
Email_Spam_Detection/
│
├── data/
│   ├── spam.csv                  # SMS Spam Collection Dataset (UCI)
│   └── spam_email_dataset.csv    # Email Spam Dataset
│
├── models/
│   ├── spam_model.pkl            # Trained LinearSVC + LabelEncoder + metadata
│   └── vectorizer.pkl            # Fitted TF-IDF Vectorizer
│
├── docs/
│   └── project_report.md         # Detailed project documentation
│
├── app.py                        # Streamlit dashboard application
├── train_model.py                # Model training pipeline script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## ✨ Features

### Machine Learning
- 🔀 **Merged dataset** – SMS Spam Collection + Email Spam Dataset
- 🧹 **Text preprocessing** – URL/email/number normalization, punctuation removal
- 🔢 **TF-IDF Vectorization** – 15,000 features, unigram + bigram
- 🤖 **LinearSVC classifier** – High-accuracy, fast, production-ready
- 📊 **Full evaluation** – Accuracy, Precision, Recall, F1, Confusion Matrix

### Streamlit Dashboard
- 🏠 **Home Page** – Hero section, stats, tech stack cards, accuracy bar
- 📊 **Data Visualization** – Spam/Ham bar chart, pie chart, word clouds, length distribution
- 📈 **Model Performance** – Metrics, classification report, confusion matrix heatmap
- 🔍 **Spam Detector** – Real-time prediction with confidence score, history, download
- ℹ️  **About Page** – Developer info, pipeline architecture, future scope

---

## 🛠️ Technology Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| ML / NLP | Scikit-Learn, NLTK, TF-IDF |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly, WordCloud |
| Web App | Streamlit |
| Serialization | Joblib |

---

## 📂 Datasets

### 1. SMS Spam Collection (UCI Machine Learning Repository)
- **Records**: 5,572 SMS messages
- **Format**: CSV with columns `v1` (label) and `v2` (text)
- **Classes**: `ham` / `spam`

### 2. Email Spam Dataset
- **Records**: 5,000 email records
- **Format**: CSV with `Subject`, `Body`, `Label`, and metadata columns
- **Text field**: `Subject + Body` merged into single text

---

## ⚙️ ML Pipeline

```
Raw Data (SMS + Email)
        │
        ▼
  Load & Merge Datasets
        │
        ▼
  Clean & Deduplicate
  (remove nulls, duplicates, normalize labels)
        │
        ▼
  Text Preprocessing
  (lowercase, remove URLs, emails, numbers, punctuation)
        │
        ▼
  Label Encoding
  (LabelEncoder: ham=0, spam=1)
        │
        ▼
  TF-IDF Vectorization
  (max_features=15000, ngram_range=(1,2), sublinear_tf=True)
        │
        ▼
  Train / Test Split
  (80% train, 20% test, stratified)
        │
        ▼
  LinearSVC Training
  (C=1.0, max_iter=2000)
        │
        ▼
  Evaluation
  (Accuracy, Precision, Recall, F1, Confusion Matrix)
        │
        ▼
  Save Artifacts
  (spam_model.pkl, vectorizer.pkl)
        │
        ▼
  Streamlit Dashboard
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Step 1: Clone the repository

```bash
git clone https://github.com/sudhanshu-ranjan/email-spam-detection.git
cd email-spam-detection/Email_Spam_Detection
```

### Step 2: Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the model

```bash
python train_model.py
```

This will:
- Load and merge both datasets
- Preprocess text and encode labels
- Train the LinearSVC model
- Save `models/spam_model.pkl` and `models/vectorizer.pkl`

### Step 5: Launch the Streamlit app

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 📖 Usage

### Training the Model

```bash
python train_model.py
```

Expected output:
```
=======================================================
  Email Spam Detection – Model Training Pipeline
  Developers : Sudhanshu Ranjan & Khushwinder Kaur | Solitaire Infosystems
=======================================================

[1/6] Loading SMS Spam Collection dataset …
      → 5,572 records loaded.
[1/6] Loading Email Spam dataset …
      → 5,000 records loaded.
[2/6] Merging datasets …
      → 10,572 → ~9,800 records after cleaning.
[3/6] Preprocessing text …
[4/6] Encoding labels & building TF-IDF features …
[5/6] Training LinearSVC model …
      Accuracy : 98.XX%
[6/6] Saving model artifacts …

✅  Training complete! Model ready for deployment.
```

### Running the App

```bash
streamlit run app.py
```

Navigate through the sidebar to explore:
- **Home** – Project overview and statistics
- **Data Visualization** – Interactive charts and word clouds
- **Model Performance** – Evaluation metrics and confusion matrix
- **Spam Detector** – Paste a message and get real-time prediction
- **About Project** – Full project documentation

---

## 📱 App Pages

| Page | Description |
|---|---|
| 🏠 Home | Hero section, dataset stats, tech stack, accuracy progress bar |
| 📊 Data Visualization | Bar chart, pie chart, word clouds, length distribution |
| 📈 Model Performance | Accuracy/Precision/Recall/F1 cards, classification report, confusion matrix |
| 🔍 Spam Detector | Text input, real-time prediction, confidence score, history |
| ℹ️ About Project | Developer bio, dataset info, pipeline, future scope |

---

## 📊 Results

| Metric | Score |
|---|---|
| ✅ Accuracy | ~98%+ |
| 🔬 Precision | ~98%+ |
| 📡 Recall | ~98%+ |
| ⚖️ F1-Score | ~98%+ |

*Exact scores depend on dataset state after deduplication.*

---

## 🔭 Future Scope

- [ ] Deploy on AWS / GCP / Heroku / Streamlit Cloud
- [ ] BERT / Transformer-based classification
- [ ] Multi-language spam detection
- [ ] Gmail / Outlook API integration
- [ ] Browser extension for live email protection
- [ ] Docker containerization + CI/CD pipeline
- [ ] Active learning for continuous model improvement
- [ ] Phishing URL detection component

---

## 👨‍💻 Developers

**Sudhanshu Ranjan** & **Khushwinder Kaur**  
Intern at Solitaire Infosystems  
GitHub: [@sudhanshu-ranjan](https://github.com/sudhanshu-ranjan)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
Made with ❤️ by <strong>Sudhanshu Ranjan & Khushwinder Kaur</strong> · Solitaire Infosystems · 2025
</div>
