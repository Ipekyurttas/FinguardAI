# 🛡️ FinGuard AI

FinGuard AI is an AI-powered financial risk analysis platform that combines **Fraud Detection** and **Financial News Sentiment Analysis** in a single application.

The project consists of:

- ⚡ FastAPI Backend
- 🎨 Streamlit Dashboard
- 🤖 Machine Learning Models
- 📰 Financial News Sentiment Analysis using Transformers
- 💳 Credit Card Fraud Detection using LightGBM

---

# Features

## 📰 Financial News Sentiment Analysis

- Analyze financial news sentiment.
- Predict whether news is:
  - Positive
  - Negative
  - Neutral
- Uses a Transformer-based NLP model.
- Displays prediction confidence.

---

## 💳 Fraud Detection

Predicts whether a transaction is fraudulent using:

- Transaction Amount
- Transaction Type
- Merchant Category
- Country
- Transaction Time
- Device Risk Score
- IP Risk Score

Returns

- Fraud Probability
- Risk Percentage
- Safe / High Risk Decision

---

# Technologies

## Backend

- FastAPI
- Python
- Uvicorn

## Machine Learning

- LightGBM
- Hugging Face Transformers
- PyTorch
- Scikit-Learn
- Joblib
- Pandas
- NumPy

## Frontend

- Streamlit

---

# Project Structure

```
FinGuardAI
│
├── backend
│   └── main.py
│
├── models
│   ├── fraud_detection_model.txt
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   └── sentiment_model/
│
├── results/
│
├── frontend
│   └── streamlit_app.py
│
├── requirements.txt
│
└── README.md
```

---

# API Endpoints

## Sentiment Prediction

```
POST /predict/sentiment
```

Request

```json
{
    "text":"Apple shares increased after earnings."
}
```

Response

```json
{
    "sentiment":"positive",
    "confidence":0.98
}
```

---

## Fraud Prediction

```
POST /predict/fraud
```

Request

```json
{
  "features":[
      250,
      0,
      15,
      1,
      14,
      25,
      10
  ]
}
```

Response

```json
{
    "is_fraud":0,
    "fraud_probability":0.08,
    "risk_percentage":8.21,
    "status":"Safe"
}
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/Ipekyurttas/FinguardAI.git
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate

Windows

```bash
.venv\Scripts\activate
```

macOS/Linux

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Run Backend

```bash
uvicorn backend.main:app --reload
```

---

# Run Streamlit

```bash
streamlit run frontend/streamlit_app.py
```

---

# Model Files

Large trained model files are **not included** in this repository because they exceed GitHub's file size limits.

Place the trained models inside the following directory before running the project:

```
models/
```

Required files

- fraud_detection_model.txt
- scaler.pkl
- feature_columns.pkl
- sentiment_model/

---

# Future Improvements

- Real-time market monitoring
- Explainable AI (SHAP)
- User authentication
- Model retraining pipeline
- Docker deployment
- Cloud deployment (AWS/Azure)

---

# Author

**İpek Nur Yurttaş**

Software Engineer

GitHub:
https://github.com/Ipekyurttas
