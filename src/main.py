import os
import joblib
import torch
import numpy as np
import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

app = FastAPI(title="FinGuard AI - Stable API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FRAUD_MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_detection_model.txt")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_columns.pkl")
SENTIMENT_MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model")

fraud_model = None
scaler = None
feature_columns = None
sentiment_model = None
sentiment_tokenizer = None

try:
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURES_PATH)

    fraud_model = lgb.Booster(model_file=FRAUD_MODEL_PATH)

    sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_PATH)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH)

    print("✅ Tüm modeller yüklendi")

except Exception as e:
    print(f"❌ Yükleme hatası: {e}")


class SentimentRequest(BaseModel):
    text: str

class FraudRequest(BaseModel):
    features: list


@app.post("/predict/sentiment")
async def predict_sentiment(request: SentimentRequest):
    inputs = sentiment_tokenizer(
        request.text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][prediction].item()

    labels = {0: "positive", 1: "negative", 2: "neutral"}

    return {
        "sentiment": labels[prediction],
        "confidence": round(confidence, 4)
    }


@app.post("/predict/fraud")
async def predict_fraud(request: FraudRequest):
    try:
        df_input = pd.DataFrame([request.features], columns=feature_columns)

        scaled_data = scaler.transform(df_input)

        prob = float(fraud_model.predict(scaled_data)[0])
        
        prediction = int(prob > 0.5)

        return {
            "is_fraud": prediction,
            "fraud_probability": round(prob, 4),
            "risk_percentage": round(prob * 100, 2),
            "status": "High Risk" if prediction else "Safe"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)