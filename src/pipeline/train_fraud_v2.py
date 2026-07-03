import pandas as pd
import lightgbm as lgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

print("Model Eğitimi Başlatıldı...")


df = pd.read_csv('data/processed/processed_fraud_data.csv')
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = lgb.LGBMClassifier(
    n_estimators=100,      
    learning_rate=0.03,
    max_depth=3,           
    feature_fraction=0.3,                    
    min_data_in_leaf=200,   
    reg_alpha=300.0,        
    reg_lambda=300.0,       
    random_state=42
)

model.fit(X_train, y_train)

y_probs = model.predict_proba(X_test)[:, 1]
y_pred = (y_probs > 0.5).astype(int)

print("\n--- Performans Raporu ---")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_probs):.4f}")
print(f"PR-AUC (AUPRC): {average_precision_score(y_test, y_probs):.4f}")


os.makedirs("models", exist_ok=True)
model.booster_.save_model("models/fraud_detection_model.txt")
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

print("\nModel ve feature_columns.pkl başarıyla kaydedildi.")