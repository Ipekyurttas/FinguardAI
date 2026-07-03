import pandas as pd
import numpy as np
import os
from src.components.feature_engineering import encode_categorical, scale_features, apply_smote

print("FinGuard AI: Ön İşleme Başlatıldı...")


df = pd.read_csv('data/raw/synthetic_fraud_dataset.csv') 
target_col = 'is_fraud'


drop_cols = ['transaction_id', 'user_id']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])


numeric_cols = ['amount', 'device_risk_score', 'ip_risk_score']
for col in ['device_risk_score', 'ip_risk_score']:
    if col in df.columns:
        noise = np.random.normal(1, 0.40, size=len(df)) 
        df[col] = df[col] * noise

riskli_kategoriler = [15, 25] 
for cat in riskli_kategoriler:
    cat_mask = (df['merchant_category'] == cat)
    df.loc[cat_mask & (np.random.rand(len(df)) < 0.35), target_col] = 1


categorical_cols = df.select_dtypes(include=['object']).columns
df = encode_categorical(df, categorical_cols)


high_amount_threshold = df['amount'].quantile(0.95)
heavy_fraud_idx = df[df['amount'] > high_amount_threshold].sample(frac=0.15).index
df.loc[heavy_fraud_idx, target_col] = 1 

night_region_mask = (df['hour'] < 6) | (df['country'] > 30)
df.loc[night_region_mask & (np.random.rand(len(df)) < 0.15), target_col] = 1

df['amount'] = np.log1p(df['amount'])


flip_idx = df.sample(frac=0.02, random_state=42).index
df.loc[flip_idx, target_col] = 1 - df.loc[flip_idx, target_col]


X = df.drop(target_col, axis=1) 
y = df[target_col]


X_scaled, feature_cols = scale_features(X, save_path="models/scaler.pkl")


try:
    X_balanced, y_balanced = apply_smote(X_scaled, y)
    print("✅ SMOTE ile dengeli ve karmaşık veri sağlandı.")
except Exception as e:
    print(f"⚠️ SMOTE atlandı: {e}")
    X_balanced, y_balanced = X_scaled, y


processed_df = pd.DataFrame(X_balanced, columns=feature_cols)
processed_df[target_col] = y_balanced

os.makedirs('data/processed', exist_ok=True)
processed_df.to_csv('data/processed/processed_fraud_data.csv', index=False)

print(f"Yeni Dağılım:\n{processed_df[target_col].value_counts()}")
print("İşlem Tamamlandı")