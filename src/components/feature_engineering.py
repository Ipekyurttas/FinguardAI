import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

def encode_categorical(df, columns):
    """Kategorik değişkenleri Label Encoding ile sayısal hale getirir.""" 
    le = LabelEncoder()
    for col in columns:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    return df

def scale_features(X, save_path="models/scaler.pkl"):
    """Veriyi ölçeklendirir ve sütun isimlerini döndürür."""
    scaler = StandardScaler()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, save_path)
    return X_scaled, X.columns.tolist() 

def apply_smote(X, y):
    """Dengesiz veri setini SMOTE algoritması ile dengeler.""" 
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled