import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
from src.components.xai_explainer import calculate_shap_values, plot_shap_summary, analyze_errors

print("FinGuard AI: Model Yorumlama ve XAI Süreci Başlatıldı...")

df = pd.read_csv('data/processed/processed_fraud_data.csv')

target_col = 'is_fraud' 

if target_col not in df.columns:
    print(f"Hata: Veri setinde '{target_col}' sütunu bulunamadı!")
    print(f"Mevcut Sütunlar: {df.columns.tolist()}")
else:
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    model_path = "models/fraud_detection_model.txt" 
    
    try:
        import lightgbm as lgb
        model = lgb.Booster(model_file=model_path)
    except:
        model = joblib.load("models/fraud_model.pkl") 

    print("SHAP değerleri hesaplanıyor...")
    explainer, shap_values = calculate_shap_values(model, X)
    plot_shap_summary(shap_values, X)

    print("Hatalı tahminler analiz ediliyor...")
    fn, fp = analyze_errors(model, X, y)

    print(f"\n--- Hata Analizi Sonuçları ---")
    print(f"Toplam Test Örneği: {len(X)}")
    print(f"Atlanan Dolandırıcılık Sayısı (False Negatives): {len(fn)}")
    print(f"Hatalı Dolandırıcılık Uyarısı (False Positives): {len(fp)}")

    os.makedirs('results', exist_ok=True)
    if len(fn) > 0:
        fn.to_csv('results/false_negatives.csv', index=False)
        print("Atlanan vakalar 'results/false_negatives.csv' dosyasına kaydedildi.")

    print("\nÖrnek bir işlemin karar analizi yapılıyor...")
    plt.figure(figsize=(10, 6))
    
    shap_vals_obj = explainer(X)
    shap.plots.bar(shap_vals_obj[0], show=False)
    plt.tight_layout()
    plt.savefig('results/single_prediction_explanation.png')
    print("Tekil tahmin analizi 'results/single_prediction_explanation.png' olarak kaydedildi.")