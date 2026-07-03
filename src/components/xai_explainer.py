import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def calculate_shap_values(model, X_train):
    """Model için SHAP değerlerini hesaplar ve özet grafiği çizer."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    return explainer, shap_values

def plot_shap_summary(shap_values, X_train, save_path='results/shap_summary.png'):
    """Özelliklerin önem düzeyini gösteren SHAP özet grafiğini kaydeder."""
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, show=False)
    plt.savefig(save_path)
    plt.close()
    print(f"SHAP özet grafiği kaydedildi: {save_path}")

def analyze_errors(model, X_test, y_test):
    """Hatalı tahminleri (False Positives ve False Negatives) ayıklar."""
    y_pred = model.predict(X_test)
    test_results = X_test.copy()
    test_results['actual'] = y_test
    test_results['predicted'] = y_pred
    
    false_negatives = test_results[(test_results['actual'] == 1) & (test_results['predicted'] == 0)]
    
    false_positives = test_results[(test_results['actual'] == 0) & (test_results['predicted'] == 1)]
    
    return false_negatives, false_positives