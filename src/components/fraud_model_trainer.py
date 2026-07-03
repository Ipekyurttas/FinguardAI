import xgboost as xgb
import lightgbm as lgb
import optuna
from sklearn.metrics import precision_recall_curve, auc, classification_report
import pandas as pd
import numpy as np

def train_xgboost(X_train, y_train, params=None):
    """XGBoost modelini eğitir."""
    if params is None:
        params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6}
    model = xgb.XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train, params=None):
    """LightGBM modelini eğitir."""
    if params is None:
        params = {'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31}
    model = lgb.LGBMClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_fraud_model(model, X_test, y_test):
    """Precision-Recall ve AUPRC metriklerini hesaplar."""
    y_probs = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    auprc = auc(recall, precision)
    
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    
    return auprc, precision, recall, report

def optimize_with_optuna(X_train, y_train, X_val, y_val, model_type="xgb"):
    """Optuna kullanarak hiperparametre optimizasyonu yapar."""
    def objective(trial):
        if model_type == "xgb":
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0)
            }
            model = xgb.XGBClassifier(**param, random_state=42, eval_metric='logloss')
        else:
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            }
            model = lgb.LGBMClassifier(**param, random_state=42)
            
        model.fit(X_train, y_train)
        y_probs = model.predict_proba(X_val)[:, 1]
        precision, recall, _ = precision_recall_curve(y_val, y_probs)
        return auc(recall, precision) 

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    return study.best_params