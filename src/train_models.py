import os
import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from preprocessing import preprocess_pipeline

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluates the model and returns a dictionary of metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    # We focus on the positive class (Machine failure = 1)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    metrics = {
        "Model": model_name,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "Confusion Matrix": cm
    }
    return metrics

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "ai4i2020.csv")
    
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_pipeline(data_path)
    
    # Calculate scale_pos_weight for XGBoost
    neg_class = (y_train == 0).sum()
    pos_class = (y_train == 1).sum()
    scale_pos_weight = neg_class / pos_class
    
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        print(f"{name} F1-Score: {metrics['F1-Score']:.4f}")
        
    # Find the best model based on F1-Score
    best_result = max(results, key=lambda x: x['F1-Score'])
    best_model_name = best_result['Model']
    best_model = trained_models[best_model_name]
    
    print("\n" + "="*40)
    print(f"Best Model Selected: {best_model_name} (F1: {best_result['F1-Score']:.4f})")
    print("="*40)
    
    # Save artifacts
    models_dir = os.path.join(script_dir, "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(models_dir, "model.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    
    with open(os.path.join(models_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)
        
    # Save the evaluation results so we can show them in the app
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(results, f)
        
    print("Model, scaler, and metadata saved to 'models/' directory.")

if __name__ == "__main__":
    main()
