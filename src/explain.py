import shap
import numpy as np

def get_shap_explanation(model, X_instance, feature_names, top_n=3):
    """
    Computes SHAP values for a single prediction and returns the top contributing features.
    Works for Tree-based models (RandomForest, XGBoost).
    
    Args:
        model: Trained model (RF or XGBoost)
        X_instance: A single row of scaled features (1D or 2D array of shape 1, n_features)
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        dict: Top features and their SHAP values
        list: All shap values for the instance (for plotting if needed)
    """
    # Ensure X_instance is 2D
    if len(X_instance.shape) == 1:
        X_instance = X_instance.reshape(1, -1)
        
    # Use TreeExplainer for tree models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_instance)
    
    # SHAP values structure varies between RF and XGBoost
    # For RandomForest (binary classification), shap_values is a list of arrays (one per class). We want class 1 (failure)
    if isinstance(shap_values, list):
        instance_shap_values = shap_values[1][0] 
    else:
        # For XGBoost, it might just return a single array for binary classification
        # (Actually, depending on version, it might be 2D array of shape (n_samples, n_features))
        if len(shap_values.shape) == 3: # (n_samples, n_features, n_classes)
            instance_shap_values = shap_values[0, :, 1]
        elif len(shap_values.shape) == 2: # (n_samples, n_features)
            instance_shap_values = shap_values[0]
        else:
            instance_shap_values = shap_values
            
    # Pair feature names with their absolute SHAP values for sorting
    feature_impacts = []
    for i, name in enumerate(feature_names):
        val = instance_shap_values[i]
        feature_impacts.append((name, val, abs(val)))
        
    # Sort by absolute SHAP value in descending order
    feature_impacts.sort(key=lambda x: x[2], reverse=True)
    
    # Extract top N features
    top_features = {}
    for name, val, abs_val in feature_impacts[:top_n]:
        top_features[name] = round(float(val), 4)
        
    return top_features, instance_shap_values
