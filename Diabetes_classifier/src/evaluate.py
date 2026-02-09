"""
Model evaluation utilities for the Diabetes Classifier project.
Provides comprehensive evaluation metrics and reporting.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): True test labels
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred
    }
    
    # Print evaluation report
    print("\n" + "="*50)
    print("MODEL EVALUATION REPORT")
    print("="*50)
    
    print(f"\nTest samples: {len(y_test)}")
    print(f"\nMetrics:")
    print(f"  - Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall:    {metrics['recall']:.4f}")
    print(f"  - F1-Score:  {metrics['f1_score']:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"                 Predicted")
    print(f"                 0      1")
    print(f"  Actual  0  [{cm[0,0]:4d}]  [{cm[0,1]:4d}]  (Non-Diabetic)")
    print(f"          1  [{cm[1,0]:4d}]  [{cm[1,1]:4d}]  (Diabetic)")
    
    # Calculate specificity and sensitivity
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\nAdditional Metrics:")
    print(f"  - Sensitivity (Recall): {sensitivity:.4f}")
    print(f"  - Specificity:          {specificity:.4f}")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Non-Diabetic', 'Diabetic'],
                               zero_division=0))
    
    print("="*50)
    
    return metrics


def get_feature_importance(model, feature_names):
    """
    Extract feature importance from the model coefficients.
    
    Args:
        model: Trained Logistic Regression model
        feature_names (list): List of feature names
        
    Returns:
        list: List of (feature_name, importance) tuples sorted by importance
    """
    if hasattr(model, 'coef_'):
        coefficients = model.coef_[0]
        importance = list(zip(feature_names, np.abs(coefficients)))
        importance.sort(key=lambda x: x[1], reverse=True)
        
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE (by coefficient magnitude)")
        print("="*50)
        for feature, imp in importance:
            print(f"  {feature:20s}: {imp:.4f}")
        print("="*50)
        
        return importance
    else:
        print("Model does not have coefficients attribute")
        return []
