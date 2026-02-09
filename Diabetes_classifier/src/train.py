"""
Model training utilities for the Diabetes Classifier project.
Handles model training and persistence.
"""

import joblib
import os
from model import create_model


def train_model(X_train, y_train, model=None):
    """
    Train the classification model on the training data.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        model: Model instance (if None, creates default LogisticRegression)
        
    Returns:
        trained_model: Fitted model instance
    """
    if model is None:
        model = create_model()
    
    print("\n" + "="*50)
    print("TRAINING MODEL")
    print("="*50)
    
    # Train the model
    model.fit(X_train, y_train)
    
    print(f"Training complete!")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Features: {X_train.shape[1]}")
    
    # Print model coefficients if available
    if hasattr(model, 'coef_'):
        print(f"\nModel coefficients shape: {model.coef_.shape}")
        print(f"Model intercept: {model.intercept_[0]:.4f}")
    
    print("="*50)
    
    return model


def save_model(model, filepath='models/diabetes_model.joblib'):
    """
    Save the trained model to disk using joblib.
    
    Args:
        model: Trained model to save
        filepath (str): Path to save the model
        
    Returns:
        str: Path where model was saved
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save the model
    joblib.dump(model, filepath)
    print(f"\nModel saved to: {filepath}")
    
    return filepath


def load_model(filepath='models/diabetes_model.joblib'):
    """
    Load a saved model from disk.
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        model: Loaded model instance
    """
    model = joblib.load(filepath)
    print(f"Model loaded from: {filepath}")
    return model


def save_scaler(scaler, filepath='models/scaler.joblib'):
    """
    Save the fitted scaler to disk.
    
    Args:
        scaler: Fitted scaler instance
        filepath (str): Path to save the scaler
        
    Returns:
        str: Path where scaler was saved
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(scaler, filepath)
    print(f"Scaler saved to: {filepath}")
    return filepath
