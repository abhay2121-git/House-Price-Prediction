"""
Model definition for the Diabetes Classifier project.
Uses Logistic Regression as the baseline classification model.
"""

from sklearn.linear_model import LogisticRegression


def create_model(random_state=42, max_iter=1000):
    """
    Create a Logistic Regression model for binary classification.
    
    Args:
        random_state (int): Random seed for reproducibility
        max_iter (int): Maximum number of iterations for convergence
        
    Returns:
        LogisticRegression: Configured logistic regression model
    """
    model = LogisticRegression(
        random_state=random_state,
        max_iter=max_iter,
        solver='lbfgs',  # Good for small datasets
        class_weight='balanced'  # Handle class imbalance
    )
    
    print("Model created: Logistic Regression")
    print(f"  - Solver: lbfgs")
    print(f"  - Max iterations: {max_iter}")
    print(f"  - Class weight: balanced")
    
    return model


def get_model_params():
    """
    Get default model parameters.
    
    Returns:
        dict: Dictionary of model parameters
    """
    return {
        'random_state': 42,
        'max_iter': 1000,
        'solver': 'lbfgs',
        'class_weight': 'balanced'
    }
