"""
Data preprocessing utilities for the Diabetes Classifier project.
Handles missing values, zero value imputation, and feature scaling.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def handle_zero_values(df, zero_sensitive_columns):
    """
    Handle zero values that represent missing data in medical measurements.
    Replaces zeros with NaN, then fills with median values.
    
    Args:
        df (pd.DataFrame): Input dataframe
        zero_sensitive_columns (list): Columns where 0 indicates missing data
        
    Returns:
        pd.DataFrame: Dataframe with zero values handled
    """
    df_processed = df.copy()
    
    print("\n" + "="*50)
    print("HANDLING ZERO VALUES")
    print("="*50)
    
    for col in zero_sensitive_columns:
        if col in df_processed.columns:
            zero_count = (df_processed[col] == 0).sum()
            zero_percentage = (zero_count / len(df_processed)) * 100
            
            print(f"\n{col}: {zero_count} zeros ({zero_percentage:.2f}%)")
            
            if zero_count > 0:
                # Replace 0 with NaN
                df_processed[col] = df_processed[col].replace(0, np.nan)
                # Fill with median
                median_value = df_processed[col].median()
                df_processed[col] = df_processed[col].fillna(median_value)
                print(f"  -> Filled with median: {median_value:.2f}")
    
    print("="*50)
    return df_processed


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        test_size (float): Proportion of test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nData split complete:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler (zero mean, unit variance).
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    
    # Fit on training data and transform both sets
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print("\nFeature scaling complete (StandardScaler)")
    
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(df, target_column='Outcome', test_size=0.2, random_state=42):
    """
    Complete preprocessing pipeline: handle zeros, split, and scale.
    
    Args:
        df (pd.DataFrame): Raw dataset
        target_column (str): Name of target column
        test_size (float): Proportion for test set
        random_state (int): Random seed
        
    Returns:
        dict: Dictionary with all preprocessed data and scaler
    """
    # Columns where 0 indicates missing data (medical measurements)
    zero_sensitive_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 
                           'Insulin', 'BMI']
    
    print("\nStarting preprocessing pipeline...")
    
    # Step 1: Handle zero values
    df_clean = handle_zero_values(df, zero_sensitive_cols)
    
    # Step 2: Split features and target
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]
    
    # Step 3: Train-test split
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    
    # Step 4: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    print("\nPreprocessing complete!")
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': list(X.columns)
    }
