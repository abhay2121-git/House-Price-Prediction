"""
Data loading utilities for the Diabetes Classifier project.
Handles loading and initial inspection of the diabetes dataset.
"""

import pandas as pd
import numpy as np


def load_data(filepath):
    """
    Load the diabetes dataset from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def inspect_data(df):
    """
    Perform initial data inspection and print summary statistics.
    
    Args:
        df (pd.DataFrame): Dataset to inspect
        
    Returns:
        dict: Dictionary containing data information
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    print("\n" + "="*50)
    print("DATA INSPECTION REPORT")
    print("="*50)
    print(f"Shape: {info['shape']}")
    print(f"\nColumns: {info['columns']}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nDuplicate rows: {info['duplicate_rows']}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nBasic Statistics:")
    print(df.describe())
    print("="*50)
    
    return info


def get_feature_target_split(df, target_column='Outcome'):
    """
    Split the dataframe into features (X) and target (y).
    
    Args:
        df (pd.DataFrame): Dataset
        target_column (str): Name of the target column
        
    Returns:
        tuple: (X, y) where X is features and y is target
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"\nFeature matrix X shape: {X.shape}")
    print(f"Target vector y shape: {y.shape}")
    print(f"Class distribution:\n{y.value_counts()}")
    
    return X, y
