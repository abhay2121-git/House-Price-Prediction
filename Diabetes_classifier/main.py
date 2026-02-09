"""
Diabetes Classification - Main Pipeline

This script runs the complete ML pipeline for diabetes classification:
1. Load and inspect data
2. Perform EDA and generate visualizations
3. Preprocess data (handle zeros, split, scale)
4. Train Logistic Regression model
5. Evaluate model performance
6. Generate model performance plots
7. Save trained model and scaler

Usage:
    python main.py

The pipeline will:
- Load data from data/diabetes.csv
- Save EDA plots to outputs/figures/
- Save model performance plots to outputs/figures/
- Save trained model to models/diabetes_model.joblib
- Save scaler to models/scaler.joblib
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'visuals'))

import pandas as pd
import numpy as np

# Import project modules
from data_loader import load_data, inspect_data, get_feature_target_split
from preprocessing import preprocess_pipeline
from model import create_model
from train import train_model, save_model, save_scaler
from evaluate import evaluate_model, get_feature_importance
from eda_plots import run_eda_plots
from model_plots import run_model_plots


def main():
    """
    Main function that orchestrates the complete ML pipeline.
    """
    print("\n" + "="*60)
    print("DIABETES CLASSIFICATION - ML PIPELINE")
    print("="*60)
    print("\nStarting diabetes classification pipeline...\n")
    
    # Define paths
    DATA_PATH = 'data/diabetes.csv'
    MODEL_OUTPUT_PATH = 'models/diabetes_model.joblib'
    SCALER_OUTPUT_PATH = 'models/scaler.joblib'
    
    # ============================================================
    # STEP 1: LOAD DATA
    # ============================================================
    print("\n" + "-"*60)
    print("STEP 1: LOADING DATA")
    print("-"*60)
    
    try:
        df = load_data(DATA_PATH)
        data_info = inspect_data(df)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please ensure the diabetes.csv file exists in the data/ directory.")
        return
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # ============================================================
    # STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
    # ============================================================
    print("\n" + "-"*60)
    print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("-"*60)
    
    try:
        # Show plots interactively first, then save
        run_eda_plots(df, output_dir='outputs/figures')
        print("\nEDA plots displayed and saved!")
    except Exception as e:
        print(f"Warning: EDA plotting encountered an issue: {str(e)}")
        print("Continuing with pipeline...")
    
    # ============================================================
    # STEP 3: DATA PREPROCESSING
    # ============================================================
    print("\n" + "-"*60)
    print("STEP 3: DATA PREPROCESSING")
    print("-"*60)
    
    try:
        preprocessed_data = preprocess_pipeline(df, 
                                                target_column='Outcome',
                                                test_size=0.2,
                                                random_state=42)
        
        X_train = preprocessed_data['X_train']
        X_test = preprocessed_data['X_test']
        y_train = preprocessed_data['y_train']
        y_test = preprocessed_data['y_test']
        scaler = preprocessed_data['scaler']
        feature_names = preprocessed_data['feature_names']
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        return
    
    # ============================================================
    # STEP 4: MODEL TRAINING
    # ============================================================
    print("\n" + "-"*60)
    print("STEP 4: MODEL TRAINING")
    print("-"*60)
    
    try:
        # Create and train the model
        model = create_model(random_state=42)
        trained_model = train_model(X_train, y_train, model=model)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return
    
    # ============================================================
    # STEP 5: MODEL EVALUATION
    # ============================================================
    print("\n" + "-"*60)
    print("STEP 5: MODEL EVALUATION")
    print("-"*60)
    
    try:
        # Evaluate the model
        metrics = evaluate_model(trained_model, X_test, y_test)
        
        # Get feature importance
        importance_data = get_feature_importance(trained_model, feature_names)
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return
    
    # ============================================================
    # STEP 6: MODEL VISUALIZATIONS
    # ============================================================
    print("\n" + "-"*60)
    print("STEP 6: GENERATING MODEL VISUALIZATIONS")
    print("-"*60)
    
    try:
        run_model_plots(trained_model, X_test, y_test, metrics, 
                       importance_data, feature_names,
                       output_dir='outputs/figures')
    except Exception as e:
        print(f"Warning: Model plotting encountered an issue: {str(e)}")
        print("Continuing with pipeline...")
    
    # ============================================================
    # STEP 7: SAVE MODEL AND SCALER
    # ============================================================
    print("\n" + "-"*60)
    print("STEP 7: SAVING MODEL AND SCALER")
    print("-"*60)
    
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save model and scaler
        save_model(trained_model, MODEL_OUTPUT_PATH)
        save_scaler(scaler, SCALER_OUTPUT_PATH)
        
    except Exception as e:
        print(f"Error saving model/scaler: {str(e)}")
        return
    
    # ============================================================
    # PIPELINE COMPLETE
    # ============================================================
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nSummary:")
    print(f"  - Dataset: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")
    print(f"  - Model: Logistic Regression")
    print(f"\nPerformance Metrics:")
    print(f"  - Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall:    {metrics['recall']:.4f}")
    print(f"  - F1-Score:  {metrics['f1_score']:.4f}")
    print(f"\nOutput Files:")
    print(f"  - Model: {MODEL_OUTPUT_PATH}")
    print(f"  - Scaler: {SCALER_OUTPUT_PATH}")
    print(f"  - Plots: outputs/figures/")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
