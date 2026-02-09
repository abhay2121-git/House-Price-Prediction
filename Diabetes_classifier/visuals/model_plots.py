"""
Model performance visualization utilities.
Creates plots for evaluating model performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os

# Set consistent style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def save_plot(fig, filename, output_dir='outputs/figures'):
    """
    Save a matplotlib figure to disk.
    
    Args:
        fig: Matplotlib figure object
        filename (str): Name of the output file
        output_dir (str): Directory to save the figure
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close(fig)


def plot_confusion_matrix(cm, save=True):
    """
    Plot the confusion matrix as a heatmap.
    
    Args:
        cm (np.array): Confusion matrix
        save (bool): Whether to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Labels for the matrix
    labels = ['Non-Diabetic', 'Diabetic']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                square=True, ax=ax, cbar_kws={'shrink': 0.8})
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, pad=20)
    
    plt.tight_layout()
    
    if save:
        save_plot(fig, 'confusion_matrix.png')
    else:
        plt.show()
    
    return fig


def plot_roc_curve(model, X_test, y_test, save=True):
    """
    Plot the ROC curve and calculate AUC.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): True test labels
        save (bool): Whether to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get prediction probabilities
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    ax.plot(fpr, tpr, color='#e74c3c', linewidth=2, 
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', 
            label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, pad=20)
    ax.legend(loc='lower right')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        save_plot(fig, 'roc_curve.png')
    else:
        plt.show()
    
    print(f"  AUC Score: {roc_auc:.4f}")
    
    return fig, roc_auc


def plot_precision_recall_curve(model, X_test, y_test, save=True):
    """
    Plot the precision-recall curve.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): True test labels
        save (bool): Whether to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get prediction probabilities
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    
    # Plot
    ax.plot(recall, precision, color='#3498db', linewidth=2)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, pad=20)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        save_plot(fig, 'precision_recall_curve.png')
    else:
        plt.show()
    
    return fig


def plot_feature_importance(importance_data, feature_names, save=True):
    """
    Plot feature importance as a horizontal bar chart.
    
    Args:
        importance_data (list): List of (feature, importance) tuples
        feature_names (list): List of feature names
        save (bool): Whether to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract features and importances
    features = [item[0] for item in importance_data]
    importances = [item[1] for item in importance_data]
    
    # Create horizontal bar plot
    y_pos = np.arange(len(features))
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    
    ax.barh(y_pos, importances, color=colors, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()  # Top to bottom
    ax.set_xlabel('Absolute Coefficient Value', fontsize=12)
    ax.set_title('Feature Importance (Logistic Regression)', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save:
        save_plot(fig, 'feature_importance.png')
    else:
        plt.show()
    
    return fig


def plot_metrics_comparison(metrics, save=True):
    """
    Plot a bar chart comparing key metrics.
    
    Args:
        metrics (dict): Dictionary containing evaluation metrics
        save (bool): Whether to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score']
    ]
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    bars = ax.bar(metric_names, metric_values, color=colors, edgecolor='black')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Metrics', fontsize=14, pad=20)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save:
        save_plot(fig, 'metrics_comparison.png')
    else:
        plt.show()
    
    return fig


def run_model_plots(model, X_test, y_test, metrics, importance_data, 
                   feature_names, output_dir='outputs/figures'):
    """
    Run all model performance plots.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): True test labels
        metrics (dict): Evaluation metrics dictionary
        importance_data (list): Feature importance data
        feature_names (list): List of feature names
        output_dir (str): Directory to save plots
    """
    print("\n" + "="*50)
    print("GENERATING MODEL PERFORMANCE PLOTS")
    print("="*50)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n1. Confusion matrix...")
    plot_confusion_matrix(metrics['confusion_matrix'], save=True)
    
    print("2. ROC curve...")
    plot_roc_curve(model, X_test, y_test, save=True)
    
    print("3. Precision-Recall curve...")
    plot_precision_recall_curve(model, X_test, y_test, save=True)
    
    print("4. Feature importance...")
    plot_feature_importance(importance_data, feature_names, save=True)
    
    print("5. Metrics comparison...")
    plot_metrics_comparison(metrics, save=True)
    
    print(f"\nAll model plots saved to: {output_dir}/")
    print("="*50)
