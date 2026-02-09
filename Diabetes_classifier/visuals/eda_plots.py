"""
Exploratory Data Analysis (EDA) visualization utilities.
Creates plots for understanding the diabetes dataset.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set style for consistent plots
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


def plot_target_distribution(df, target_col='Outcome', save=True):
    """
    Plot the distribution of the target variable.
    
    Args:
        df (pd.DataFrame): Dataset
        target_col (str): Name of target column
        save (bool): Whether to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    counts = df[target_col].value_counts()
    labels = ['Non-Diabetic (0)', 'Diabetic (1)']
    colors = ['#3498db', '#e74c3c']
    
    ax.bar(labels, [counts[0], counts[1]], color=colors, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Diabetes Outcomes')
    ax.set_ylim(0, max(counts) * 1.1)
    
    # Add value labels on bars
    for i, v in enumerate([counts[0], counts[1]]):
        ax.text(i, v + max(counts)*0.02, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        save_plot(fig, 'target_distribution.png')
    else:
        plt.show()
    
    return fig


def plot_correlation_matrix(df, save=True):
    """
    Plot the correlation matrix of all features.
    
    Args:
        df (pd.DataFrame): Dataset
        save (bool): Whether to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    corr_matrix = df.corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, ax=ax, fmt='.2f', cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Matrix', fontsize=14, pad=20)
    
    plt.tight_layout()
    
    if save:
        save_plot(fig, 'correlation_matrix.png')
    else:
        plt.show()
    
    return fig


def plot_feature_distributions(df, save=True):
    """
    Plot distributions of all features by outcome.
    
    Args:
        df (pd.DataFrame): Dataset
        save (bool): Whether to save the plot
    """
    feature_cols = [col for col in df.columns if col != 'Outcome']
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(feature_cols):
        ax = axes[idx]
        
        # Plot distributions for both classes
        df[df['Outcome'] == 0][col].hist(ax=ax, alpha=0.7, bins=20, 
                                          label='Non-Diabetic', color='#3498db')
        df[df['Outcome'] == 1][col].hist(ax=ax, alpha=0.7, bins=20, 
                                          label='Diabetic', color='#e74c3c')
        
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {col}')
        ax.legend()
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save:
        save_plot(fig, 'feature_distributions.png')
    else:
        plt.show()
    
    return fig


def plot_boxplots(df, save=True):
    """
    Plot boxplots for all features by outcome.
    
    Args:
        df (pd.DataFrame): Dataset
        save (bool): Whether to save the plot
    """
    feature_cols = [col for col in df.columns if col != 'Outcome']
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(feature_cols):
        ax = axes[idx]
        
        sns.boxplot(data=df, x='Outcome', y=col, ax=ax, 
                   palette=['#3498db', '#e74c3c'])
        ax.set_xlabel('Outcome (0=Non-Diabetic, 1=Diabetic)')
        ax.set_ylabel(col)
        ax.set_title(f'{col} by Outcome')
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save:
        save_plot(fig, 'feature_boxplots.png')
    else:
        plt.show()
    
    return fig


def plot_pairplot(df, save=True):
    """
    Create pairplot of key features.
    
    Args:
        df (pd.DataFrame): Dataset
        save (bool): Whether to save the plot
    """
    # Select a subset of most important features for pairplot
    key_features = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'Outcome']
    
    g = sns.pairplot(df[key_features], hue='Outcome', 
                     palette=['#3498db', '#e74c3c'],
                     diag_kind='kde', plot_kws={'alpha': 0.6})
    
    g.fig.suptitle('Pairplot of Key Features', y=1.02, fontsize=14)
    
    if save:
        os.makedirs('outputs/figures', exist_ok=True)
        g.savefig('outputs/figures/pairplot.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: outputs/figures/pairplot.png")
        plt.close()
    else:
        plt.show()
    
    return g


def run_eda_plots(df, output_dir='outputs/figures'):
    """
    Run all EDA plots and save them.
    
    Args:
        df (pd.DataFrame): Dataset
        output_dir (str): Directory to save plots
    """
    print("\n" + "="*50)
    print("GENERATING EDA PLOTS")
    print("="*50)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n1. Target distribution...")
    plot_target_distribution(df, save=True)
    
    print("2. Correlation matrix...")
    plot_correlation_matrix(df, save=True)
    
    print("3. Feature distributions...")
    plot_feature_distributions(df, save=True)
    
    print("4. Feature boxplots...")
    plot_boxplots(df, save=True)
    
    print("5. Pairplot of key features...")
    plot_pairplot(df, save=True)
    
    print(f"\nAll EDA plots saved to: {output_dir}/")
    print("="*50)
