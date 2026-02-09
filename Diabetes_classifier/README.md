# Diabetes Classification - Machine Learning Project

A complete end-to-end machine learning project for predicting diabetes using Logistic Regression. This project demonstrates best practices for data preprocessing, exploratory data analysis, model training, evaluation, and visualization.

## Problem Statement

Diabetes is a chronic disease that affects millions of people worldwide. Early detection and classification of diabetes can help in timely intervention and better disease management. This project builds a classification model to predict whether a patient has diabetes based on diagnostic measurements.

## Dataset

The dataset contains the following features:

| Feature | Description |
|---------|-------------|
| **Pregnancies** | Number of times pregnant |
| **Glucose** | Plasma glucose concentration (mg/dL) |
| **BloodPressure** | Diastolic blood pressure (mm Hg) |
| **SkinThickness** | Triceps skin fold thickness (mm) |
| **Insulin** | 2-Hour serum insulin (mu U/ml) |
| **BMI** | Body mass index (weight in kg/(height in m)²) |
| **DiabetesPedigreeFunction** | Diabetes pedigree function (genetic score) |
| **Age** | Age in years |
| **Outcome** | Target variable (0 = Non-Diabetic, 1 = Diabetic) |

## Project Structure

```
Diabetes_Classifier/
│
├── data/
│   └── diabetes.csv                 # Dataset file
│
├── src/
│   ├── __init__.py                  # Package initialization
│   ├── data_loader.py               # Data loading utilities
│   ├── preprocessing.py             # Data preprocessing pipeline
│   ├── model.py                     # Model definition
│   ├── train.py                     # Training utilities
│   └── evaluate.py                  # Evaluation metrics
│
├── visuals/
│   ├── eda_plots.py                 # EDA visualization functions
│   └── model_plots.py               # Model performance plots
│
├── main.py                          # Main pipeline script
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

## Approach

### 1. Data Loading & Inspection
- Load dataset using pandas
- Inspect data structure, types, and missing values
- Check for data quality issues

### 2. Exploratory Data Analysis (EDA)
- Target variable distribution analysis
- Correlation matrix between features
- Feature distribution analysis by outcome
- Box plots to identify outliers
- Pair plots for key features

### 3. Data Preprocessing
- **Zero Value Handling**: In medical data, zero values in features like Glucose, BloodPressure, etc., often indicate missing data. These are replaced with NaN and filled using median imputation.
- **Train-Test Split**: 80% training, 20% testing with stratification
- **Feature Scaling**: StandardScaler for normalization (zero mean, unit variance)

### 4. Model Selection
**Logistic Regression** is used as the baseline model because:
- Simple and interpretable
- Well-suited for binary classification
- Provides probability estimates
- Fast training and prediction

### 5. Model Training
- Balanced class weights to handle imbalanced data
- L-BFGS solver for optimization
- Maximum 1000 iterations for convergence

### 6. Model Evaluation
Metrics computed on test set:
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual breakdown of predictions
- **ROC Curve & AUC**: Performance across different thresholds
- **Precision-Recall Curve**: Performance on imbalanced data

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd Diabetes_Classifier
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the complete ML pipeline:

```bash
python main.py
```

This will:
1. Load and inspect the data
2. Generate EDA visualizations (saved to `outputs/figures/`)
3. Preprocess the data (handle zeros, split, scale)
4. Train the Logistic Regression model
5. Evaluate model performance
6. Generate model performance plots
7. Save the trained model and scaler (to `models/`)

## Output Files

After running the pipeline, you will find:

- **EDA Plots** (`outputs/figures/`):
  - `target_distribution.png` - Class distribution
  - `correlation_matrix.png` - Feature correlations
  - `feature_distributions.png` - Distribution by outcome
  - `feature_boxplots.png` - Box plots for each feature
  - `pairplot.png` - Pairwise feature relationships

- **Model Performance Plots** (`outputs/figures/`):
  - `confusion_matrix.png` - Prediction breakdown
  - `roc_curve.png` - ROC curve with AUC score
  - `precision_recall_curve.png` - Precision vs Recall
  - `feature_importance.png` - Feature coefficients
  - `metrics_comparison.png` - Bar chart of all metrics

- **Saved Models** (`models/`):
  - `diabetes_model.joblib` - Trained Logistic Regression model
  - `scaler.joblib` - Fitted StandardScaler for preprocessing

## Model Performance

The model is evaluated using multiple metrics to ensure comprehensive performance assessment:

- **Accuracy**: Proportion of correctly classified instances
- **Precision**: Ability to avoid false positives
- **Recall**: Ability to find all positive instances (sensitivity)
- **F1-Score**: Balance between precision and recall
- **AUC-ROC**: Model's ability to distinguish between classes

## Key Findings

Based on feature importance analysis, the most influential factors for diabetes prediction typically include:
1. **Glucose** - Primary indicator of diabetes
2. **BMI** - Body mass correlation with diabetes risk
3. **Age** - Age-related diabetes risk
4. **DiabetesPedigreeFunction** - Genetic predisposition

## Future Improvements

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Cross-validation for more robust evaluation
- Try other algorithms (Random Forest, SVM, XGBoost)
- Feature engineering (polynomial features, interaction terms)
- SMOTE for handling class imbalance
- Model ensembling for improved performance

## Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **scikit-learn** - Machine learning algorithms
- **matplotlib** - Static plotting
- **seaborn** - Statistical data visualization
- **joblib** - Model serialization

## License

This project is open source and available for educational purposes.

## Author

Diabetes Classifier Team

---

**Note**: This model is for educational and demonstration purposes. It should not be used for actual medical diagnosis without proper validation and clinical expertise.
