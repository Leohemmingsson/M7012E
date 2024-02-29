from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
import os
import re

def baseline_correction(df, columns=None):
    """Applies baseline correction to the DataFrame.
    Subtracts the minimum value of each column from itself.
    """
    if columns is None:
        columns = df.columns
    for col in columns:
        if df[col].dtype in ['float64', 'float32', 'int']:  # Apply only to numeric columns
            baseline = np.min(df[col])  # Simple baseline correction using minimum value
            df[col] -= baseline
    return df

def read_and_label(file_path):
    """Reads a CSV file, labels it based on its filename, and removes the first 100 rows."""
    label = re.sub('[0-9]*', '', os.path.basename(file_path).split('.csv')[0])
    try:
        df = pd.read_csv(file_path).iloc[100:]  # Skip first 100 rows
        df = baseline_correction(df)  # Apply baseline correction
        df['Label'] = label
        return df
    except pd.errors.EmptyDataError:
        print(f"Error reading {file_path}: File is empty or does not exist.")
        return pd.DataFrame()

def read_and_process_data(directory_path):
    """Reads and processes data from a directory of CSV files."""
    data_frames = []
    for file in os.listdir(directory_path):
        if file.endswith(".csv"):
            file_path = os.path.join(directory_path, file)
            df = read_and_label(file_path)
            if not df.empty:
                data_frames.append(df)
    data = pd.concat(data_frames, ignore_index=True)
    X = data.drop(['Label'], axis=1)
    y = data['Label']
    X.fillna(X.mean(), inplace=True)  # Handling missing values by replacing them with column means
    return X, y

def feature_engineering(X):
    """Applies polynomial features to the dataset."""
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    return X_poly

def normalize_data(X):
    """Normalizes the dataset."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def train_and_evaluate_model(X, y):
    """Trains the Gradient Boosting Classifier and evaluates it using cross-validation."""
    # Feature Engineering and Normalization
    X_poly = feature_engineering(X)
    X_scaled, _ = normalize_data(X_poly)
    
    # Model Initialization
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    
    # Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
    precision_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='precision_macro')
    recall_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='recall_macro')
    f1_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1_macro')
    
    print(f"Average Accuracy: {np.mean(accuracy_scores):.2f}")
    print(f"Average Precision: {np.mean(precision_scores):.2f}")
    print(f"Average Recall: {np.mean(recall_scores):.2f}")
    print(f"Average F1 Score: {np.mean(f1_scores):.2f}")

# Example usage remains the same with the exception of calling the updated `train_and_evaluate_model` function.
load_dotenv()
directory_path = os.getenv('FILES2')
X, y = read_and_process_data(directory_path)
train_and_evaluate_model(X, y)
