from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
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

def normalize_data(X_train, X_test):
    """Normalizes the training and testing datasets."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Use the same scaler for test data
    return X_train_scaled, X_test_scaled

def train_model(X_train, y_train):
    """Trains the HistGradientBoostingClassifier."""
    # Note: With HistGradientBoostingClassifier, polynomial features and explicit normalization might not be necessary,
    # but are kept here for consistency with the original workflow. Depending on your dataset, you might experiment
    # with skipping these steps to simplify the pipeline.
    X_train_poly = feature_engineering(X_train)
    X_train_scaled, _ = normalize_data(X_train_poly, X_train_poly)  # Dummy call for scaler fit
    
    # HistGradientBoostingClassifier supports parallel training via n_jobs parameter
    model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=3, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    return model

def evaluate_model(model, X_test, y_test, scaler):
    """Evaluates the trained model on the test dataset."""
    X_test_poly = feature_engineering(X_test)
    X_test_scaled = scaler.transform(X_test_poly)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

# Example usage with separate training and testing paths
load_dotenv()
training_directory_path = os.getenv('FILES2')  # Training data directory
testing_directory_path = os.getenv('FILES')  # Testing data directory

# Process training data
X_train, y_train = read_and_process_data(training_directory_path)

# Process testing data
X_test, y_test = read_and_process_data(testing_directory_path)

# Train model
model = train_model(X_train, y_train)

# Normalize testing data and evaluate model
_, scaler = normalize_data(X_train, X_test)  # Obtain scaler from training data normalization
evaluate_model(model, X_test, y_test, scaler)
