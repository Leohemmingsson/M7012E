import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import FastICA  # Import FastICA
import re

def read_and_label(file_path):
    """Reads a CSV file, labels it based on its filename, and removes the first 100 rows."""
    label = re.sub('[0-9]*', '', os.path.basename(file_path).split('.csv')[0])
    try:
        df = pd.read_csv(file_path).iloc[100:]  # Skip first 100 rows
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
    
    # Check and handle NaN values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace Inf with NaN
    data.fillna(data.mean(), inplace=True)  # Fill NaN with the mean of each column
    
    X = data.drop(['Label'], axis=1)
    y = data['Label']
    return X, y

def normalize_and_apply_ica(X):
    """Normalizes the data and applies ICA."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Check if any NaN or Inf values were introduced during scaling (shouldn't happen, but good to check)
    assert not np.isnan(X_scaled).any(), "NaN values found in scaled data"
    assert not np.isinf(X_scaled).any(), "Inf values found in scaled data"
    
    # Apply ICA
    ica = FastICA(n_components=min(X.shape[1], X.shape[0] - 1), random_state=42)  # Adjust n_components as necessary
    X_ica = ica.fit_transform(X_scaled)
    
    return X_ica, scaler, ica

def split_data(X_ica, y):
    """Splits data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(X_ica, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Trains a Linear Discriminant Analysis model."""
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    return lda

def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model using accuracy metric."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy*100:.2f}%")

# Example usage
directory_path = 'C:/Users/leohe/Documents/gtec/Unicorn Suite/Hybrid Black/Unicorn Recorder/test2/'
X, y = read_and_process_data(directory_path)
X_ica, scaler, ica = normalize_and_apply_ica(X)
X_train, X_test, y_train, y_test = split_data(X_ica, y)
model = train_model(X_train, y_train)
evaluate_model(model, X_test, y_test)
