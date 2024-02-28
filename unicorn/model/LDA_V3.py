import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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
    X = data.drop(['Label'], axis=1)
    y = data['Label']
    X.fillna(X.mean(), inplace=True)
    return X, y

def normalize_data(X_train, X_test):
    """Normalizes the training data and applies the transformation to the test data."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Use the same scaler for the test data
    return X_train_scaled, X_test_scaled, scaler

def split_and_normalize_data(X, y):
    """Splits data into training and testing sets and normalizes them using the training set's parameters."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled, scaler = normalize_data(X_train, X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train_scaled, y_train):
    """Trains a Linear Discriminant Analysis model."""
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_scaled, y_train)
    return lda

def evaluate_model(model, X_test_scaled, y_test):
    """Evaluates the trained model using accuracy metric."""
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy*100:.2f}%")

# Adjust your example usage accordingly
directory_path = 'C:/Users/leohe/Documents/gtec/Unicorn Suite/Hybrid Black/Unicorn Recorder/test2/'
X, y = read_and_process_data(directory_path)
X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_normalize_data(X, y)
model = train_model(X_train_scaled, y_train)
evaluate_model(model, X_test_scaled, y_test)
