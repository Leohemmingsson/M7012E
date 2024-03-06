import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle
import re

# Function to read and label data based on file name
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

# Function to apply baseline correction
def baseline_correction(df, columns=None):
    """Applies baseline correction to the DataFrame."""
    if columns is None:
        columns = df.columns
    for col in columns:
        if df[col].dtype in ['float64', 'float32', 'int']:  # Apply only to numeric columns
            baseline = np.min(df[col])
            df[col] -= baseline
    return df

# Function to read and process data from a directory
def read_and_process_data(directory_path):
    """Reads and processes data from a directory."""
    data_frames = []
    for file in os.listdir(directory_path):
        if file.endswith(".csv"):
            file_path = os.path.join(directory_path, file)
            df = read_and_label(file_path)
            if not df.empty:
                df = baseline_correction(df)
                data_frames.append(df)
    data = pd.concat(data_frames, ignore_index=True)
    X = data.drop(['Label'], axis=1)
    y = data['Label']

    preprocessing_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('ica', FastICA(n_components=10, random_state=42))
    ])

    X_transformed = preprocessing_pipeline.fit_transform(X)
    return X_transformed, y

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model using accuracy metric."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy*100:.2f}%")

# Load the pre-trained model
model_filename = 'svm_model.pkl'
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Path to your previous data
previous_data_directory_path = '/home/suad/school/Unicorn Recorder/test'

# Preprocess the new data
X_new, y_new = read_and_process_data(previous_data_directory_path)

# Evaluate the model on the new data
evaluate_model(loaded_model, X_new, y_new)
