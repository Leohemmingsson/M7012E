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
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def split_data(X, y):
    """Splits data into training and testing sets."""
    return train_test_split(X, y, test_size=0.2, random_state=42)

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

def read_unlabeled_file(file_path, scaler):
    """Processes an unlabeled file for prediction."""
    label = os.path.basename(file_path).split('1.csv')[0]  # Potentially adjust this label extraction method
    df = pd.read_csv(file_path)
    X = df.fillna(df.mean())
    X_scaled = scaler.transform(X)  # Use the same scaler as the training data
    return X_scaled, label  # Return both scaled features and label

# Example usage
directory_path = 'C:/Users/leohe/Documents/gtec/Unicorn Suite/Hybrid Black/Unicorn Recorder/test2/'
X, y, scaler = read_and_process_data(directory_path)
X_train, X_test, y_train, y_test = split_data(X, y)
model = train_model(X_train, y_train)
evaluate_model(model, X_test, y_test)

# unlabeled_file_path = 'C:/Users/leohe/Documents/gtec/Unicorn Suite/Hybrid Black/Unicorn Recorder/test/test/Down4.csv'
# X_unlabeled, unlabeled_file_label = read_unlabeled_file(unlabeled_file_path, scaler)
# prediction = model.predict(X_unlabeled)
# print(f"Model's prediction: {prediction[0]}")
# print(f"Actual label of the unlabeled file: {unlabeled_file_label}")
