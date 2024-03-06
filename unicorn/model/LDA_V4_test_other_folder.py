from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
from dotenv import load_dotenv

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

def read_and_label(file_path, label):
    """Reads a CSV file, labels it based on provided label, and removes the first 100 rows."""
    try:
        df = pd.read_csv(file_path).iloc[100:]  # Skip first 100 rows
        
        df = baseline_correction(df)  # Apply baseline correction
        df['Label'] = label
        return df
    except pd.errors.EmptyDataError:
        print(f"Error reading {file_path}: File is empty or does not exist.")
        return pd.DataFrame()

def read_and_process_data(train_directory_path, test_directory_path):
    """Reads and processes data from directories of CSV files for both training and testing."""
    train_frames = []
    test_frames = []

    # Process training data
    for file in os.listdir(train_directory_path):
        if file.endswith(".csv"):
            file_path = os.path.join(train_directory_path, file)
            label = re.sub('[0-9]*', '', os.path.basename(file_path).split('.csv')[0])
            df = read_and_label(file_path, label)
            if not df.empty:
                train_frames.append(df)

    # Process testing data
    for file in os.listdir(test_directory_path):
        if file.endswith(".csv"):
            file_path = os.path.join(test_directory_path, file)
            label = re.sub('[0-9]*', '', os.path.basename(file_path).split('.csv')[0])
            df = read_and_label(file_path, label)
            if not df.empty:
                test_frames.append(df)

    train_data = pd.concat(train_frames, ignore_index=True)
    test_data = pd.concat(test_frames, ignore_index=True)

    X_train = train_data.drop(['Label'], axis=1)
    y_train = train_data['Label']
    X_train.fillna(X_train.mean(), inplace=True)  # Handling missing values

    X_test = test_data.drop(['Label'], axis=1)
    y_test = test_data['Label']
    X_test.fillna(X_test.mean(), inplace=True)  # Handling missing values

    return X_train, y_train, X_test, y_test

def normalize_data(X_train, X_test):
    """Normalizes the training and testing data."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Use the same scaler for the test data
    return X_train_scaled, X_test_scaled, scaler

def train_model(X_train_scaled, y_train):
    """Trains a Linear Discriminant Analysis model."""
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_scaled, y_train)
    return lda

def evaluate_model(model, X_test_scaled, y_test):
    """Evaluates the trained model using accuracy metric and displays the confusion matrix in percentages."""
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy*100:.2f}%")
    
    cm = confusion_matrix(y_test, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix (Percentages)')
    plt.show()


# Example usage:
load_dotenv()
train_directory_path = os.getenv('FILES2')
test_directory_path = os.getenv('FILES')
X_train, y_train, X_test, y_test = read_and_process_data(train_directory_path, test_directory_path)
X_train_scaled, X_test_scaled, scaler = normalize_data(X_train, X_test)
model = train_model(X_train_scaled, y_train)
evaluate_model(model, X_test_scaled, y_test)
