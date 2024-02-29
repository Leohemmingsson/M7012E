from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
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

def normalize_data(X_train, X_test):
    """Normalizes the training data and applies the transformation to the test data."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Use the same scaler for the test data
    return X_train_scaled, X_test_scaled, scaler

def split_and_normalize_data(X, y):
    """Splits data into training and testing sets, normalizes them,
    and further splits the training data into overlapping subsets."""
    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalizing data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Splitting training data into overlapping subsets
    n_samples = X_train_scaled.shape[0]
    indices_1 = np.arange(0, n_samples // 2)
    indices_2 = np.arange(n_samples // 4, 3 * n_samples // 4)
    indices_3 = np.arange(n_samples // 2, n_samples)
    
    # Creating the subsets
    X_train_scaled_1 = X_train_scaled[indices_1]
    y_train_1 = y_train.iloc[indices_1]
    
    X_train_scaled_2 = X_train_scaled[indices_2]
    y_train_2 = y_train.iloc[indices_2]
    
    X_train_scaled_3 = X_train_scaled[indices_3]
    y_train_3 = y_train.iloc[indices_3]
    
    # Combine the subsets for training
    X_train_scaled_combined = np.concatenate((X_train_scaled_1, X_train_scaled_2, X_train_scaled_3), axis=0)
    y_train_combined = pd.concat([y_train_1, y_train_2, y_train_3], axis=0)
    
    return X_train_scaled_combined, X_test_scaled, y_train_combined, y_test, scaler


def train_model(X_train_scaled, y_train):
    """Trains a Linear Discriminant Analysis model."""
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_scaled, y_train)
    return lda

# def evaluate_model(model, X_test_scaled, y_test):
#     """Evaluates the trained model using accuracy metric."""
#     y_pred = model.predict(X_test_scaled)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Model accuracy: {accuracy*100:.2f}%")



def evaluate_model(model, X_test_scaled, y_test):
    """Evaluates the trained model using accuracy metric and displays the confusion matrix in percentages."""
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy*100:.2f}%")
    
    # Calculating the confusion matrix and normalizing it to show percentages
    cm = confusion_matrix(y_test, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Convert counts to percentages
    
    # print("Confusion Matrix (Percentages):")
    # print(cm_percentage)
    
    # Optionally, display the confusion matrix using Seaborn for a more visual representation
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix (Percentages)')
    plt.show()


# Example usage:
load_dotenv()
directory_path = os.getenv('FILES2')
X, y = read_and_process_data(directory_path)
X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_normalize_data(X, y)
model = train_model(X_train_scaled, y_train)
evaluate_model(model, X_test_scaled, y_test)
