import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import joblib


from sklearn.svm import SVC 
import re
from sklearn.impute import SimpleImputer

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
    """Reads and processes data, including baseline correction."""
    data_frames = []
    for file in os.listdir(directory_path):
        if file.endswith(".csv"):
            file_path = os.path.join(directory_path, file)
            df = read_and_label(file_path)
            if not df.empty:
                # Apply baseline correction 
                df = baseline_correction(df) 

                data_frames.append(df)  
    data = pd.concat(data_frames, ignore_index=True)
    X = data.drop(['Label'], axis=1)
    y = data['Label']

    # Create the preprocessing pipeline
    preprocessing_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('ica', FastICA(n_components=10, random_state=42))
    ])

    # Apply preprocessing
    X_transformed = preprocessing_pipeline.fit_transform(X)

    return X_transformed, y  

def baseline_correction(df, columns=None):
    """Applies baseline correction to the DataFrame.
    Subtracts the minimum value of each column from itself.
    """
    if columns is None:
        columns = df.columns
    for col in columns:
        if df[col].dtype in ['float64', 'float32', 'int']:  # Apply only to numeric columns
            baseline = np.min(df[col]) 
            df[col] -= baseline
    return df

def train_model(X_train, y_train):
    """Trains a Support Vector Machine (SVM) model."""
    svm = SVC(kernel='rbf', C=1.0, random_state=42) 
    svm.fit(X_train, y_train)
    return svm

def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model using accuracy metric."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy*100:.2f}%")

# Example Usage (make sure this aligns with your data structure)
directory_path = '/home/suad/school/Unicorn Recorder/test2'  # Adjust your directory path
X, y = read_and_process_data(directory_path)

# Split after preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = train_model(X_train, y_train) 
evaluate_model(model, X_test, y_test) 


# Example usage of read_unlabeled_file for prediction
# unlabeled_file_path = 'C:/Users/leohe/Documents/gtec/Unicorn Suite/Hybrid Black/Unicorn Recorder/test/test/Down4.csv'
# X_unlabeled, unlabeled_file_label = read_unlabeled_file(unlabeled_file_path, scaler)
# prediction = model.predict(X_unlabeled)

# # Print the model's prediction and the actual label
# print(f"Model's prediction: {prediction[0]}")
# print(f"Actual label of the unlabeled file: {unlabeled_file_label}")

