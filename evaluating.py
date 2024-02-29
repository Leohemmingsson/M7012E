import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import re
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 
import joblib
# --- Data Reading and Preprocessing ---

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

def baseline_correction(df, columns=None):
    """Applies baseline correction to the DataFrame."""
    if columns is None:
        columns = df.columns
    for col in columns:
        if df[col].dtype in ['float64', 'float32', 'int']:  # Apply only to numeric columns
            baseline = np.min(df[col]) 
            df[col] -= baseline
    return df

def read_and_process_data(directory_path):
    """Reads and processes data, including baseline correction."""
    data_frames = []
    for file in os.listdir(directory_path):
        if file.endswith(".csv"):
            file_path = os.path.join(directory_path, file)
            df = read_and_label(file_path)
            if not df.empty:
                df = baseline_correction(df) 
                X = df.drop('Label', axis=1)
                y = df['Label']
                data_frames.append(df)  
    data = pd.concat(data_frames, ignore_index=True)
    X = data.drop(['Label'], axis=1)
    y = data['Label']

    # Create a new preprocessing pipeline
    preprocessing_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()) 
    ])

    X_transformed = preprocessing_pipeline.fit_transform(X) 

    # Select the first 10 columns for evaluation
    X_transformed = X_transformed[:, :10]  

    return X_transformed, y


# --- Model Loading and Evaluation ---

# Load the model
model = joblib.load('my_trained_model.joblib') 

# Evaluate on the new test data
test_directory = 'C:/Users/leohe/Documents/gtec/Unicorn Suite/Hybrid Black/Unicorn Recorder/test2/'
X_test, y_test = read_and_process_data(test_directory)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy on test data: {accuracy*100:.2f}%")
