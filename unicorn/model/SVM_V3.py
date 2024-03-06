import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import re
from sklearn.impute import SimpleImputer

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def notch_filter(data, freq, fs, quality=30):
    b, a = iirnotch(freq / (0.5 * fs), quality)
    y = filtfilt(b, a, data)
    return y

def scale_signal(data, scale=50):
    """Scale data to a maximum absolute value."""
    return data * (scale / np.max(np.abs(data)))

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

def preprocess_eeg_data(data, fs=250):
    """Apply bandpass, notch filters, and scaling to EEG data."""
    eeg_columns = [col for col in data.columns if 'EEG' in col]
    for column in eeg_columns:
        # Bandpass filter
        data[column] = butter_bandpass_filter(data[column], 8, 12, fs)
        # Notch filter
        data[column] = notch_filter(data[column], 50, fs)
        # Scale the signal
        data[column] = scale_signal(data[column])
    return data

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
    """Reads, preprocesses, and combines data from CSV files."""
    data_frames = []
    for file in os.listdir(directory_path):
        if file.endswith(".csv"):
            file_path = os.path.join(directory_path, file)
            df = read_and_label(file_path)
            if not df.empty:
                df = preprocess_eeg_data(df)  # Preprocess EEG data
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

# Example Usage
# Adjust the directory path according to your dataset location
directory_path = '/home/suad/school/Unicorn Recorder/data'  # Change this to your directory path
X, y = read_and_process_data(directory_path)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the model
model = train_model(X_train, y_train)
evaluate_model(model, X_test, y_test)


