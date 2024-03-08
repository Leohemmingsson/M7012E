import pickle
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
        # df = pd.read_csv(file_path).iloc[50:]  # Skip first 100 rows
        df = pd.read_csv(file_path)
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

def _NOT_USED_baseline_correction(df, columns=None):
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
    data = data.drop(["Accelerometer X", "Accelerometer Y", "Accelerometer Z", "Gyroscope X", "Gyroscope Y", "Gyroscope Z", "Battery Level", "Counter", "Validation Indicator"], axis=1) 
    X = data.drop(['Label'], axis=1)
    y = data['Label']


    return X, y

def get_transformed_x_and_pipeline(X):
    # Create the preprocessing pipeline
    preprocessing_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('ica', FastICA(n_components=8, random_state=42))
    ])

    # Apply preprocessing
    X_transformed = preprocessing_pipeline.fit_transform(X)
    return X_transformed, preprocessing_pipeline

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
directory_path="C:/Users/leohe/Documents/gtec/Unicorn Suite/Hybrid Black/Unicorn Recorder/data/"
X, y = read_and_process_data(directory_path)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, preprocess_pipeline= get_transformed_x_and_pipeline(X_train)

X_test = preprocess_pipeline.transform(X_test)


# Train and evaluate the model
model = train_model(X_train, y_train)
evaluate_model(model, X_test, y_test)

y_pred = model.predict(X_test)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Assuming y_test and y_pred are already defined
# y_pred = model.predict(X_test)  # This should be done already in your evaluate_model function

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix using seaborn for a more visually appealing format
plt.figure(figsize=(10, 7))  # Adjust the size as needed
sns.set(font_scale=1.4)  # Adjust font size for readability
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=sorted(y_test.unique()), yticklabels=sorted(y_test.unique()))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

folder = ""

file_path = folder + 'svm.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(model, file)

file_path = folder + 'pipeline.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(preprocess_pipeline, file)

print("Saved model and pipeline")