import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
from scipy.signal import butter, sosfilt, sosfiltfilt, sosfreqz, butter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Function to apply baseline correction
def baseline_correction(df, columns=None):
    if columns is None:
        columns = df.columns
    for col in columns:
        if df[col].dtype in ['float64', 'float32', 'int']:
            baseline = np.min(df[col])
            df[col] -= baseline
    return df

# Function to read and label the EEG data
def read_and_label(file_path):
    label = re.sub('[0-9]*', '', os.path.basename(file_path).split('.csv')[0])
    try:
        df = pd.read_csv(file_path).iloc[100:]
        df = baseline_correction(df)
        df['Label'] = label
        return df
    except pd.errors.EmptyDataError:
        print(f"Error reading {file_path}: File is empty or does not exist.")
        return pd.DataFrame()

# Band-pass filter for EEG data using Second-Order Sections for stability
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data)  # Use sosfiltfilt for zero-phase filtering
    return y

# Function to read and process data from a directory, including filtering EEG data
def read_and_process_data(directory_path):
    data_frames = []
    fs = 250  # Sampling frequency for Unicorn headset
    lowcut = 8.0  # Low cut frequency for the band-pass filter
    highcut = 12.0  # High cut frequency for the band-pass filter
    
    for file in os.listdir(directory_path):
        if file.endswith(".csv"):
            file_path = os.path.join(directory_path, file)
            df = read_and_label(file_path)
            if not df.empty:
                # Apply band-pass filter to EEG channels
                for col in df.columns:
                    if 'EEG' in col:
                        df[col] = butter_bandpass_filter(df[col], lowcut, highcut, fs, order=4)
                data_frames.append(df)
    data = pd.concat(data_frames, ignore_index=True)
    X = data.drop(['Label'], axis=1)
    y = data['Label']
    X.fillna(X.mean(), inplace=True)
    return X, y

# Split and normalize data
def split_and_normalize_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Train the LDA model
def train_model(X_train_scaled, y_train):
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_scaled, y_train)
    return lda

# Evaluate the model with confusion matrix in percentages
def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy*100:.2f}%")
    cm = confusion_matrix(y_test, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    # print("Confusion Matrix (Percentages):")
    # print(cm_percentage)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix (Percentages)')
    plt.show()


# Example usage
directory_path = 'C:/Users/leohe/Documents/gtec/Unicorn Suite/Hybrid Black/Unicorn Recorder/test2/'
X, y = read_and_process_data(directory_path)
X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_normalize_data(X, y)
model = train_model(X_train_scaled, y_train)
evaluate_model(model, X_test_scaled, y_test)
