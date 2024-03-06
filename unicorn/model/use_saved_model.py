import pickle
import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import re

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
    return data * (scale / np.max(np.abs(data)))

def read_and_label(file_path):
    label = re.sub('[0-9]*', '', os.path.basename(file_path).split('.csv')[0])
    try:
        df = pd.read_csv(file_path).iloc[100:]  # Skip first 100 rows
        df['Label'] = label
        return df
    except pd.errors.EmptyDataError:
        print(f"Error reading {file_path}: File is empty or does not exist.")
        return pd.DataFrame()

def preprocess_eeg_data(data, fs=250):
    eeg_columns = [col for col in data.columns if 'EEG' in col]
    for column in eeg_columns:
        # Bandpass filter
        data[column] = butter_bandpass_filter(data[column], 8, 12, fs)
        # Notch filter
        data[column] = notch_filter(data[column], 50, fs)
        # Scale the signal
        data[column] = scale_signal(data[column])
    return data

def read_and_process_file(file_path):
    df = read_and_label(file_path)
    if not df.empty:
        df = preprocess_eeg_data(df)  # Preprocess EEG data
    X = df.drop(['Label'], axis=1)

    return X, df['Label']

# Adjust the file_path to the specific file you want to process
file_path = "C:/Users/leohe/Documents/gtec/Unicorn Suite/Hybrid Black/Unicorn Recorder/data/down5.csv"
X, true_label= read_and_process_file(file_path)


folder = ""
file_path= folder +'svm.pkl'
with open(file_path, 'rb') as file:
    loaded_model = pickle.load(file)

file_path= folder +'pipeline.pkl'
with open(file_path, 'rb') as file:
    preprocess_pipeline = pickle.load(file)

X_transformed = preprocess_pipeline.transform(X)

y_pred = loaded_model.predict(X_transformed)
print("Predicted labels:", y_pred)
