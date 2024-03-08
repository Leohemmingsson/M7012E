import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import re
import pickle
import os

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
        data[column] = butter_bandpass_filter(data[column], 8, 12, fs)
        data[column] = notch_filter(data[column], 50, fs)
        data[column] = scale_signal(data[column])
    return data

def read_and_process_file(file_path):
    df = read_and_label(file_path)
    if not df.empty:
        df = preprocess_eeg_data(df)
    df = df.drop(["Accelerometer X", "Accelerometer Y", "Accelerometer Z", "Gyroscope X", "Gyroscope Y", "Gyroscope Z", "Battery Level", "Counter", "Validation Indicator"], axis=1) 
    X = df.drop(['Label'], axis=1)
    return X, df['Label']

def get_highest_occurrence(arr):
    unique, counts = np.unique(arr, return_counts=True)
    return unique[np.argmax(counts)]

# Adjust the file_path to the specific file you want to process
file_path = "C:/Users/leohe/Documents/gtec/Unicorn Suite/Hybrid Black/Unicorn Recorder/data/test/dummy10.csv"
X, true_label = read_and_process_file(file_path)

folder = ""
svm_model_path = folder +'svm.pkl'
with open(svm_model_path, 'rb') as file:
    loaded_svm_model = pickle.load(file)

pipeline_path = folder +'pipeline.pkl'
with open(pipeline_path, 'rb') as file:
    preprocess_pipeline = pickle.load(file)

X_transformed = preprocess_pipeline.transform(X)
y_pred = loaded_svm_model.predict(X_transformed)

y_pred = get_highest_occurrence(y_pred)
true_label = get_highest_occurrence(true_label)


print("Predicted labels:", y_pred)
print("True label:", true_label)
