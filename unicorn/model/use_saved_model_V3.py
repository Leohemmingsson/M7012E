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

def preprocess_eeg_data(data, fs=250):
    eeg_columns = [col for col in data.columns if 'EEG' in col]
    for column in eeg_columns:
        data[column] = butter_bandpass_filter(data[column], 8, 12, fs)
        data[column] = notch_filter(data[column], 50, fs)
        data[column] = scale_signal(data[column])
    return data

def read_and_process_file(values):
    df = pd.DataFrame(values[1:], columns=values[0], dtype=float)
    # df = df.iloc[50:]
    if not df.empty:
        df = preprocess_eeg_data(df)

    df = df.drop(["Accelerometer X", "Accelerometer Y", "Accelerometer Z", "Gyroscope X", "Gyroscope Y", "Gyroscope Z", "Battery Level", "Counter", "Validation Indicator"], axis=1) 
    return df

def get_highest_occurrence(arr):
    unique, counts = np.unique(arr, return_counts=True)
    return unique[np.argmax(counts)]
