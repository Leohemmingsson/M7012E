import re
import os
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # No longer needed
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC  # Import the Support Vector Classifier

def read_and_label(file_path):
    label = os.path.basename(file_path).split('.csv')[0]
    label = re.sub('[0-9]*', '', label)
    df = pd.read_csv(file_path)
    df['Label'] = label
    return df

def read_and_process_data(directory_path):
    data = pd.DataFrame()
    for file in os.listdir(directory_path):
        if file.endswith(".csv"):
            file_path = os.path.join(directory_path, file)
            data = pd.concat([data, read_and_label(file_path)], ignore_index=True)
    X = data.drop(['Label'], axis=1)
    y = data['Label']
    X.fillna(X.mean(), inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    # Replace LDA with SVM
    svm = SVC(kernel='linear')  # You can adjust the kernel and other parameters as needed
    svm.fit(X_train, y_train)
    return svm

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy*100:.2f}%")

def read_unlabeled_file(file_path, scaler):
    label = os.path.basename(file_path).split('1.csv')[0]  # Extract label for reference
    df = pd.read_csv(file_path)
    X = df.fillna(df.mean())
    X_scaled = scaler.transform(X)  # Use the same scaler as the training data
    return X_scaled, label  # Return both scaled features and label

# Example usage
directory_path = 'C:/Users/leohe/Documents/gtec/Unicorn Suite/Hybrid Black/Unicorn Recorder/test/'
X, y, scaler = read_and_process_data(directory_path)
X_train, X_test, y_train, y_test = split_data(X, y)
model = train_model(X_train, y_train)
evaluate_model(model, X_test, y_test)

unlabeled_file_path = 'C:/Users/leohe/Documents/gtec/Unicorn Suite/Hybrid Black/Unicorn Recorder/test/test/Down4.csv'
X_unlabeled, unlabeled_file_label = read_unlabeled_file(unlabeled_file_path, scaler)
prediction = model.predict(X_unlabeled)

# Print the model's prediction and the actual label
print(f"Model's prediction: {prediction[0]}")
print(f"Actual label of the unlabeled file: {unlabeled_file_label}")
