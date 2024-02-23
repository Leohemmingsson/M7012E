import re
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC  # Support Vector Classifier import statement


def read_label(file_path):
    """Read and label dataset based on file name."""
    print(f"Processing {file_path}...")
    label = os.path.basename(file_path).split(".csv")[0]
    label = re.sub("[0-9]*", "", label)  # Remove digits
    df = pd.read_csv(file_path)
    df["Label"] = label
    return df


def process_data(directory_path):
    """Process all CSV files in the specified directory."""
    print("Processing data...")
    combined_data = pd.DataFrame()
    for file in os.listdir(directory_path):
        if file.endswith(".csv"):
            data_path = os.path.join(directory_path, file)
            combined_data = pd.concat(
                [combined_data, read_label(data_path)], ignore_index=True
            )
    print("Data processing complete.")
    X = combined_data.drop("Label", axis=1)
    y = combined_data["Label"]
    X.fillna(X.mean(), inplace=True)
    scaler = StandardScaler()
    print("Scaling features...")
    X_scaled = scaler.fit_transform(X)
    print("Feature scaling complete.")
    return X_scaled, y, scaler


def split_dataset(X, y):
    """Split dataset into training and test sets."""
    print("Splitting dataset...")
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_SVC(X_train, y_train):
    """Train Support Vector Classifier."""
    print("Training SVC...")
    classifier = SVC(kernel="linear")
    classifier.fit(X_train, y_train)
    print("SVC training complete.")
    return classifier


def assess_model(classifier, X_test, y_test):
    """Evaluate the trained model."""
    print("Assessing model...")
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy * 100:.2f}%")


def process_unseen(file_path, scaler):
    """Process an unlabeled file."""
    print(f"Processing new file: {file_path}")
    label = os.path.basename(file_path).split("1.csv")[0]
    df = pd.read_csv(file_path)
    X = df.fillna(df.mean())
    X_scaled = scaler.transform(X)
    return X_scaled, label


# Paths and execution
directory = "/home/leo/Documents/test2/"
X, y, scaler = process_data(directory)
X_train, X_test, y_train, y_test = split_dataset(X, y)
model = train_SVC(X_train, y_train)
assess_model(model, X_test, y_test)

# Uncomment to predict new files
# new_file = "/path/to/unlabeled.csv"
# X_new, label_new = process_unseen(new_file, scaler)
# prediction = model.predict(X_new)
# print(f"Prediction: {prediction[0]}, Actual label: {label_new}")
