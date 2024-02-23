import re
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier  # Import XGBoost Classifier


def read_and_label(file_path):
    print(f"Reading and labeling {file_path}...")
    label = os.path.basename(file_path).split(".csv")[0]
    label = re.sub("[0-9]*", "", label)
    df = pd.read_csv(file_path)
    # Replicate each row in the dataframe 3 times
    df = pd.concat(
        [df] * 3, ignore_index=True
    )  # This line is added to replicate the data
    df["Label"] = label
    return df


def read_and_process_data(directory_path):
    print("Starting to read and process data...")
    data = pd.DataFrame()
    for file in os.listdir(directory_path):
        if file.endswith(".csv"):
            file_path = os.path.join(directory_path, file)
            data = pd.concat([data, read_and_label(file_path)], ignore_index=True)
    print("Finished reading and processing data.")
    X = data.drop(["Label"], axis=1)
    y = data["Label"]
    X.fillna(X.mean(), inplace=True)
    # Label encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    scaler = StandardScaler()
    print("Scaling data...")
    X_scaled = scaler.fit_transform(X)
    print("Data scaling completed.")
    return X_scaled, y_encoded, scaler, label_encoder


def split_data(X, y):
    print("Splitting data into training and testing sets...")
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    print("Training model...")
    model = XGBClassifier(
        objective="multi:softprob",  # Specify multi-class classification
        eval_metric="mlogloss",  # Logarithmic loss for multi-class classification
        use_label_encoder=False,  # As of xgboost version >= 1.3.0, the use_label_encoder option should be set to False
    )
    model.fit(X_train, y_train)
    print("Model training completed.")
    return model


def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy*100:.2f}%")


def read_unlabeled_file(file_path, scaler, label_encoder):
    print(f"Processing unlabeled file: {file_path}")
    label = os.path.basename(file_path).split("1.csv")[0]  # Extract label for reference
    df = pd.read_csv(file_path)
    X = df.fillna(df.mean())
    X_scaled = scaler.transform(X)  # Use the same scaler as the training data
    return X_scaled, label  # Return both scaled features and label


# Example usage
directory_path = "/home/leo/Documents/test2/"
X, y, scaler, label_encoder = read_and_process_data(directory_path)
X_train, X_test, y_train, y_test = split_data(X, y)
model = train_model(X_train, y_train)
evaluate_model(model, X_test, y_test)

unlabeled_file_path = "/home/leo/Documents/test2/test/Rotate24.csv"
X_unlabeled, unlabeled_file_label = read_unlabeled_file(
    unlabeled_file_path, scaler, label_encoder
)
prediction = model.predict(X_unlabeled)
predicted_label = label_encoder.inverse_transform(prediction)[0]
print(f"Model's prediction: {predicted_label}")
print(f"Actual label of the unlabeled file: {unlabeled_file_label}")
