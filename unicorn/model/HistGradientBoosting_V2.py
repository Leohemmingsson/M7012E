from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from dotenv import load_dotenv
import os
import re

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

def read_and_process_data(directory_path):
    """Reads and processes data from a directory of CSV files."""
    data_frames = []
    for file in os.listdir(directory_path):
        if file.endswith(".csv"):
            file_path = os.path.join(directory_path, file)
            df = read_and_label(file_path)
            if not df.empty:
                data_frames.append(df)
    data = pd.concat(data_frames, ignore_index=True)
    X = data.drop(['Label'], axis=1)
    y = data['Label']
    X.fillna(X.mean(), inplace=True)  # Handling missing values by replacing them with column means
    return X, y

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    """Trains the HistGradientBoostingClassifier and evaluates it on the test set."""
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

# Example usage with separate training and testing paths
load_dotenv()
training_directory_path = os.getenv('FILES2')  # Training data directory
testing_directory_path = os.getenv('FILES')  # Testing data directory

# Process training data
X_train, y_train = read_and_process_data(training_directory_path)

# Process testing data
X_test, y_test = read_and_process_data(testing_directory_path)

# Train and evaluate model
train_and_evaluate_model(X_train, y_train, X_test, y_test)
