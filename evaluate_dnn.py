import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Define the same DNN model class used during training.
class DNN(nn.Module):
    def __init__(self, input_size, num_classes, num_states):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.class_out = nn.Linear(32, num_classes)
        self.state_out = nn.Linear(32, num_states)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        class_pred = self.class_out(x)
        state_pred = self.state_out(x)
        return class_pred, state_pred

def preprocess_features(X):
    # Handle NaN and Inf values like in training
    X = np.where(np.isfinite(X), X, np.nan)
    col_means = np.nanmean(X, axis=0)
    indices = np.where(np.isnan(X))
    X[indices] = np.take(col_means, indices[1])
    # Clip extreme values
    X = np.clip(X, -1e6, 1e6).astype("float32")
    return X

def load_scaler(scaler_path="scaler.pkl"):
    # Optionally, if you saved the scaler during training, load it here.
    import pickle
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return scaler
    return None

def evaluate_model(model_path, csv_path, results_file="results.txt", sample_fraction=0.1):
    # Check CSV exists
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' does not exist!")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file '{csv_path}': {e}")
        return

    if df.empty:
        print(f"CSV file '{csv_path}' is empty. Skipping evaluation.")
        return

    # Sample a fraction of rows
    df_sampled = df.sample(frac=sample_fraction, random_state=42)
    num_columns = df_sampled.shape[1]

    # Assume last two columns are labels, rest are features.
    y_true_class = df_sampled.iloc[:, num_columns - 2].values
    y_true_state = df_sampled.iloc[:, num_columns - 1].values
    X = df_sampled.iloc[:, :num_columns - 2].values

    # Preprocess the features (handle NaNs, clipping, etc.)
    X = preprocess_features(X)
    
    # Apply StandardScaler transformation
    scaler = load_scaler()  # If you saved the scaler during training.
    if scaler is None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Map original state labels (0, 2, 4, 7) to new labels (0, 1, 2, 3)
    mapping = {0: 0, 2: 1, 4: 2, 7: 3}
    y_true_state = np.array([mapping[val] for val in y_true_state])

    # Load the stored model using safe globals.
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' does not exist!")
        return

    try:
        with torch.serialization.safe_globals([DNN]):
            model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
        model.eval()
    except Exception as e:
        print(f"Error loading model '{model_path}': {e}")
        return

    # Run inference.
    with torch.no_grad():
        outputs = model(X_tensor)

    y_pred_class = torch.argmax(outputs[0], dim=1).numpy()
    y_pred_state = torch.argmax(outputs[1], dim=1).numpy()

    # Generate and write classification reports.
    report_class = classification_report(y_true_class, y_pred_class)
    report_state = classification_report(y_true_state, y_pred_state)

    with open(results_file, "a") as f:
        f.write(f"=== Evaluation for Model: {model_path} ===\n")
        f.write("---- 'class' Classification Report ----\n")
        f.write(report_class)
        f.write("\n---- 'state' Classification Report ----\n")
        f.write(report_state)
        f.write("\n" + "=" * 100 + "\n")

    print(f"Evaluation completed for model '{model_path}'. Reports written to '{results_file}'.")

if __name__ == "__main__":
    model_path = "/Users/saahil/Desktop/College/Sem 6/GenAI/RAG/dnn_model_full.pth"
    csv_path = "/Users/saahil/Desktop/College/Sem 6/GenAI/RAG/processed_dataset.csv"
    
    evaluate_model(model_path, csv_path)
