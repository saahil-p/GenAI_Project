import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
import os

# -------------------------------
# 1. Define Custom Dataset Class
# -------------------------------
class CustomDataset(Dataset):
    def __init__(self, csv_path, scaler=None, fit_scaler=False):
        self.data = pd.read_csv(csv_path)
        self.num_features = self.data.shape[1] - 2

        features = self.data.iloc[:, :self.num_features].values
        features = np.where(np.isfinite(features), features, np.nan)
        col_means = np.nanmean(features, axis=0)
        indices = np.where(np.isnan(features))
        features[indices] = np.take(col_means, indices[1])
        features = np.clip(features, -1e6, 1e6).astype("float32")

        if fit_scaler:
            self.scaler = StandardScaler()
            self.scaler.fit(features)
        else:
            self.scaler = scaler

        self.features = self.scaler.transform(features)
        self.class_labels = self.data.iloc[:, self.num_features].values.astype(int)
        self.state_labels = self.data.iloc[:, self.num_features + 1].values.astype(int)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "class": torch.tensor(self.class_labels[idx], dtype=torch.long),
            "state": torch.tensor(self.state_labels[idx], dtype=torch.long),
        }

# -------------------------------
# 2. Define Model Architecture
# -------------------------------
class ANN(nn.Module):
    def __init__(self, input_size, num_classes, num_states, hidden_size=64):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.class_out = nn.Linear(hidden_size, num_classes)
        self.state_out = nn.Linear(hidden_size, num_states)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        class_pred = self.class_out(x)
        state_pred = self.state_out(x)
        return class_pred, state_pred

# -------------------------------
# 3. Load Model Checkpoint
# -------------------------------
checkpoint_path = "checkpoint.pth"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

csv_path = "processed_dataset.csv"
dataset = CustomDataset(csv_path, fit_scaler=True)

test_size = int(0.2 * len(dataset))
_, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - test_size, test_size])
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

input_size = dataset.num_features
num_classes = len(set(dataset.class_labels))
num_states = len(set(dataset.state_labels))

model = ANN(input_size, num_classes, num_states).to(device)
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# -------------------------------
# 4. Evaluate and Generate Report
# -------------------------------
all_class_preds, all_state_preds = [], []
all_class_labels, all_state_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        features = batch["features"].to(device)
        class_labels = batch["class"].cpu().numpy()
        state_labels = batch["state"].cpu().numpy()

        class_pred, state_pred = model(features)
        class_pred = torch.argmax(class_pred, dim=1).cpu().numpy()
        state_pred = torch.argmax(state_pred, dim=1).cpu().numpy()

        all_class_preds.extend(class_pred)
        all_state_preds.extend(state_pred)
        all_class_labels.extend(class_labels)
        all_state_labels.extend(state_labels)

print("Classification Report for Class Labels:")
print(classification_report(all_class_labels, all_class_preds))

print("Classification Report for State Labels:")
print(classification_report(all_state_labels, all_state_preds))
