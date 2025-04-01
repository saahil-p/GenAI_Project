import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# 1. Load and Preprocess Data
# -------------------------------
class CustomDataset(Dataset):
    def __init__(self, csv_path, scaler=None, fit_scaler=True):
        self.data = pd.read_csv(csv_path)
        self.num_features = self.data.shape[1] - 2  # Assuming last 2 columns are labels

        features = self.data.iloc[:, :self.num_features].values

        # Handle NaN, Inf values
        features = np.where(np.isfinite(features), features, np.nan)
        col_means = np.nanmean(features, axis=0)
        indices = np.where(np.isnan(features))
        features[indices] = np.take(col_means, indices[1])

        # Clip extreme values
        features = np.clip(features, -1e6, 1e6).astype("float32")

        # Normalize features
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

# Load dataset
csv_path = "/Users/saahil/Desktop/College/Sem 6/TDL_PROJ/combined files/processed_dataset.csv"
dataset = CustomDataset(csv_path)

# Split dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

input_size = dataset.num_features
num_classes = len(set(dataset.class_labels))
num_states = len(set(dataset.state_labels))

# -------------------------------
# 2. Define the DNN Model
# -------------------------------
class DNN(nn.Module):
    def __init__(self, input_size, num_classes, num_states):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.class_out = nn.Linear(32, num_classes)
        self.state_out = nn.Linear(32, num_states)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        class_pred = self.class_out(x)
        state_pred = self.state_out(x)
        return class_pred, state_pred

# -------------------------------
# 3. Train the Model
# -------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = DNN(input_size, num_classes, num_states).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

original_labels = torch.tensor([0, 2, 4, 7])

# Create a mapping dictionary
label_map = {val.item(): i for i, val in enumerate(original_labels)}

# Function to apply the mapping
def remap_labels(labels):
    return torch.tensor([label_map[val.item()] for val in labels], dtype=torch.long)

def train(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            features = batch["features"].to(device)
            class_labels = batch["class"].to(device)
            state_labels = batch["state"].to(device)

            # Apply remapping
            state_labels = remap_labels(state_labels).to(device)

            optimizer.zero_grad()
            class_pred, state_pred = model(features)

            loss1 = criterion(class_pred, class_labels)
            loss2 = criterion(state_pred, state_labels)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")


# -------------------------------
# 4. Evaluate the Model
# -------------------------------
def evaluate(model, test_loader):
    model.eval()
    correct_class, correct_state, total = 0, 0, 0
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            class_labels = batch["class"].to(device)
            state_labels = batch["state"].to(device)

            class_pred, state_pred = model(features)
            class_pred = torch.argmax(class_pred, dim=1)
            state_pred = torch.argmax(state_pred, dim=1)

            correct_class += (class_pred == class_labels).sum().item()
            correct_state += (state_pred == state_labels).sum().item()
            total += class_labels.size(0)

    print(f"Class Accuracy: {100 * correct_class / total:.2f}%")
    print(f"State Accuracy: {100 * correct_state / total:.2f}%")

if __name__ == "__main__":
    train(model, train_loader, criterion, optimizer)
    evaluate(model, test_loader)

