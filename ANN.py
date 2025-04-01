import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import platform
import os

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

# -------------------------------
# File path and Dataset Setup
# -------------------------------
csv_path = "processed_dataset.csv"
dataset = CustomDataset(csv_path)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

num_workers = 0 if platform.system() == "Darwin" else 2
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers)

input_size = dataset.num_features
num_classes = len(set(dataset.class_labels))
num_states = len(set(dataset.state_labels))

# -------------------------------
# 2. Define the ANN Model
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
# 3. Setup Device, Loss & Optimizer
# -------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = ANN(input_size, num_classes, num_states).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# Checkpoint Functions
# -------------------------------
checkpoint_path = "checkpoint.pth"

def save_checkpoint(epoch, model, optimizer, path=checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(model, optimizer, path=checkpoint_path):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        print(f"Resuming from epoch {checkpoint['epoch'] + 1}")
        return checkpoint['epoch'] + 1
    return 0

# -------------------------------
# 4. Train the Model with Checkpointing
# -------------------------------
def train(model, train_loader, criterion, optimizer, epochs=10):
    start_epoch = load_checkpoint(model, optimizer)
    model.train()
    for epoch in range(start_epoch, epochs):
        total_loss = 0
        for batch in train_loader:
            features = batch["features"].to(device)
            class_labels = batch["class"].to(device)
            state_labels = batch["state"].to(device)

            optimizer.zero_grad()
            class_pred, state_pred = model(features)
            loss1 = criterion(class_pred, class_labels)
            loss2 = criterion(state_pred, state_labels)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
        save_checkpoint(epoch, model, optimizer)

# -------------------------------
# 5. Evaluate the Model
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

# -------------------------------
# 6. Run Training & Evaluation
# -------------------------------
if __name__ == "__main__":
    train(model, train_loader, criterion, optimizer, epochs=10)
    evaluate(model, test_loader)