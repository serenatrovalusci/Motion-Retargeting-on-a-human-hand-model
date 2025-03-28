import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import joblib

# ----------------------------
# 1. Data Loading & Preprocessing
# ----------------------------
def load_data(dataset_path):
    data = pd.read_csv(dataset_path)
    data.columns = data.columns.str.strip()

    closure_columns = ['ThumbClosure', 'IndexClosure', 'MiddleClosure']
    joint_columns = [col for col in data.columns if col not in closure_columns]

    X = data[closure_columns].values
    y = data[joint_columns].values

    print("Input features (X):", closure_columns)
    print("Target features (y):", joint_columns)
    print(f"Number of output features: {len(joint_columns)}")

    return X, y

X, y = load_data('hand_dataset_3.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.round(X_train, 3)
X_test = np.round(X_test, 3)
y_train = np.round(y_train, 3)
y_test = np.round(y_test, 3)

scaler_y = StandardScaler().fit(y_train)
y_train = scaler_y.transform(y_train)
y_test = scaler_y.transform(y_test)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

# ----------------------------
# 2. Model Definition
# ----------------------------
class HandPoseFCNN(nn.Module):
    def __init__(self, input_dim=3, output_dim=27):
        super().__init__()
        self.net = nn.Sequential(
    nn.Linear(input_dim, 512),
    nn.LeakyReLU(),
    nn.BatchNorm1d(512),

    nn.Linear(512, 256),
    nn.LeakyReLU(),
    nn.Dropout(0.3),

    nn.Linear(256, 128),
    nn.LeakyReLU(),
    nn.Dropout(0.3),

    nn.Linear(128, 64),
    nn.LeakyReLU(),
    nn.BatchNorm1d(64),

    nn.Linear(64, output_dim)

        )

    def forward(self, x):
        return self.net(x)

model = HandPoseFCNN()

# ----------------------------
# 3. Training Setup
# ----------------------------
criterion = nn.SmoothL1Loss()  # Less sensitive to outliers
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)

# ----------------------------
# 4. Training Loop with Model Saving
# ----------------------------
def train(model, train_loader, test_loader, epochs=1500):
    train_losses, test_losses = [], []
    best_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                preds = model(xb)
                test_loss += criterion(preds, yb).item() * xb.size(0)
        test_loss /= len(test_loader.dataset)

        scheduler.step(test_loss)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, 'best_model.pth')

        if epoch % 50 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch:3d} | LR: {lr:.5f} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}')

    # Load best model before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, test_losses

# ----------------------------
# 5. Test Function
# ----------------------------
def test_model(model, test_loader):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            total_loss += loss.item() * xb.size(0)
            all_preds.append(preds)
            all_targets.append(yb)

    mse = total_loss / len(test_loader.dataset)
    predictions = torch.cat(all_preds).cpu().numpy()
    targets = torch.cat(all_targets).cpu().numpy()
    mae = np.mean(np.abs(predictions - targets))

    print(f"\n Final Test MSE: {mse:.4f}")
    print(f" Final Test MAE: {mae:.4f}")

    return predictions, targets, mse, mae

# ----------------------------
# 6. Train the Model
# ----------------------------
train_losses, test_losses = train(model, train_loader, test_loader)

# Save the scaler
joblib.dump(scaler_y, "scaler_y.save")


# ----------------------------
# 7. Visualization of Losses
# ----------------------------
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Progress')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# 8. Load Best Model and Test
# ----------------------------
model.load_state_dict(torch.load('best_model.pth'))
preds, targets, test_mse, test_mae = test_model(model, test_loader)

# Invert to real values
preds_real = scaler_y.inverse_transform(preds)
targets_real = scaler_y.inverse_transform(targets)

# Optional: save to CSV or compute metrics
mae_real = np.mean(np.abs(preds_real - targets_real))
print(f" Real-space MAE: {mae_real:.4f}")

print("all_preds", preds)
print("all_targets", targets)

# ----------------------------
# 9. Per-Joint MSE Plot
# ----------------------------
per_joint_mse = np.mean((preds - targets) ** 2, axis=0)

plt.figure(figsize=(12, 5))
plt.bar(range(len(per_joint_mse)), per_joint_mse)
plt.xlabel("Joint Index")
plt.ylabel("MSE")
plt.title("Per-Joint MSE on Test Set")
plt.grid(True)
plt.show()

# ----------------------------
# 10. Save Final Model
# ----------------------------
torch.save(model.state_dict(), 'hand_pose_fcnn.pth')
print("\n Final model saved to hand_pose_fcnn.pth")
