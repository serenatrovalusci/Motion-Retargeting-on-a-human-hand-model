import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import joblib
from HandPoseClass import *

# ----------------------------
# 1. Data Loading & Preprocessing
# ----------------------------
def load_data(dataset_path, closure_columns, z_thresh=3.0):
    data = pd.read_csv(dataset_path)
    data.columns = data.columns.str.strip()
    joint_columns = [col for col in data.columns if col not in closure_columns]

    # Remove last 453 rows
    data = data.iloc[:-453, :]

    X = data[closure_columns].values
    y = data[joint_columns].values

    print("Input features (X):", closure_columns)
    print("Target features (y):", joint_columns)
    print(f"Output features count: {len(joint_columns)}")

    print("Before outlier removal:", X.shape, y.shape)
    X, y = remove_outliers_zscore(X, y, z_thresh)
    print("After outlier removal:", X.shape, y.shape)

    return X, y, joint_columns

def remove_outliers_zscore(X, y, threshold):
    z_scores = np.abs(zscore(y, axis=0))
    mask = (z_scores < threshold).all(axis=1)
    return X[mask], y[mask]

def prepare_dataloaders(X, y, test_size=0.2, batch_size=64):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    X_train, X_test = np.round(X_train, 3), np.round(X_test, 3)
    y_train, y_test = np.round(y_train, 3), np.round(y_test, 3)

    scaler_y = StandardScaler().fit(y_train)
    y_train = scaler_y.transform(y_train)
    y_test = scaler_y.transform(y_test)

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)), batch_size=batch_size)

    return train_loader, test_loader, scaler_y

# ----------------------------
# 2. Training & Evaluation
# ----------------------------
def train(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=1500):
    train_losses, test_losses = [], []
    best_loss, best_model_state = float('inf'), None

    for epoch in range(epochs):
        model.train()
        train_loss = sum_step_loss(model, train_loader, criterion, optimizer, training=True)

        model.eval()
        test_loss = sum_step_loss(model, test_loader, criterion, training=False)

        scheduler.step(test_loss)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if test_loss < best_loss:
            best_loss = test_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, 'hand_pose_fcnn.pth')

        if epoch % 50 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch:3d} | LR: {lr:.5f} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}')

    if best_model_state:
        model.load_state_dict(best_model_state)

    return train_losses, test_losses

def sum_step_loss(model, loader, criterion, optimizer=None, training=False):
    total_loss = 0.0
    for xb, yb in loader:
        if training:
            optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

def test_model(model, test_loader, criterion):
    model.eval()
    all_preds, all_targets, total_loss = [], [], 0.0

    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb)
            total_loss += criterion(preds, yb).item() * xb.size(0)
            all_preds.append(preds)
            all_targets.append(yb)

    predictions = torch.cat(all_preds).cpu().numpy()
    targets = torch.cat(all_targets).cpu().numpy()
    mse = total_loss / len(test_loader.dataset)
    mae = np.mean(np.abs(predictions - targets))

    print(f"\nFinal Test MSE: {mse:.4f}")
    print(f"Final Test MAE: {mae:.4f}")

    return predictions, targets, mse, mae

# ----------------------------
# 3. Visualization
# ----------------------------
def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_per_joint_mse(preds, targets):
    per_joint_mse = np.mean((preds - targets) ** 2, axis=0)
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(per_joint_mse)), per_joint_mse)
    plt.xlabel("Joint Index")
    plt.ylabel("MSE")
    plt.title("Per-Joint MSE on Test Set")
    plt.grid(True)
    plt.show()

# ----------------------------
# 4. Main Pipeline
# ----------------------------
if __name__ == "__main__":
    closure_columns = ['ThumbClosure', 'IndexClosure', 'MiddleClosure', 'ThumbAbduction']
    X, y, joint_columns = load_data('hand_dataset_6.csv', closure_columns)

    train_loader, test_loader, scaler_y = prepare_dataloaders(X, y)

    model = HandPoseFCNN()
    criterion = nn.MSELoss()  # More sensitive to outliers
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)

    train_losses, test_losses = train(model, train_loader, test_loader, criterion, optimizer, scheduler)

    joblib.dump(scaler_y, "scaler_y.save")

    plot_losses(train_losses, test_losses)

    # Best model is already loaded at this point
    preds, targets, test_mse, test_mae = test_model(model, test_loader, criterion)

    preds_real = scaler_y.inverse_transform(preds)
    targets_real = scaler_y.inverse_transform(targets)

    mae_real = np.mean(np.abs(preds_real - targets_real))
    print(f"Real-space MAE: {mae_real:.4f}")

    print("all_preds", preds)
    print("all_targets", targets)

    plot_per_joint_mse(preds, targets)

    print("\nFinal model saved to hand_pose_fcnn.pth")
