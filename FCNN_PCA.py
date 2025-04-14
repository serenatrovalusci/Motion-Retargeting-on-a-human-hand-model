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
from sklearn.decomposition import PCA
import argparse
import os

# ----------------------------
# 1. Data Loading & Preprocessing
# ----------------------------
def load_data(dataset_path, closure_columns, z_thresh=2.5):
    data = pd.read_csv(dataset_path)
    data.columns = data.columns.str.strip()
    joint_columns = [col for col in data.columns if col not in closure_columns]

    # Remove last 453 rows
    # data = data.iloc[:-453, :]

    # 1. Standardize
    scaler = StandardScaler()
    joint_angles_standardized = scaler.fit_transform(joint_columns)

    # 2. PCA with 95% variance retention
    pca = PCA(n_components=0.97)
    synergy_columns = pca.fit_transform(joint_angles_standardized)

    X = data[closure_columns].values
    y = data[synergy_columns].values

    print(f"Original dimensions: {joint_angles_standardized.shape[1]}")
    print(f"Reduced dimensions: {synergy_columns.shape[1]}")
    print(f"Explained variance: {np.sum(pca.explained_variance_ratio_):.2f}")

    print("Input features (X):", closure_columns)
    print("Target features (y):", synergy_columns)
    print(f"Output features count: {len(synergy_columns)}")

    print("Before outlier removal:", X.shape, y.shape)
    X, y = remove_outliers_zscore(X, y, z_thresh)
    print("After outlier removal:", X.shape, y.shape)

    # import matplotlib.pyplot as plt

    # plt.boxplot(y, vert=False)
    # plt.title("Distribution of joint outputs")
    # plt.xlabel("Z-score")
    # plt.show()


    return X, y, joint_columns

def generate_pca_dataset(dataset_path, closure_columns, pca_variance_threshold=0.97):
    data = pd.read_csv(dataset_path)
    data.columns = data.columns.str.strip()

    # Separate joint and closure/abduction columns
    joint_columns = [col for col in data.columns if col not in closure_columns]
    joint_data = data[joint_columns].values
    closure_data = data[closure_columns].values

    # Standardize joint data
    scaler = StandardScaler()
    joint_standardized = scaler.fit_transform(joint_data)
  
    # Apply PCA
    pca = PCA(n_components=pca_variance_threshold)
    synergy_data = pca.fit_transform(joint_standardized)

    print(f"PCA reduced joint dimensions from {joint_data.shape[1]} to {synergy_data.shape[1]}")
    print(f"Explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")

    joblib.dump(pca, 'pca_joint.save')
    

    return closure_data, synergy_data, scaler


def remove_outliers_zscore(X, y, threshold = 2.5):
    z_scores = np.abs(zscore(y, axis=0))
    mask = (z_scores < threshold).all(axis=1)
    return X[mask], y[mask]

def prepare_dataloaders(X, y, test_size=0.2, batch_size=64):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    X_train, X_test = np.round(X_train, 3), np.round(X_test, 3)
    y_train, y_test = np.round(y_train, 3), np.round(y_test, 3)

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)), batch_size=batch_size)

    return train_loader, test_loader

# ----------------------------
# 2. Training & Evaluation
# ----------------------------
def train(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=1000):
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
            torch.save(best_model_state, 'hand_pose_fcnn_PCA.pth')

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
# 4. Mode-specific Logic
# ----------------------------

def run_training():
    closure_columns = ['ThumbClosure', 'IndexClosure', 'MiddleClosure', 'ThumbAbduction']
    X, y, scaler = generate_pca_dataset('last_dataset.csv', closure_columns)
    joblib.dump(scaler, "scaler_PCA.save")
    X, y = remove_outliers_zscore(X, y)
    train_loader, test_loader = prepare_dataloaders(X, y)
    
    model = HandPoseFCNN_PCA()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)

    train_losses, test_losses = train(model, train_loader, test_loader, criterion, optimizer, scheduler)
    plot_losses(train_losses, test_losses)
    print("\nModel saved to hand_pose_fcnn_PCA.pth")

def run_testing():
    if not os.path.exists("hand_pose_fcnn_PCA.pth"):
        print("Model not found! Please train it first.")
        return

    closure_columns = ['ThumbClosure', 'IndexClosure', 'MiddleClosure', 'ThumbAbduction']
    X, y, scaler = generate_pca_dataset('last_dataset.csv', closure_columns)
    X, y = remove_outliers_zscore(X, y)
    _, test_loader = prepare_dataloaders(X, y)
    joblib.dump(scaler, "scaler_PCA.save")
    model = HandPoseFCNN_PCA()
    model.load_state_dict(torch.load("hand_pose_fcnn_PCA.pth"))
    model.eval()

    criterion = nn.MSELoss()
    preds, targets, test_mse, test_mae = test_model(model, test_loader, criterion)

    mae_real = np.mean(np.abs(preds - targets))
    print(f"Reconstructed-space MAE: {mae_real:.4f}")

    plot_per_joint_mse(preds, targets)

# ----------------------------
# 5. Entry Point
# ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test the HandPoseFCNN_PCA model.")
    parser.add_argument("--mode", choices=["train", "test"], required=True, help="Run mode: train or test")
    args = parser.parse_args()

    if args.mode == "train":
        run_training()
    elif args.mode == "test":
        run_testing()


