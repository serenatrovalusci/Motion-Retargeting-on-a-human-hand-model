import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
import joblib
from HandPoseClass import *
import os
import matplotlib.pyplot as plt


def build_synergy_targets(y):
    thumb = np.concatenate([np.sin(np.deg2rad(y[:, :9])), np.cos(np.deg2rad(y[:, :9]))], axis=1)
    fix_indices = [13, 14, 16, 17, 25, 26]
    fix = np.concatenate([np.sin(np.deg2rad(y[:, fix_indices])), np.cos(np.deg2rad(y[:, fix_indices]))], axis=1)
    raw_indices = [i for i in range(27) if i not in list(range(9)) + fix_indices]
    raw = y[:, raw_indices]
    return np.concatenate([thumb, fix, raw], axis=1)

def generate_pca_dataset(csv_path, closure_columns, pca_var=0.97):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    y = df[[c for c in df.columns if c not in closure_columns]].values
    X = df[closure_columns].values
    Y_synergy = build_synergy_targets(y)

    scaler = StandardScaler().fit(Y_synergy)
    Y_scaled = scaler.transform(Y_synergy)
    pca = PCA(n_components=pca_var).fit(Y_scaled)
    Y_pca = pca.transform(Y_scaled)

    joblib.dump(scaler, "scaler_joint_11.save")
    joblib.dump(pca, "pca_joint_11.save")
    return X, Y_pca

def prepare_dataloaders(X, Y, batch_size=64):
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(Ytr)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(Xte), torch.FloatTensor(Yte)), batch_size=batch_size)
    return train_loader, test_loader

def train(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=300):
    train_losses, test_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        total_train_samples = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * len(xb)
            total_train_samples += len(xb)

        train_loss = running_train_loss / total_train_samples
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            running_test_loss = 0.0
            total_test_samples = 0
            for xb, yb in test_loader:
                preds = model(xb)
                loss = criterion(preds, yb)
                running_test_loss += loss.item() * len(xb)
                total_test_samples += len(xb)

        test_loss = running_test_loss / total_test_samples
        test_losses.append(test_loss)

        scheduler.step(test_loss)

        if epoch % 50 == 0:
            print(f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

    torch.save(model.state_dict(), "hand_pose_fcnn_PCA_11_new.pth")
    return train_losses, test_losses

if __name__ == "__main__":
    closure_cols = ['ThumbClosure', 'IndexClosure', 'MiddleClosure', 'ThumbAbduction']
    X, Y_pca = generate_pca_dataset("last_dataset.csv", closure_cols)
    train_loader, test_loader = prepare_dataloaders(X, Y_pca)
    model = HandPoseFCNN(input_dim=4, output_dim=Y_pca.shape[1])
    optim_ = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(optim_, patience=10, factor=0.5)

    train_losses, test_losses = train(model, train_loader, test_loader, nn.MSELoss(), optim_, sched)

    # 🔽 Save loss graph
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve (PCA)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/loss_curve_pca.png")
    plt.close()

