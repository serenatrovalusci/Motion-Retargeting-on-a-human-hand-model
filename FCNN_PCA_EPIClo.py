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

    joblib.dump(scaler, "scaler_joint.save")
    joblib.dump(pca, "pca_joint.save")
    return X, Y_pca

def prepare_dataloaders(X, Y, batch_size=64):
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(Ytr)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(Xte), torch.FloatTensor(Yte)), batch_size=batch_size)
    return train_loader, test_loader

def train(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=300):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            loss = sum(criterion(model(xb), yb) * len(xb) for xb, yb in test_loader) / len(test_loader.dataset)
        scheduler.step(loss)
        if epoch % 50 == 0:
            print(f"Epoch {epoch} | Test loss: {loss:.6f}")
    torch.save(model.state_dict(), "hand_pose_fcnn_PCA.pth")

if __name__ == "__main__":
    closure_cols = ['ThumbClosure', 'IndexClosure', 'MiddleClosure', 'ThumbAbduction']
    X, Y_pca = generate_pca_dataset("last_dataset.csv", closure_cols)
    train_loader, test_loader = prepare_dataloaders(X, Y_pca)
    model = HandPoseFCNN(input_dim=4, output_dim=Y_pca.shape[1])
    optim_ = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(optim_, patience=10, factor=0.5)
    train(model, train_loader, test_loader, nn.MSELoss(), optim_, sched)
