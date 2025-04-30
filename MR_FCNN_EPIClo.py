#  FINAL MR_FCNN_fixed.py
# - output_dim = 42
# - Thumb: joints 0–8 → sin/cos (18 outputs)
# - Index PIP (13/14), DIP (16/17), Middle DIP (25/26) → sin/cos (6 joints × 2 = 12 outputs)
# - Remaining 12 joints → raw (12 outputs)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from HandPoseClass import *

def load_data(dataset_path, closure_columns, z_thresh=2.5):
    data = pd.read_csv(dataset_path)
    data.columns = data.columns.str.strip()
    joint_columns = [col for col in data.columns if col not in closure_columns]

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
    from scipy.stats import zscore
    z_scores = np.abs(zscore(y, axis=0))
    mask = (z_scores < threshold).all(axis=1)
    return X[mask], y[mask]

def prepare_dataloaders(X, y, test_size=0.2, batch_size=64):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # THUMB (0–8)
    thumb_train_rad = np.deg2rad(y_train[:, :9])
    thumb_test_rad = np.deg2rad(y_test[:, :9])
    thumb_train = np.concatenate([np.sin(thumb_train_rad), np.cos(thumb_train_rad)], axis=1)
    thumb_test = np.concatenate([np.sin(thumb_test_rad), np.cos(thumb_test_rad)], axis=1)

    # PIP/DIP fix: joints = [13,14,16,17,25,26]
    fix_indices = [13,14,16,17,25,26]
    fix_train_rad = np.deg2rad(y_train[:, fix_indices])
    fix_test_rad = np.deg2rad(y_test[:, fix_indices])
    fix_train = np.concatenate([np.sin(fix_train_rad), np.cos(fix_train_rad)], axis=1)
    fix_test = np.concatenate([np.sin(fix_test_rad), np.cos(fix_test_rad)], axis=1)

    # Remaining joints (raw)
    raw_indices = [i for i in range(27) if i not in list(range(0,9)) + fix_indices]
    raw_train = y_train[:, raw_indices]
    raw_test = y_test[:, raw_indices]

    y_train_final = np.concatenate([thumb_train, fix_train, raw_train], axis=1)
    y_test_final = np.concatenate([thumb_test, fix_test, raw_test], axis=1)

    scaler_y = StandardScaler().fit(y_train_final)
    y_train_scaled = scaler_y.transform(y_train_final)
    y_test_scaled = scaler_y.transform(y_test_final)

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train_scaled)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test_scaled)), batch_size=batch_size)

    return train_loader, test_loader, scaler_y

def train(model, train_loader, test_loader, optimizer, scheduler, epochs=300):
    train_losses, test_losses = [], []
    best_loss, best_model_state = float('inf'), None

    for epoch in range(epochs):
        model.train()
        train_loss = sum_step_loss(model, train_loader, optimizer, training=True)

        model.eval()
        test_loss = sum_step_loss(model, test_loader, training=False)

        scheduler.step(test_loss)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if test_loss < best_loss:
            best_loss = test_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, 'hand_pose_fcnn.pth')

        if epoch % 50 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d} | LR: {lr:.5f} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    if best_model_state:
        model.load_state_dict(best_model_state)

    return train_losses, test_losses

def sum_step_loss(model, loader, optimizer=None, training=False):
    total_loss = 0.0
    for xb, yb in loader:
        preds = model(xb)

        weights = torch.ones_like(preds)
        weights[:, :18] = 2.0   # thumb sin/cos
        weights[:, 18:30] = 2.0 # pip/dip sin/cos

        loss = torch.mean(weights * (preds - yb) ** 2)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

if __name__ == "__main__":
    closure_columns = ['ThumbClosure', 'IndexClosure', 'MiddleClosure', 'ThumbAbduction']
    X, y, joint_columns = load_data('last_dataset.csv', closure_columns)
    train_loader, test_loader, scaler_y = prepare_dataloaders(X, y)

    model = HandPoseFCNN(input_dim=4, output_dim=42)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)

    train_losses, test_losses = train(model, train_loader, test_loader, optimizer, scheduler)
    joblib.dump(scaler_y, "scaler_y.save")
    print("Training complete. Model saved.")
