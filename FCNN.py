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
from scipy.stats import zscore

# Loads data from a CSV file, selects input and output columns, and removes outliers
def load_data(dataset_path, closure_columns, z_thresh=2.5):

    data = pd.read_csv(dataset_path)  # Load dataset
    data.columns = data.columns.str.strip()  # Removes any whitespace from column names, prevents bugs
    joint_columns = [col for col in data.columns if col not in closure_columns]

    X = data[closure_columns].values  # Input features (closure parameters and thumb abduction parameter)
    Y = data[joint_columns].values    # Target features (joint angles)

    print(f"Joint angles count: {len(joint_columns)}")
    print(f"Closure parameters count: {len(closure_columns)}")

    # Remove outliers using Z-score

    print("Before outlier removal:", X.shape, Y.shape)
    X, Y = remove_outliers_zscore(X, Y, z_thresh)  
    print("After outlier removal:", X.shape, Y.shape)

    return X, Y, joint_columns

# Removes samples from the dataset whose joint values are outliers (based on Z-score)
def remove_outliers_zscore(X, Y, threshold):
    
    z_scores = np.abs(zscore(Y, axis=0))
    mask = (z_scores < threshold).all(axis=1)  # Keep rows where all joint Z-scores are below the threshold

    return X[mask], Y[mask]

# Prepares PyTorch DataLoaders for training and testing
def prepare_dataloaders(X, Y, test_size=0.2, batch_size=64):

    # Split into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    #Convert thumb joint angles from degrees to sine and cosine components to avoid ambiguity caused by angle 
    #periodicity and improve prediction stability and accuracy 

    thumb_train_rad = np.deg2rad(Y_train[:, :9])
    thumb_test_rad = np.deg2rad(Y_test[:, :9])
    thumb_train = np.concatenate([np.sin(thumb_train_rad), np.cos(thumb_train_rad)], axis=1)
    thumb_test = np.concatenate([np.sin(thumb_test_rad), np.cos(thumb_test_rad)], axis=1)

    # Convert PIP/DIP joint angles to sine and cosine components
    
    fix_indices = [13,14,16,17,25,26]
    fix_train_rad = np.deg2rad(Y_train[:, fix_indices])
    fix_test_rad = np.deg2rad(Y_test[:, fix_indices])
    fix_train = np.concatenate([np.sin(fix_train_rad), np.cos(fix_train_rad)], axis=1)
    fix_test = np.concatenate([np.sin(fix_test_rad), np.cos(fix_test_rad)], axis=1)

    # Keep remaining joints as raw angles
    raw_indices = [i for i in range(27) if i not in list(range(0,9)) + fix_indices]
    raw_train = Y_train[:, raw_indices]
    raw_test = Y_test[:, raw_indices]

    # Concatenate all processed outputs
    Y_train_final = np.concatenate([thumb_train, fix_train, raw_train], axis=1)
    Y_test_final = np.concatenate([thumb_test, fix_test, raw_test], axis=1)

    # Normalize output values
    scaler_y = StandardScaler().fit(Y_train_final)
    Y_train_scaled = scaler_y.transform(Y_train_final)
    Y_test_scaled = scaler_y.transform(Y_test_final)

    # Create PyTorch DataLoaders
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train_scaled)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(Y_test_scaled)), batch_size=batch_size)

    return train_loader, test_loader, scaler_y

# Trains the model, tracks best state, and saves it
def train(model, train_loader, test_loader, optimizer, scheduler, loss_fn, epochs=300, save_path="FCNN_weights.pth"):
    train_losses, test_losses = [], []
    best_loss, best_model_state = float('inf'), None

    for epoch in range(epochs):
        model.train()
        train_loss = sum_step_loss(model, train_loader, loss_fn, optimizer, training=True)

        model.eval()
        test_loss = sum_step_loss(model, test_loader, loss_fn, training=False)

        scheduler.step(test_loss)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if test_loss < best_loss:
            best_loss = test_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, save_path)

        if epoch % 50 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d} | LR: {lr:.5f} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    if best_model_state:
        model.load_state_dict(best_model_state)

    return train_losses, test_losses

# Computes the total weighted MSE loss across one epoch
def sum_step_loss(model, loader, loss_fn, optimizer=None, training=False):
    total_loss = 0.0
    for xb, yb in loader:
        preds = model(xb)
        loss = loss_fn(preds, yb)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

def weighted_mse_loss(preds, targets):
    weights = torch.ones_like(preds)
    weights[:, :18] = 2.0   # Thumb (sin/cos)
    weights[:, 18:30] = 2.0 # PIP/DIP (sin/cos)
    return torch.mean(weights * (preds - targets) ** 2)


# Entry point for running the training pipeline
if __name__ == "__main__":
    closure_columns = ['ThumbClosure', 'IndexClosure', 'MiddleClosure', 'ThumbAbduction']
    X, Y, joint_columns = load_data('dataset.csv', closure_columns)
    train_loader, test_loader, scaler_y = prepare_dataloaders(X, Y)

    model = HandPoseFCNN(input_dim=4, output_dim=42)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)

    train_losses, test_losses = train(model, train_loader, test_loader, optimizer, scheduler,weighted_mse_loss, epochs=300)
    joblib.dump(scaler_y, "scaler_y.save")
    print("Training complete. Model saved.")
