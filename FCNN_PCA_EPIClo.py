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

def generate_sincos_dataset(y):
    thumb = np.concatenate([np.sin(np.deg2rad(y[:, :9])), np.cos(np.deg2rad(y[:, :9]))], axis=1)
    fix_indices = [13, 14, 16, 17, 25, 26]
    fix = np.concatenate([np.sin(np.deg2rad(y[:, fix_indices])), np.cos(np.deg2rad(y[:, fix_indices]))], axis=1)
    raw_indices = [i for i in range(27) if i not in list(range(9)) + fix_indices]
    raw = y[:, raw_indices]
    return np.concatenate([thumb, fix, raw], axis=1)

def generate_pca_dataset(csv_path, closure_columns, pca_var=0.97):
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()
    y = data[[c for c in data.columns if c not in closure_columns]].values
    X = data[closure_columns].values
    Y_sincos = generate_sincos_dataset(y)

    scaler = StandardScaler().fit(Y_sincos)
    Y_scaled = scaler.transform(Y_sincos)
    pca = PCA(n_components=pca_var).fit(Y_scaled)
    Y_pca = pca.transform(Y_scaled)

    joblib.dump(scaler, "reconstruction_scaler.save")
    joblib.dump(pca, "reconstruction_pca.save")
    return X, Y_pca

def prepare_dataloaders(X, Y, batch_size=64):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(Xtrain), torch.FloatTensor(Ytrain)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(Xtest), torch.FloatTensor(Ytest)), batch_size=batch_size)
    return train_loader, test_loader

def train(model, train_loader, test_loader, optimizer, scheduler, loss_fn, epochs=300, save_path="FCNN_PCA.pth"):
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

def mse_loss(preds, targets):
    return torch.nn.functional.mse_loss(preds, targets)

if __name__ == "__main__":
    closure_cols = ['ThumbClosure', 'IndexClosure', 'MiddleClosure', 'ThumbAbduction']
    X, Y_pca = generate_pca_dataset("dataset.csv", closure_cols)
    train_loader, test_loader = prepare_dataloaders(X, Y_pca)
    model = HandPoseFCNN(input_dim=4, output_dim=Y_pca.shape[1])
    optim_ = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(optim_, patience=10, factor=0.5)
    train(model, train_loader, test_loader, optim_, sched, mse_loss, epochs=300)
