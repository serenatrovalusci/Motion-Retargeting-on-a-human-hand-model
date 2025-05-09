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
from sklearn.decomposition import PCA
import argparse
import os


def generate_sincos_dataset(Y):
    thumb = np.concatenate([np.sin(np.deg2rad(Y[:, :9])), np.cos(np.deg2rad(Y[:, :9]))], axis=1)
    fix_indices = [13, 14, 16, 17, 25, 26]
    fix = np.concatenate([np.sin(np.deg2rad(Y[:, fix_indices])), np.cos(np.deg2rad(Y[:, fix_indices]))], axis=1)
    raw_indices = [i for i in range(27) if i not in list(range(9)) + fix_indices]
    raw = Y[:, raw_indices]
    return np.concatenate([thumb, fix, raw], axis=1)


def load_data(csv_path, closure_columns):
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()
    joint_columns = [c for c in data.columns if c not in closure_columns]
    X = data[closure_columns].values
    Y = data[joint_columns].values

    print(f"Joint angles count: {Y.shape[1]}")
    print(f"Closure parameters count: {X.shape[1]}")
    print("Before outlier removal:", X.shape, Y.shape)

    X, Y = remove_outliers_zscore(X, Y, z_thresh)
    print("After outlier removal:", X.shape, Y.shape)

    Y_sincos = generate_sincos_dataset(Y)

    scaler = StandardScaler().fit(Y_sincos)
    Y_scaled = scaler.transform(Y_sincos)

    return X, Y_scaled, scaler


def fit_pca(y_scaled, pca_var=0.97):
    pca = PCA(n_components=pca_var).fit(y_scaled)
    print("PCA components used:", pca.n_components_)
    return pca


# def transform_to_pca(y_tensor, pca):
#     y_np = y_tensor.detach().cpu().numpy()
#     y_pca = pca.transform(y_np)
#     return torch.tensor(y_pca, dtype=y_tensor.dtype, device=y_tensor.device)

def transform_to_pca(y_tensor, pca):
    # Convert numpy components to PyTorch tensors once
    components = torch.tensor(pca.components_, dtype=y_tensor.dtype, device=y_tensor.device)
    mean = torch.tensor(pca.mean_, dtype=y_tensor.dtype, device=y_tensor.device)

    # Apply PCA transformation manually (centered dot-product)
    return (y_tensor - mean) @ components.T



def remove_outliers_zscore(X, Y, threshold):
    z_scores = np.abs(zscore(Y, axis=0))
    mask = (z_scores < threshold).all(axis=1)
    return X[mask], Y[mask]


def prepare_dataloaders(X, Y, batch_size=64):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(Xtrain), torch.FloatTensor(Ytrain)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(Xtest), torch.FloatTensor(Ytest)), batch_size=batch_size)
    return train_loader, test_loader


def train(model, train_loader, test_loader, optimizer, scheduler, loss_fn, epochs=300, save_path="FCNN_PCA.pth", pca=None):
    train_losses, test_losses = [], []
    best_loss, best_model_state = float('inf'), None

    for epoch in range(epochs):
        model.train()
        train_loss = sum_step_loss(model, train_loader, loss_fn, optimizer, training=True, pca=pca)

        model.eval()
        test_loss = sum_step_loss(model, test_loader, loss_fn, training=False, pca=pca)

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


def sum_step_loss(model, loader, loss_fn, optimizer=None, training=False, pca=None):
    total_loss = 0.0
    for xb, yb in loader:
        preds = model(xb)
        if pca is not None:
            preds = transform_to_pca(preds, pca)
            yb = transform_to_pca(yb, pca)
        loss = loss_fn(preds, yb)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def mse_loss(preds, targets):
    return torch.nn.functional.mse_loss(preds, targets)


def weighted_mse_loss(preds, targets):
    weights = torch.ones_like(preds)
    weights[:, :18] = 2.0   # Thumb (sin/cos)
    weights[:, 18:30] = 2.0 # PIP/DIP (sin/cos)
    return torch.mean(weights * (preds - targets) ** 2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train hand pose model. Optionally use PCA during loss computation to focus on principal motion components."
    )
    parser.add_argument('--use_pca', action='store_true',
                        help='Use PCA inside the loss function (model still predicts full joint space)')
    parser.add_argument('--pca_variance', type=float, default=0.97,
                        help='Explained variance ratio for PCA used in loss (default: 0.995)')
    parser.add_argument('--z_thresh', type=float, default=2.5,
                        help='Z-score threshold for outlier removal')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--save_model', type=str, default=None,
                        help='Custom filename to save the trained model')
    parser.add_argument('--plot_mse', action='store_true',
                        help='Plot and save loss curves after training')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("----------- Training Configuration -----------")
    print(f"Using PCA in loss:         {'Yes' if args.use_pca else 'No'}")
    print(f"PCA explained variance:    {args.pca_variance if args.use_pca else 'N/A'}")
    print(f"Z-score outlier threshold: {args.z_thresh}")
    print(f"Epochs:                    {args.epochs}")
    print(f"Batch size:                {args.batch_size}")
    print(f"Plot loss curve:           {'Yes' if args.plot_mse else 'No'}")
    print("---------------------------------------------")

    closure_columns = ['ThumbClosure', 'IndexClosure', 'MiddleClosure', 'ThumbAbduction']
    z_thresh = args.z_thresh

    # Load and preprocess data
    X, Y_scaled, scaler_y = load_data('dataset.csv', closure_columns)

    # Save scaler
    joblib.dump(scaler_y, "scaler_y.save")

    # Prepare PCA if needed
    if args.use_pca:
        pca = fit_pca(Y_scaled, pca_var=args.pca_variance)
        joblib.dump(pca, "reconstruction_pca.save")
        loss_type = mse_loss
        save_path = args.save_model if args.save_model else "FCNN_PCA_LOSS.pth"
    else:
        pca = None
        loss_type = weighted_mse_loss
        save_path = args.save_model if args.save_model else "FCNN.pth"

    output_dim = Y_scaled.shape[1]

    train_loader, test_loader = prepare_dataloaders(X, Y_scaled, batch_size=args.batch_size)

    model = HandPoseFCNN(input_dim=X.shape[1], output_dim=output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)

    train_losses, test_losses = train(
        model, train_loader, test_loader,
        optimizer, scheduler,
        loss_fn=loss_type,
        epochs=args.epochs,
        save_path=save_path,
        pca=pca
    )

    # Plotting
    if args.plot_mse:
        os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('MSE Loss Curve (PCA in loss)' if args.use_pca else 'Weighted MSE Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("plots/mse_loss_curve.png")
        plt.close()

    print("Training complete. Model saved to:", save_path)
    print(f"Best test loss: {min(test_losses):.6f}")
