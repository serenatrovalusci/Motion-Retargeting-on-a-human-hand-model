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
import argparse
import os


def generate_sincos_dataset(Y, fix_indices):
    mixed_columns = []
    for i in range(Y.shape[1]):
        if i in fix_indices:
            sin_val = np.sin(np.deg2rad(Y[:, [i]]))
            cos_val = np.cos(np.deg2rad(Y[:, [i]]))
            mixed_columns.extend([sin_val, cos_val])
        else:
            mixed_columns.append(Y[:, [i]])
    return np.concatenate(mixed_columns, axis=1)


def load_data(csv_path, closure_columns, fix_indices, z_thresh=2.5):
    print("Loading Dataset from csv file...")
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()
    data = data.iloc[:, 1:]  # remove the first column (time)
    joint_columns = [c for c in data.columns if c not in closure_columns]
    X = data[closure_columns].values
    Y = data[joint_columns].values

    X, Y = remove_outliers_zscore(X, Y, z_thresh)
    Y_sincos = generate_sincos_dataset(Y, fix_indices)

    scaler = StandardScaler().fit(Y_sincos)
    Y_scaled = scaler.transform(Y_sincos)

    print("All components:", Y_scaled.shape[1])

    return X, Y_scaled, scaler


def encode_with_autoencoder(Y_tensor, model_ae):
    return model_ae.encode(Y_tensor)



def remove_outliers_zscore(X, Y, threshold):
    z_scores = np.abs(zscore(Y, axis=0))
    mask = (z_scores < threshold).all(axis=1)
    return X[mask], Y[mask]


def prepare_dataloaders(X, Y, batch_size=64):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(Xtrain), torch.FloatTensor(Ytrain)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(Xtest), torch.FloatTensor(Ytest)), batch_size=batch_size)
    return train_loader, test_loader


def train(model, train_loader, test_loader, optimizer, scheduler, loss_fn, epochs=300, save_path="FCNN_VAE.pth", fix_indices=[]):
    train_losses, test_losses = [], []
    best_loss, best_model_state = float('inf'), None

    for epoch in range(epochs):
        model.train()
        train_loss = sum_step_loss(model, train_loader, loss_fn, optimizer, training=True, fix_indices=fix_indices)

        model.eval()
        test_loss = sum_step_loss(model, test_loader, loss_fn, training=False, fix_indices=fix_indices)

        scheduler.step(test_loss)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if test_loss < best_loss:
            best_loss = test_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, save_path)

        if epoch % 20 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d} | LR: {lr:.5f} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    if best_model_state:
        model.load_state_dict(best_model_state)

    return train_losses, test_losses


def sum_step_loss(model, loader, loss_fn, optimizer=None, training=False, fix_indices=None):
    total_loss = 0.0
    for xb, yb in loader:
        preds = model(xb)

        # Move constraints to correct device
        device = preds.device
        min_dev = min_vals.to(device)
        max_dev = max_vals.to(device)

        # 1. Original AE-encoded loss
        loss_main = loss_fn(preds, yb)

        # 2. Constraint loss
        loss_constraint = joint_constraint_loss(preds, min_dev, max_dev)

        # 3. Combine
        alpha = 1.0
        beta = 0.01  # You can tune this
        loss = alpha * loss_main + beta * loss_constraint

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def mse_loss_with_encoder(preds, targets, encoder_model):
    preds_encoded = encode_with_autoencoder(preds, encoder_model)
    targets_encoded = encode_with_autoencoder(targets, encoder_model)
    return torch.nn.functional.mse_loss(preds_encoded, targets_encoded)


def joint_constraint_loss(preds, min_tensor, max_tensor):
    lower_violation = torch.relu(min_tensor - preds)
    upper_violation = torch.relu(preds - max_tensor)
    return torch.mean(lower_violation**2 + upper_violation**2)


def weighted_mse_loss(preds, targets, fix_indices):
    weights = torch.ones_like(preds)
    mixed_col = 0
    for i in range(45):
        if i in fix_indices:
            weights[:, mixed_col: mixed_col + 1] = 2.0
            mixed_col += 2
        else:
            mixed_col += 1
    return torch.mean(weights * (preds - targets) ** 2)


def parse_args():
    parser = argparse.ArgumentParser(description="Train hand pose model with AE encoder-based loss.")
    parser.add_argument('--model', type=str, help='Choose the model: FCNN/Transformer')
    parser.add_argument('--z_thresh', type=float, default=2.5, help='Z-score threshold for outlier removal')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--save_model', type=str, default=None, help='Optional custom model save path')
    parser.add_argument('--plot_mse', action='store_true', help='Plot MSE loss curves')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("\nHelping prompt : ")
    print("python train.py --model FCNN --save_model FCNN_Example \n")

    print("----------- Training Configuration -----------")
    print(f"Z-score outlier threshold: {args.z_thresh}")
    print(f"Epochs:                    {args.epochs}")
    print(f"Batch size:                {args.batch_size}")
    print(f"Plot loss curve:           {'Yes' if args.plot_mse else 'No'}")
    print("---------------------------------------------")

    closure_columns = ['ThumbClosure', 'IndexClosure', 'MiddleClosure', 'ThumbAbduction']
    z_thresh = args.z_thresh
    fix_Indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 16, 17, 25, 26, 34, 43]

    X, Y, scaler_y = load_data('hand_dataset_all_fingers.csv', closure_columns, fix_Indices, z_thresh)

    joblib.dump(scaler_y, "scaler_AE.save")

    ae_model = HandPoseAE(input_dim=Y.shape[1], latent_dim=30)
    ae_model.load_state_dict(torch.load("HandPoseAE_2.pth"))

    ae_model.eval()
    loss_type = lambda preds, targets: mse_loss_with_encoder(preds, targets, ae_model)
    save_path = args.save_model if args.save_model else args.model + "_AE_Encoded.pth"

    output_dim = Y.shape[1]
    train_loader, test_loader = prepare_dataloaders(X, Y, batch_size=args.batch_size)

    if args.model == 'FCNN':
        model = HandPoseFCNN(input_dim=4, output_dim=output_dim)
    elif args.model == 'Transformer':
        model = HandPoseTransformer(input_dim=4, fix_indices=fix_Indices)

    print(f"\nThis is the model: \n\n{model}")

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)

    min_vals = torch.tensor(np.load("min_vals.npy"), dtype=torch.float32)
    max_vals = torch.tensor(np.load("max_vals.npy"), dtype=torch.float32)


    train_losses, test_losses = train(
        model, train_loader, test_loader,
        optimizer, scheduler,
        loss_fn=loss_type,
        epochs=args.epochs,
        save_path=save_path,
        fix_indices=fix_Indices
    )

    if args.plot_mse:
        os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('MSE Loss Curve (AE encoder-based)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("plots/mse_loss_curve.png")
        plt.close()

    print("Training complete. Model saved to:", save_path)
    print(f"Best test loss: {min(test_losses):.6f}")
