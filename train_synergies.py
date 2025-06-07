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
from datetime import datetime

def create_training_directory(base_dir="training_results/training_synergies_results"):
    # Create a unique directory name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_dir = os.path.join(base_dir, f"training_{timestamp}")
    os.makedirs(training_dir, exist_ok=True)
    return training_dir

def save_training_info(training_dir, info_dict):
    # Save training info to a text file
    info_file = os.path.join(training_dir, "training_info.txt")
    with open(info_file, 'w') as f:
        for key, value in info_dict.items():
            f.write(f"{key}: {value}\n")


def generate_sincos_dataset(Y, fix_indices):
    # In questo modo avremo un dataset tipo: [sin1, cos1, sin2, cos2, ang3, ang4, ...] 
    # l'ordine degli angoli non è mischiato, i primi sono del pollice, poi ci sono quelli dell'indice ecc...
    
    mixed_columns = []
    for i in range(Y.shape[1]):
        if i in fix_indices:
            sin_val = np.sin(np.deg2rad(Y[:, [i]]))
            cos_val = np.cos(np.deg2rad(Y[:, [i]]))
            mixed_columns.extend([sin_val, cos_val])
        else:
            mixed_columns.append(Y[:, [i]])
    return np.concatenate(mixed_columns, axis=1)


def load_data(csv_path, closure_columns, fix_indices):
    print("Loading Dataset from csv file...")
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()
    data = data.iloc[:, 1:] # remuve the first column (time)
    joint_columns = [c for c in data.columns if c not in closure_columns]
    X = data[closure_columns].values
    Y = data[joint_columns].values

    X, Y = remove_outliers_zscore(X, Y, z_thresh)
    Y_sincos = generate_sincos_dataset(Y, fix_indices)

    scaler = StandardScaler().fit(Y_sincos)
    Y_scaled = scaler.transform(Y_sincos)

    print("All components:", Y_scaled.shape[1])

    return X, Y_scaled, scaler


def generate_pca_dataset(Y, pca_var=0.995):
    pca = PCA(n_components=pca_var).fit(Y)
    Y_pca = pca.transform(Y)
    print(f"Numbers of PCA components related to a variance of {pca_var}: {Y_pca.shape[1]}\n")
    return pca, Y_pca.shape[1], Y_pca

def remove_outliers_zscore(X, Y, threshold):
    z_scores = np.abs(zscore(Y, axis=0))
    mask = (z_scores < threshold).all(axis=1)
    return X[mask], Y[mask]


def prepare_dataloaders(X, Y, batch_size=64):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(Xtrain), torch.FloatTensor(Ytrain)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(Xtest), torch.FloatTensor(Ytest)), batch_size=batch_size)
    return train_loader, test_loader


def train(model, train_loader, test_loader, optimizer, scheduler, loss_fn, epochs=300, save_path="FCNN_PCA.pth", fix_indices=[], pca=None):
    train_losses, test_losses = [], []
    best_loss, best_model_state = float('inf'), None

    for epoch in range(epochs):
        model.train()
        train_loss = compute_loss(model, train_loader, loss_fn, optimizer, training=True, fix_indices=fix_indices, pca=pca)

        model.eval()
        test_loss = compute_loss(model, test_loader, loss_fn, training=False, fix_indices=fix_indices, pca=pca)

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


def compute_loss(model, loader, loss_fn, optimizer=None, training=False, fix_indices=None, pca=None):
    total_loss = 0.0
    for xb, yb in loader:
        preds = model(xb)

        if pca is not None:
            loss = loss_fn(preds, yb)
        else:
            loss = loss_fn(preds, yb, fix_indices)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def mse_loss(preds, targets):
    return torch.nn.functional.mse_loss(preds, targets)


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
    parser = argparse.ArgumentParser(description="Train hand pose model with or without PCA.")
    parser.add_argument('--model', type=str, default="Transformer",help='Choose the model: FCNN/Transformer')
    parser.add_argument('--pca_variance', type=float, default=1, help='If want to use PCA, define the variance')
    parser.add_argument('--z_thresh', type=float, default=2.5, help='Z-score threshold for outlier removal')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--save_model', type=str, default=None, help='Optional custom model save path')
    parser.add_argument('--plot_mse', action='store_true', help='Plot MSE loss curves')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("\nHelping prompt : ")
    print("python train_synergies.py --model FCNN --pca_variance 0.995 \n")

    print("----------- Training Configuration -----------")
    print(f"Using PCA in loss:         {'Yes' if args.pca_variance < 1.0 else 'No'}")
    print(f"PCA explained variance:    {args.pca_variance if args.pca_variance < 1.0 else 'N/A'}")
    print(f"Z-score outlier threshold: {args.z_thresh}")
    print(f"Epochs:                    {args.epochs}")
    print(f"Batch size:                {args.batch_size}")
    print(f"Plot loss curve:           {'Yes'}") #if args.plot_mse else 'No'
    print("---------------------------------------------\n")

    # Create training directory
    training_dir = create_training_directory()
    print(f"All training results will be saved in: {training_dir}")

    closure_columns = ['ThumbClosure', 'IndexClosure', 'MiddleClosure', 'ThumbAbduction']
    z_thresh = args.z_thresh
    # angoli fixati per una performance migliore
    fix_Indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 16, 17, 25, 26, 34, 43] # all the thumb angles(9) + 4 index angles + 2 middle angles
    #fix_Indices = []
    X, Y, scaler_y = load_data('dataset/hand_dataset_all_fingers.csv', closure_columns, fix_Indices)

    # Prepare training info dictionary
    training_info = {
        "Model": args.model,
        "PCA Variance": args.pca_variance if args.pca_variance < 1.0 else "No PCA",
        "Z-score Threshold": args.z_thresh,
        "Epochs": args.epochs,
        "Batch Size": args.batch_size,
        "Fixed Indices": fix_Indices,
        "Number of Fixed Indices": len(fix_Indices),
        "Input Dimension": 4,
        "Output Dimension": Y.shape[1],
        "Dataset": 'dataset/hand_dataset_all_fingers.csv',
        "Closure Columns": closure_columns,
        "Training Start Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    if args.pca_variance < 1.0:
        print("You have selected PCA\n")
        pca, N, Y = generate_pca_dataset(Y, pca_var=args.pca_variance)
        loss_type = mse_loss
        model_save_path = os.path.join(training_dir, args.save_model if args.save_model else f"{args.model}_{N}_PCA.pth")
        training_info["PCA Components"] = N
        training_info["Explained Variance"] = args.pca_variance
    else:
        print("You have selected NO PCA\n")
        pca = None
        loss_type = weighted_mse_loss
        model_save_path = os.path.join(training_dir, args.save_model if args.save_model else f"{args.model}.pth")
        training_info["PCA Components"] = "N/A"

    output_dim = Y.shape[1]
    training_info["Final Output Dimension"] = output_dim

    # Save scaler in the training directory
    scaler_path = os.path.join(training_dir, "scaler.save")
    joblib.dump(scaler_y, scaler_path)
    training_info["Scaler Path"] = scaler_path

    # Save pca in the training directory
    pca_path = os.path.join(training_dir, "pca.save")
    joblib.dump(pca, pca_path)

    train_loader, test_loader = prepare_dataloaders(X, Y, batch_size=args.batch_size)

    if args.model == 'FCNN':
        model = HandPoseFCNN(input_dim=4, output_dim=output_dim)
    elif args.model == 'Transformer':
        model = HandPoseTransformer(input_dim=4, fix_indices=fix_Indices, pca_dim=output_dim if args.pca_variance < 1.0 else 0)

    print("This is the dataset:", training_info["Dataset"])
    
    print(f"\nThis is the model: \n\n{model}\n")
    training_info["Model Architecture"] = str(model)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
    
    start_time = datetime.now()
    train_losses, test_losses = train(
        model, train_loader, test_loader,
        optimizer, scheduler,
        loss_fn=loss_type,
        epochs=args.epochs,
        save_path=model_save_path,
        fix_indices=fix_Indices,
        pca=pca 
    )
    duration = datetime.now() - start_time
    # Update training info with final results
    training_info["Training Start Time"] = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    training_info["Training Duration"] = str(duration)
    training_info["Best Test Loss"] = min(test_losses)
    training_info["Final Learning Rate"] = optimizer.param_groups[0]['lr']
    training_info["Model Save Path"] = model_save_path

    # Save training info
    save_training_info(training_dir, training_info)

    
    plot_dir = os.path.join(training_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('MSE Loss Curve (PCA)' if args.pca_variance < 1.0 else 'Weighted MSE Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "mse_loss_curve.png")
    plt.savefig(plot_path)
    plt.close()
    training_info["Plot Path"] = plot_path

    print("\nTraining complete. All results saved in:", training_dir)
    print(f"Best test loss: {min(test_losses):.6f}")
    print("\nTraining Summary:")
    for key, value in training_info.items():
        print(f"{key}: {value}")