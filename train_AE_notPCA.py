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
from datetime import datetime, timedelta

def create_training_directory(base_dir="training_results/training_latentspace_results"):
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

def find_min_max(Y):

    min_vals = torch.tensor(np.min(Y, axis=0), dtype=torch.float32)
    max_vals = torch.tensor(np.max(Y, axis=0), dtype=torch.float32)
    return min_vals, max_vals

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


def train(model, train_loader, test_loader, optimizer, scheduler, loss_fn, min_vals, max_vals,epochs=300, save_path="FCNN_VAE.pth"): 
    train_losses, test_losses = [], []
    best_loss, best_model_state = float('inf'), None

    for epoch in range(epochs):
        model.train()
        train_loss = sum_step_loss(model, train_loader, loss_fn, min_vals, max_vals, optimizer, training=True)

        model.eval()
        test_loss = sum_step_loss(model, test_loader, loss_fn, min_vals, max_vals, optimizer, training=False)

        scheduler.step(test_loss)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if test_loss < best_loss:
            best_loss = test_loss
            best_model_state = model.state_dict()
            #torch.save(best_model_state, save_path)

        if epoch % 20 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d} | LR: {lr:.5f} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    if best_model_state:
        model.load_state_dict(best_model_state)

    return train_losses, test_losses


def sum_step_loss(model, loader, loss_fn, min_vals, max_vals, optimizer=None, training=False):
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
        loss = loss_main + loss_constraint

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
    parser.add_argument('--model_reduction', type=str, help='Choose the model: AE/VAE')
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

    # Create training directory
    training_dir = create_training_directory()
    print(f"All training results will be saved in: {training_dir}")

    closure_columns = ['ThumbClosure', 'IndexClosure', 'MiddleClosure', 'ThumbAbduction']
    z_thresh = args.z_thresh
    fix_Indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 16, 17, 25, 26, 34, 43]
    latent_dimension = 30;
    X, Y, scaler_y = load_data('dataset/hand_dataset_all_fingers.csv', closure_columns, fix_Indices, z_thresh)
    
    min_vals, max_vals = find_min_max(Y)
    training_info = {
    #"Model": args.model,
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
    
    #joblib.dump(scaler_y, "scaler_AE.save")
    if args.model_reduction == 'AE':
        model_reduction = HandPoseAE(input_dim=Y.shape[1], latent_dim = latent_dimension)
        model_save_path = os.path.join(training_dir, args.save_model if args.save_model else f"{args.model}_{latent_dimension}_latentdim.pth")
        #model_reduction.load_state_dict(torch.load("HandPoseAE_2.pth"))   
    elif args.model_reduction == 'VAE':
        model_reduction = HandPoseVAE(input_dim=Y.shape[1], latent_dim=latent_dimension)
        model_save_path = os.path.join(training_dir, args.save_model if args.save_model else f"{args.model}_{latent_dimension}_latentdim.pth")
        #model_reduction.load_state_dict(torch.load("HandPoseVAE_2.pth"))
    
    model_reduction.eval()
    loss_type = lambda preds, targets: mse_loss_with_encoder(preds, targets, model_reduction)
    #save_path = args.save_model if args.save_model else args.model + f"_{args.model_reduction}.pth"
    save_path = model_save_path;

    training_info["Latent Space Dimension"] = latent_dimension
    training_info["Model Reduction Type"] = args.model  # AE or VAE

    output_dim = Y.shape[1]
    train_loader, test_loader = prepare_dataloaders(X, Y, batch_size=args.batch_size)
    training_info["Final Output Dimension"] = output_dim

    if args.model == 'FCNN':
        model = HandPoseFCNN(input_dim=4, output_dim=output_dim)
    elif args.model == 'Transformer':
        model = HandPoseTransformer(input_dim=4, fix_indices=fix_Indices)
    
    print("This is the dataset:", training_info["Dataset"])

    print(f"\nThis is the model: \n\n{model}")
    training_info["Regression Model Architecture"] = str(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)

    train_losses, test_losses = train(
        model, train_loader, test_loader,
        optimizer, scheduler,
        loss_fn=loss_type,
        min_vals=min_vals,
        max_vals=max_vals,
        epochs=args.epochs,
        save_path=save_path
    )
    # Update training info with final results
    training_info["Training Start Time"] = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    training_info["Training Duration"] = str(datetime.now() - (datetime.now()))
    training_info["Best Test Loss"] = min(test_losses)
    training_info["Final Learning Rate"] = optimizer.param_groups[0]['lr']
    training_info["Model Save Path"] = model_save_path

    # Save training info
    save_training_info(training_dir, training_info)

    if args.plot_mse:

        plot_dir = os.path.join(training_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('MSE Loss Curve (AE encoder-based)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, "mse_loss_curve_latentspace.png")
        #plt.savefig("plots/mse_loss_curve.png")
        plt.savefig(plot_path)
        plt.close()
        training_info["Plot Path"] = plot_path

    print("Training complete. Model saved to:", save_path)
    print(f"Best test loss: {min(test_losses):.6f}")
    print("\nTraining Summary:")
    for key, value in training_info.items():
        print(f"{key}: {value}")
