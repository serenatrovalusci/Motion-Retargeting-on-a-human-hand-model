# mygrid.py
import wandb
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


MODEL_PATH = "FCNN_PCA_grid.pth"
CLOSURE_COLS = ['ThumbClosure', 'IndexClosure', 'MiddleClosure', 'ThumbAbduction']

DATA = pd.read_csv("dataset.csv")
DATA.columns = DATA.columns.str.strip()
Y_RAW = DATA[[c for c in DATA.columns if c not in CLOSURE_COLS]].values
X_RAW = DATA[CLOSURE_COLS].values

def generate_sincos_dataset(y):
    thumb = np.concatenate([np.sin(np.deg2rad(y[:, :9])), np.cos(np.deg2rad(y[:, :9]))], axis=1)
    fix_indices = [13, 14, 16, 17, 25, 26]
    fix = np.concatenate([np.sin(np.deg2rad(y[:, fix_indices])), np.cos(np.deg2rad(y[:, fix_indices]))], axis=1)
    raw_indices = [i for i in range(27) if i not in list(range(9)) + fix_indices]
    raw = y[:, raw_indices]
    return np.concatenate([thumb, fix, raw], axis=1)

Y_SINCOS_RAW = generate_sincos_dataset(Y_RAW)
SCALER = StandardScaler().fit(Y_SINCOS_RAW)
Y_SCALED_RAW = SCALER.transform(Y_SINCOS_RAW)
PCA_MODEL = PCA(n_components=0.995).fit(Y_SCALED_RAW) 
Y_PCA_GLOBAL = PCA_MODEL.transform(Y_SCALED_RAW)

print(f"PCA components: {Y_PCA_GLOBAL.shape[1]}")
joblib.dump(SCALER, "reconstruction_scaler_grid.save")
joblib.dump(PCA_MODEL, "reconstruction_pca_grid.save")



def prepare_dataloaders(X, Y, batch_size=64):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(Xtrain), torch.FloatTensor(Ytrain)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(Xtest), torch.FloatTensor(Ytest)), batch_size=batch_size)
    return train_loader, test_loader

def train_epoch(model, train_loader, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        preds = model(xb)
        loss = loss_fn(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(train_loader.dataset)

def evaluate_epoch(model, test_loader, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb)
            loss = loss_fn(preds, yb)
            total_loss += loss.item() * xb.size(0)
    return total_loss / len(test_loader.dataset)

def mse_loss(preds, targets):
    return torch.nn.functional.mse_loss(preds, targets)

# This is the function that wandb.agent will call for each run
def train_sweep_run():
    # Initialize wandb for this specific run
    with wandb.init() as run: # wandb.init() will pick up config from the agent
        config = run.config
        print(f'Current configuration: {config}')

     
        train_loader, test_loader = prepare_dataloaders(X_RAW, Y_PCA_GLOBAL, batch_size=config.batch_size)

        # Initialize model with hyperparameters from config
     
        model = HandPoseFCNN(input_dim=X_RAW.shape[1],
                             output_dim=Y_PCA_GLOBAL.shape[1],
                             dropout=config.dropout) # Corrected typo

        # Optimizer with hyperparameters from config
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=False)

        # Training loop
        best_loss = float('inf')
        epochs = 300 

        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, mse_loss, optimizer)
            test_loss = evaluate_epoch(model, test_loader, mse_loss)
            scheduler.step(test_loss)

            wandb.log({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss, "lr": optimizer.param_groups[0]['lr']})

            if test_loss < best_loss:
                best_loss = test_loss
                
                # torch.save(model.state_dict(), f"best_model_run_{run.id}.pth")

            if epoch % 50 == 0:
                print(f"Epoch {epoch:3d} | LR: {optimizer.param_groups[0]['lr']:.5f} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

        print(f'Best Test Loss for this run: {best_loss:.5f}')
        wandb.log({"best_test_loss": best_loss}) # This is what the sweep will track

if __name__ == "__main__":
    sweep_config = {
        'method': 'grid',
        'metric': {'name': 'best_test_loss', 'goal': 'minimize'},
        'parameters': {
            'lr': {'values': [0.01, 0.005, 0.001]},
            'dropout': {'values': [0.0, 0.3, 0.6, 0.7]}, 
            'batch_size': {'values': [64, 128]}, 
            'weight_decay': {'values': [1e-4, 1e-5]}
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_config, project="FirstTry_WeartGridSearch")
    print(f"Sweep ID: {sweep_id}")
    print(f"Sweep URL: https://wandb.ai/primiceri-2021543-sapienza-universit-di-roma/WeartGridSearch/sweeps/{sweep_id}")


    # Calculate total number of runs for a grid search
    num_runs = 1
    for param in sweep_config['parameters']:
        if 'values' in sweep_config['parameters'][param]:
            num_runs *= len(sweep_config['parameters'][param]['values'])
    print(f"Grid search will perform {num_runs} runs.")

    # The agent will call train_sweep_run for each combination of parameters
    wandb.agent(sweep_id, function=train_sweep_run, count=num_runs)

    torch.cuda.empty_cache()